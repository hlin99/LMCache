# SPDX-License-Identifier: Apache-2.0

# Standard
import fcntl
import os
import shutil

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryObj,
    MixedMemoryAllocator,
)

logger = init_logger(__name__)


# HELPER FUNCTIONS
def _check_shm_capacity(required_bytes: int) -> None:
    """Verify ``/dev/shm`` has sufficient free space.

    Args:
        required_bytes: Number of bytes the L1 pool requires.

    Raises:
        RuntimeError: If ``/dev/shm`` does not have enough free space.
    """
    shm_stat = shutil.disk_usage("/dev/shm")
    if shm_stat.free < required_bytes:
        size_gib = max(1, (required_bytes + 2**30 - 1) // 2**30)
        raise RuntimeError(
            f"Insufficient /dev/shm space: need {required_bytes / 2**30:.1f} GiB, "
            f"available {shm_stat.free / 2**30:.1f} GiB. "
            f"Use 'docker run --shm-size={size_gib}g' or "
            f"set Kubernetes emptyDir.medium=Memory to increase /dev/shm size."
        )


def _unlink_stale_shm(shm_name: str) -> None:
    """Remove stale shared-memory segments with the ``lmcache_l1_pool_`` prefix.

    Scans ``/dev/shm`` for all files matching the ``lmcache_l1_pool_*``
    prefix and removes only those that are **not** held by a running
    server.  Each active server holds an exclusive ``flock`` on its SHM
    file for its entire lifetime; a non-blocking ``flock`` attempt here
    distinguishes stale (lock succeeds → safe to delete) from live (lock
    fails → another server is using it, skip).

    Args:
        shm_name: POSIX shared-memory name for the new pool (e.g.
            ``"lmcache_l1_pool_12345"``).
    """
    shm_dir = "/dev/shm"
    prefix = "lmcache_l1_pool_"
    try:
        entries = os.listdir(shm_dir)
    except OSError as exc:
        logger.warning(
            "Cannot list %s; skipping stale SHM cleanup: %s", shm_dir, exc
        )
        return

    for entry in entries:
        if not entry.startswith(prefix):
            continue
        shm_path = os.path.join(shm_dir, entry)
        try:
            fd = os.open(shm_path, os.O_RDWR)
            try:
                # Non-blocking exclusive lock: succeeds only if no
                # other process holds a lock (i.e. the segment is stale).
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Lock acquired → stale segment, safe to remove.
                os.unlink(shm_path)
                logger.info(
                    "Removed stale shared-memory segment: %s", shm_path
                )
            except OSError:
                # Lock failed → another server is actively using it.
                logger.debug(
                    "SHM segment in use by another server, skipping: %s",
                    shm_path,
                )
            finally:
                os.close(fd)
        except OSError:
            logger.warning(
                "Failed to check/remove stale SHM segment: %s",
                shm_path,
                exc_info=True,
            )


def create_memory_allocator(config: L1MemoryManagerConfig) -> MemoryAllocatorInterface:
    """
    Create a memory allocator based on the provided configuration.

    Args:
        config (L1MemoryManagerConfig): Configuration for the memory manager.

    Returns:
        MemoryAllocatorInterface: An instance of a memory allocator.

    Raises:
        RuntimeError: If ``config.shm_name`` is set but ``/dev/shm`` has
            insufficient free space.
    """
    if config.shm_name:
        # Clean stale segments first to free space for the capacity check
        _unlink_stale_shm(config.shm_name)
        _check_shm_capacity(config.size_in_bytes)

    if config.use_lazy:
        if config.shm_name:
            logger.warning(
                "LazyMemoryAllocator does not support named shared memory; "
                "shm_name=%r will be ignored. The lazy allocator is used "
                "only in CUDA environments where workers use GPU IPC "
                "instead of SHM.",
                config.shm_name,
            )
        logger.debug(
            "use lazy memory allocator, init size is %d bytes, "
            "final size is %d bytes, align bytes is %d bytes",
            config.init_size_in_bytes,
            config.size_in_bytes,
            config.align_bytes,
        )
        return LazyMemoryAllocator(
            config.init_size_in_bytes, config.size_in_bytes, config.align_bytes
        )
    else:
        logger.debug(
            "use mixed memory allocator, total size is %d bytes, "
            "align bytes is %d bytes",
            config.size_in_bytes,
            config.align_bytes,
        )
        return MixedMemoryAllocator(
            config.size_in_bytes,
            align_bytes=config.align_bytes,
            shm_name=config.shm_name,
        )


# MAIN CLASS
class L1MemoryManager:
    """
    L1MemoryManager manages the allocation and deallocation of L1 memory.

    Observability metrics to emit:
    1. Memory usage
    2. Active allocations
    """

    def __init__(self, config: L1MemoryManagerConfig):
        self._allocator = create_memory_allocator(config)
        self._size_in_bytes = config.size_in_bytes
        self._align_bytes = config.align_bytes
        # shm_name is only meaningful when MixedMemoryAllocator is used
        # (non-lazy path). For LazyMemoryAllocator (CUDA), it's cleared.
        if config.use_lazy:
            self._shm_name = ""
        else:
            self._shm_name = config.shm_name

        # Hold an exclusive flock on the SHM file for the server's
        # lifetime.  _unlink_stale_shm uses a non-blocking flock to
        # detect whether a segment is still in use; this lock prevents
        # another server's cleanup from deleting our segment.  The
        # kernel releases the lock automatically on process crash /
        # SIGKILL, so stale segments become unlocked and removable.
        self._shm_lock_fd: int | None = None
        if self._shm_name:
            shm_path = f"/dev/shm/{self._shm_name}"
            try:
                fd = os.open(shm_path, os.O_RDWR)
                fcntl.flock(fd, fcntl.LOCK_EX)
                self._shm_lock_fd = fd
            except OSError:
                logger.warning(
                    "Failed to acquire flock on SHM segment: %s",
                    shm_path,
                    exc_info=True,
                )

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """
        Allocate memory objects based on the provided layout description and count.
        This function should be thread-safe

        Args:
            layout_desc (MemoryLayoutDesc): Description of the memory layout.
            count (int): Number of memory objects to allocate.

        Returns:
            tuple[L1Error, list[MemoryObj]]: Error code and list of
            allocated memory objects.
            Error code will be `L1Error.OUT_OF_MEMORY` if allocation
            fails; otherwise, it will be `L1Error.SUCCESS`.

        Note:
            If the allocation fails, the memory object list will be empty.
        """
        objects = self._allocator.batched_allocate(
            layout_desc.shapes, layout_desc.dtypes, count
        )
        if objects is None:
            return L1Error.OUT_OF_MEMORY, []
        return L1Error.SUCCESS, objects

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """
        Free the provided memory objects.
        This function should be thread-safe.

        Args:
            mem_objs (list[MemoryObj]): List of memory objects to free.

        Returns:
            L1Error: Error code indicating the result of the operation.
            It will be `L1Error.SUCCESS` if the operation succeeds.
        """
        self._allocator.batched_free(mem_objs)
        return L1Error.SUCCESS

    def get_memory_usage(self) -> tuple[int, int]:
        """
        Get the current memory usage. This function will mainly be used to support
        eviction decision.

        Returns:
            tuple[int, int]: A tuple containing used memory in bytes and total memory
            in bytes.

        Note:
            In the future, we may want to make a "callback" based mechanism to
            trigger eviction when the memory usage reaches a watermark.
        """

        # HACK: now trying to read this from the address manager in a ad-hoc
        # manner
        def get_address_manager(allocator: MemoryAllocatorInterface):
            if isinstance(allocator, MixedMemoryAllocator) and hasattr(
                allocator.pin_allocator, "address_manager"
            ):
                return allocator.pin_allocator.address_manager
            elif isinstance(allocator, LazyMemoryAllocator):
                return allocator.get_address_manager()
            else:
                raise NotImplementedError(
                    "get_memory_usage is not implemented for this allocator type."
                )

        address_manager = get_address_manager(self._allocator)
        free_size = address_manager.get_free_size()
        total_size = address_manager.get_heap_size()
        used_size = total_size - free_size
        return used_size, total_size

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """
        Return an L1MemoryDesc describing the underlying memory buffer.

        Returns:
            L1MemoryDesc: Pointer, size, and alignment of the L1 buffer.

        Raises:
            NotImplementedError: If the allocator type does not support this operation.
        """
        if isinstance(self._allocator, MixedMemoryAllocator):
            buffer = self._allocator.buffer
        elif isinstance(self._allocator, LazyMemoryAllocator):
            # TODO(ApostaC): need to test if the RDMA registration works
            # before the lazy expansion is finished
            buffer = self._allocator.get_underlying_buffer()
        else:
            raise NotImplementedError(
                "get_l1_memory_desc is not implemented for this allocator type."
            )
        return L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=self._size_in_bytes,
            align_bytes=self._align_bytes,
        )

    def get_shm_pool_info(self) -> dict[str, object]:
        """Return shared-memory pool metadata for worker attachment.

        Returns:
            A dictionary with ``"shm_name"`` (str) and ``"pool_size"``
            (int, bytes).  ``shm_name`` is empty when the pool is not
            backed by named shared memory.
        """
        return {
            "shm_name": self._shm_name,
            "pool_size": self._size_in_bytes,
        }

    def close(self) -> None:
        """
        Close the memory manager and release all resources.

        If the pool is backed by named shared memory, the flock is
        released and the segment is unlinked so that the kernel can
        reclaim the pages once all processes detach.
        """
        self._allocator.close()
        # Release the flock before unlinking so _unlink_stale_shm
        # (called by other servers) can detect this segment as stale.
        if self._shm_lock_fd is not None:
            try:
                os.close(self._shm_lock_fd)
            except OSError:
                pass
            self._shm_lock_fd = None
        if self._shm_name:
            _unlink_stale_shm(self._shm_name)

    # Debugging APIs
    def memcheck(self):
        return self._allocator.memcheck()
