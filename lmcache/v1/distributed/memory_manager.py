# SPDX-License-Identifier: Apache-2.0

# Standard
import shutil
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class ShmPoolInfo:
    """Shared-memory pool information for the L1 memory manager.

    Attributes:
        shm_name: Name of the POSIX shared memory segment (without leading
            slash), or an empty string when shm is disabled.
        pool_size: Total size of the L1 pool in bytes.
        shm_enabled: Whether the shared memory pool is active.
        base_ptr: Virtual address of the first byte of the pool, or 0 when
            shm is disabled.  Workers compute slot offsets by subtracting
            this value from the per-slot pointer.
    """

    shm_name: str
    pool_size: int
    shm_enabled: bool
    base_ptr: int


# HELPER FUNCTIONS
def _check_shm_capacity(required_bytes: int) -> bool:
    """Return True if /dev/shm has enough free space, False otherwise.

    Args:
        required_bytes: Minimum number of free bytes needed.

    Returns:
        True when /dev/shm has at least ``required_bytes`` available.
    """
    try:
        shm_stat = shutil.disk_usage("/dev/shm")
    except FileNotFoundError:
        logger.warning(
            "/dev/shm does not exist on this system. "
            "SHM L1 pool disabled; falling back to CPU bounce-buffer path."
        )
        return False

    if shm_stat.free < required_bytes:
        logger.warning(
            "Insufficient /dev/shm space: need %.1f GiB, available %.1f GiB. "
            "SHM L1 pool disabled; falling back to CPU bounce-buffer path. "
            "Increase /dev/shm size via your container runtime or system "
            "configuration (e.g. '--shm-size' for Docker, "
            "'emptyDir.medium: Memory' for Kubernetes).",
            required_bytes / 2**30,
            shm_stat.free / 2**30,
        )
        return False

    return True


def create_memory_allocator(
    config: L1MemoryManagerConfig,
    shm_name: Optional[str] = None,
) -> MemoryAllocatorInterface:
    """
    Create a memory allocator based on the provided configuration.

    Args:
        config (L1MemoryManagerConfig): Configuration for the memory manager.
        shm_name (Optional[str]): When non-None, the allocator will back its
            buffer with this POSIX shared memory segment name.  Only
            MixedMemoryAllocator supports shm; LazyMemoryAllocator ignores
            this argument.

    Returns:
        MemoryAllocatorInterface: An instance of a memory allocator.
    """
    if config.use_lazy:
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
            "align bytes is %d bytes, shm_name=%s",
            config.size_in_bytes,
            config.align_bytes,
            shm_name,
        )
        kwargs: dict = {"align_bytes": config.align_bytes}
        if shm_name is not None:
            kwargs["shm_name"] = shm_name
        return MixedMemoryAllocator(
            config.size_in_bytes,
            **kwargs,
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
        self._size_in_bytes = config.size_in_bytes
        self._align_bytes = config.align_bytes

        # --- Shared-memory pool setup ---
        # Determine whether shm is requested and feasible.
        shm_name_to_use: Optional[str] = None
        if config.use_shm_l1_pool:
            if _check_shm_capacity(config.size_in_bytes):
                shm_name_to_use = config.shm_name
                if config.use_lazy:
                    logger.warning(
                        "use_shm_l1_pool=True is incompatible with use_lazy=True. "
                        "SHM path requires MixedMemoryAllocator; "
                        "disabling lazy allocation for the shm pool."
                    )
                logger.info(
                    "SHM L1 pool enabled: name='%s', size=%.1f GiB",
                    shm_name_to_use,
                    config.size_in_bytes / 2**30,
                )
            # If capacity check failed, shm_name_to_use stays None → disabled.

        self._shm_enabled: bool = shm_name_to_use is not None
        self._shm_name: str = shm_name_to_use or ""

        # When shm is requested, override use_lazy to False so we always
        # get a MixedMemoryAllocator (which supports shm_name).
        effective_config = config
        if self._shm_enabled and config.use_lazy:
            from dataclasses import replace

            effective_config = replace(config, use_lazy=False)

        self._allocator = create_memory_allocator(
            effective_config,
            shm_name=shm_name_to_use,
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

    def get_shm_pool_info(self) -> ShmPoolInfo:
        """Return shared-memory pool metadata for this L1 manager.

        Returns:
            ShmPoolInfo: Contains the shm segment name, total pool size in
            bytes, whether shm is enabled, and the base virtual-address
            pointer of the pool buffer.  When shm is disabled
            ``shm_enabled`` is False and ``base_ptr`` is 0.
        """
        if not self._shm_enabled:
            return ShmPoolInfo(
                shm_name="",
                pool_size=self._size_in_bytes,
                shm_enabled=False,
                base_ptr=0,
            )

        # shm is backed by MixedMemoryAllocator; get_l1_memory_desc() is
        # available for that allocator type.
        desc = self.get_l1_memory_desc()
        return ShmPoolInfo(
            shm_name=self._shm_name,
            pool_size=self._size_in_bytes,
            shm_enabled=True,
            base_ptr=desc.ptr,
        )

    def compute_shm_slot(self, memory_obj: MemoryObj) -> tuple[int, int]:
        """Compute the (offset, byte_length) of a MemoryObj within the shm pool.

        The offset is the byte distance from the start of the shared-memory
        segment to the first byte of ``memory_obj``'s data.  Workers use this
        offset to construct a zero-copy tensor view via ``torch.frombuffer`` on
        the mapped shm buffer.

        Args:
            memory_obj: A MemoryObj previously allocated from this L1 pool.

        Returns:
            tuple[int, int]: ``(shm_offset, byte_length)`` where
            ``shm_offset`` is the byte offset from the pool base and
            ``byte_length`` is the logical size of the object in bytes.

        Raises:
            RuntimeError: If shm is not enabled or the allocator type does not
                support offset computation.
            ValueError: If ``memory_obj.raw_tensor`` is None.
        """
        if not self._shm_enabled:
            raise RuntimeError(
                "compute_shm_slot called but shm L1 pool is not enabled"
            )
        if not isinstance(self._allocator, MixedMemoryAllocator):
            raise RuntimeError(
                "compute_shm_slot is only supported for MixedMemoryAllocator"
            )
        raw = memory_obj.raw_tensor
        if raw is None:
            raise ValueError("memory_obj.raw_tensor is None; object may be invalid")
        base_ptr = self._allocator.buffer.data_ptr()
        offset = raw.data_ptr() - base_ptr
        byte_length = memory_obj.get_size()
        return offset, byte_length

    def close(self) -> None:
        """
        Close the memory manager and release all resources.
        """
        self._allocator.close()

    # Debugging APIs
    def memcheck(self):
        return self._allocator.memcheck()
