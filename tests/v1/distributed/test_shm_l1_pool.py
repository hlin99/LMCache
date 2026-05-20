# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock
import mmap
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.memory_manager import create_memory_allocator
from lmcache.v1.memory_management import MixedMemoryAllocator
from lmcache.v1.multiprocess.non_gpu_context import NonGpuContextMetadata
from lmcache.v1.multiprocess.non_gpu_context import create_non_gpu_context
from lmcache.v1.multiprocess.non_gpu_context_pickle import NonGpuContextPickle
from lmcache.v1.multiprocess.non_gpu_context_shm import NonGpuContextShm
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)


class _CompletedFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):  # noqa: ARG002
        return self._value


def _create_shm_file(shm_name: str, size: int) -> str:
    path = os.path.join("/dev/shm", shm_name.lstrip("/"))
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    os.close(fd)
    return path


def test_shm_segment_creation_and_cleanup() -> None:
    shm_name = f"lmcache_l1_pool_test_{os.getpid()}"
    cfg = L1MemoryManagerConfig(
        size_in_bytes=1024 * 1024,
        use_lazy=False,
        shm_name=shm_name,
    )
    allocator = create_memory_allocator(cfg)
    assert isinstance(allocator, MixedMemoryAllocator)
    assert allocator.shm_name == shm_name
    shm_path = os.path.join("/dev/shm", shm_name)
    assert os.path.exists(shm_path)
    allocator.close()
    assert not os.path.exists(shm_path)


def test_non_gpu_context_shm_tensor_view_from_buffer() -> None:
    shm_name = f"lmcache_test_view_{os.getpid()}"
    shm_path = _create_shm_file(shm_name, 4096)
    try:
        with open(shm_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 4096, access=mmap.ACCESS_WRITE)
            src = torch.arange(8, dtype=torch.float32).reshape(2, 4)
            mm[: src.numel() * src.element_size()] = src.numpy().tobytes()
            mm.close()

        context = NonGpuContextShm(
            metadata=NonGpuContextMetadata(
                layout_desc=MemoryLayoutDesc(
                    shapes=[torch.Size([2, 4])],
                    dtypes=[torch.float32],
                ),
                block_size=1,
                use_mla=False,
            ),
            mq_client=MagicMock(),
            mq_timeout=1.0,
            shm_name=shm_name,
            pool_size=4096,
        )
        try:
            view = context._make_tensor_view(
                offset=0,
                length=src.numel() * src.element_size(),
                shape=[2, 4],
                dtype_str="float32",
            )
            assert torch.equal(view, src)
        finally:
            context.close()
    finally:
        if os.path.exists(shm_path):
            os.unlink(shm_path)


def test_non_gpu_context_shm_store_retrieve_flow_with_mocked_mq() -> None:
    shm_name = f"lmcache_test_flow_{os.getpid()}"
    shm_path = _create_shm_file(shm_name, 4096)
    slots = [
        {
            "offset": 0,
            "length": 16,
            "shape": [2, 2],
            "dtype": "float32",
        }
    ]

    mq_client = MagicMock()

    def _submit_request(req_type, payload, response_cls):  # noqa: ARG001
        if req_type == RequestType.PREPARE_STORE:
            return _CompletedFuture(PrepareStoreResponse(context={"slots": slots}))
        if req_type == RequestType.COMMIT_STORE:
            _, _, commit_cpu_data = payload
            assert commit_cpu_data == b""
            return _CompletedFuture(True)
        if req_type == RequestType.PREPARE_RETRIEVE:
            return _CompletedFuture(
                PrepareRetrieveResponse(
                    success=True, data=b"", context={"slots": slots}
                )
            )
        if req_type == RequestType.COMMIT_RETRIEVE:
            return _CompletedFuture(True)
        raise AssertionError(f"Unexpected request type: {req_type}")

    mq_client.submit_request.side_effect = _submit_request

    context = NonGpuContextShm(
        metadata=NonGpuContextMetadata(
            layout_desc=MemoryLayoutDesc(
                shapes=[torch.Size([2, 2])],
                dtypes=[torch.float32],
            ),
            block_size=1,
            use_mla=False,
        ),
        mq_client=mq_client,
        mq_timeout=1.0,
        shm_name=shm_name,
        pool_size=4096,
    )
    try:
        store_views = context.prepare_store(key="k", instance_id=1)
        assert store_views is not None
        store_views[0].copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        )
        assert context.commit_store("k", 1, store_views)

        retrieve_views = context.prepare_retrieve(key="k", instance_id=1)
        assert retrieve_views is not None
        assert torch.equal(
            retrieve_views[0],
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        )
        assert context.commit_retrieve("k", 1)
    finally:
        context.close()
        if os.path.exists(shm_path):
            os.unlink(shm_path)


def test_non_gpu_context_shm_init_raises_when_segment_missing() -> None:
    with pytest.raises(FileNotFoundError):
        NonGpuContextShm(
            metadata=NonGpuContextMetadata(
                layout_desc=MemoryLayoutDesc(
                    shapes=[torch.Size([2, 2])],
                    dtypes=[torch.float32],
                ),
                block_size=1,
                use_mla=False,
            ),
            mq_client=MagicMock(),
            mq_timeout=1.0,
            shm_name="lmcache_missing_shm_segment",
            pool_size=4096,
        )


def test_create_non_gpu_context_falls_back_to_pickle_without_shm_info() -> None:
    context = create_non_gpu_context(
        metadata=NonGpuContextMetadata(
            layout_desc=MemoryLayoutDesc(
                shapes=[torch.Size([2, 2])],
                dtypes=[torch.float32],
            ),
            block_size=1,
            use_mla=False,
        ),
        mq_client=MagicMock(),
        mq_timeout=1.0,
        shm_name="",
        pool_size=0,
    )
    assert isinstance(context, NonGpuContextPickle)


def test_non_gpu_context_shm_close_is_idempotent() -> None:
    shm_name = f"lmcache_test_close_{os.getpid()}"
    shm_path = _create_shm_file(shm_name, 4096)
    try:
        context = NonGpuContextShm(
            metadata=NonGpuContextMetadata(
                layout_desc=MemoryLayoutDesc(
                    shapes=[torch.Size([2, 2])],
                    dtypes=[torch.float32],
                ),
                block_size=1,
                use_mla=False,
            ),
            mq_client=MagicMock(),
            mq_timeout=1.0,
            shm_name=shm_name,
            pool_size=4096,
        )
        context.close()
        context.close()
    finally:
        if os.path.exists(shm_path):
            os.unlink(shm_path)
