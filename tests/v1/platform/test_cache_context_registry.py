# SPDX-License-Identifier: Apache-2.0
"""Registry wiring tests for ``BaseCacheContext``."""

# Standard
from collections.abc import Generator
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import cache_context as cache_context_module
from lmcache.v1.platform._registry import _discover_base_classes
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base_cache_context import (
    BaseCacheContext as ShimBaseCacheContext,
)
from lmcache.v1.platform.cache_context import create_cache_context


class _FakeWrapper:
    """Minimal stand-in for a KV-cache IPC wrapper."""

    def __init__(self, device_type: str) -> None:
        self._device_type = device_type

    def to_tensor(self) -> torch.Tensor:
        """Return an empty tensor on the configured device for dispatch tests."""
        return torch.empty(0, device=torch.device(self._device_type))


class _FakeContext(BaseCacheContext):
    """Bare-bones ``BaseCacheContext`` subclass used as a dispatch stub."""

    device_type = "cpu"

    def __init__(
        self,
        kv_caches: Any,
        lmcache_tokens_per_chunk: int,
        layout_hints: Any,
        engine_group_infos: Any,
        engine_type: Any,
        separate_object_groups: bool = True,
    ) -> None:
        self.kv_caches = kv_caches
        self.lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        self.layout_hints = layout_hints
        self.engine_group_infos = engine_group_infos
        self.engine_type = engine_type
        self.separate_object_groups = separate_object_groups

    @property
    def stream(self) -> Any:  # pragma: no cover - never invoked
        """Return a required stub stream for abstract-method satisfaction."""
        return None

    @property
    def cupy_stream(self) -> Any:  # pragma: no cover - never invoked
        """Return a required stub CuPy stream for abstract-method satisfaction."""
        return None

    @property
    def max_batch_size(self) -> int:  # pragma: no cover - never invoked
        """Return a required stub batch limit for abstract-method satisfaction."""
        return 0

    def close(self) -> None:  # pragma: no cover - never invoked
        """Provide a required stub close method for abstract-method satisfaction."""
        return None

    def get_kernel_group_kv_pointers(
        self, kernel_group_idx: int
    ) -> torch.Tensor:  # pragma: no cover
        """Return a required stub tensor for abstract-method satisfaction."""
        return torch.empty(0)

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:  # pragma: no cover
        """Return a required stub tensor for abstract-method satisfaction."""
        return torch.empty(0)

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:  # pragma: no cover
        """Return a required stub tensor for abstract-method satisfaction."""
        return torch.empty(0)

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:  # pragma: no cover
        """Return a required stub shape/dtype pair for abstract-method satisfaction."""
        return torch.Size(()), torch.float32

    def cache_size_per_token(self) -> int:  # pragma: no cover
        """Return a required stub size value for abstract-method satisfaction."""
        return 0


@pytest.fixture
def isolated_registry() -> Generator[None, None, None]:
    """Snapshot the backend table so test-installed fakes do not leak."""
    saved = cache_context_module.snapshot_backends()
    cache_context_module.restore_backends({})
    try:
        yield
    finally:
        cache_context_module.restore_backends(saved)


def test_base_cache_context_is_discovered_from_base_package() -> None:
    """BaseCacheContext is discoverable from ``lmcache.v1.platform.base``."""
    assert BaseCacheContext in _discover_base_classes()


def test_base_cache_context_backward_compat_shim_reexports_same_class() -> None:
    """The legacy module path re-exports the canonical BaseCacheContext class."""
    assert ShimBaseCacheContext is BaseCacheContext


def test_create_cache_context_dispatch_still_uses_registered_backend(
    isolated_registry: Any,
) -> None:
    """create_cache_context still dispatches through the existing backend map."""
    cache_context_module.restore_backends({"cpu": _FakeContext})

    wrappers = [_FakeWrapper("cpu")]
    ctx = create_cache_context(wrappers, lmcache_tokens_per_chunk=128)  # type: ignore[arg-type]

    assert isinstance(ctx, _FakeContext)
    assert ctx.kv_caches is wrappers
    assert ctx.lmcache_tokens_per_chunk == 128
