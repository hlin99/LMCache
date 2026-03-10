# SPDX-License-Identifier: Apache-2.0

# Standard
import unittest.mock

# Third Party
import pytest
import torch

# First Party
import lmcache.non_cuda_equivalents as _py_ops


@pytest.mark.no_shared_allocator
def test_tensor_from_cuda_ptr_success_path_does_not_log_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful CUDA pointer wrapping should not emit error-level logs."""
    ptr = 1234
    shape = (2, 2)
    dtype = torch.float16
    numel = 4
    total_bytes = numel * torch.empty((), dtype=dtype).element_size()

    def _fake_as_tensor(wrapper: object, device: torch.device) -> torch.Tensor:
        cuda_interface = getattr(wrapper, "__cuda_array_interface__")
        assert cuda_interface["data"] == (ptr, False)
        assert cuda_interface["shape"] == (numel,)
        return torch.arange(numel, dtype=dtype)

    error_mock = unittest.mock.Mock()
    monkeypatch.setattr(_py_ops.torch, "as_tensor", _fake_as_tensor)
    monkeypatch.setattr(_py_ops.logger, "error", error_mock)

    out = _py_ops._tensor_from_cuda_ptr(
        ptr=ptr,
        shape=shape,
        dtype=dtype,
        device=torch.device("cuda"),
        numel=numel,
        total_bytes=total_bytes,
    )

    assert out.shape == shape
    error_mock.assert_not_called()
