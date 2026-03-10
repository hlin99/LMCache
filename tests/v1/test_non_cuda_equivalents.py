# SPDX-License-Identifier: Apache-2.0
# Standard
import ctypes

# Third Party
import torch

# First Party
import lmcache.non_cuda_equivalents as lmc_ops


def test_multi_layer_kv_transfer_pointer_uses_memcpy(monkeypatch):
    memcpy_calls: list[tuple[int, int, int]] = []

    def fake_cuda_memcpy(dst_ptr: int, src_ptr: int, nbytes: int) -> None:
        memcpy_calls.append((dst_ptr, src_ptr, nbytes))
        ctypes.memmove(dst_ptr, src_ptr, nbytes)

    monkeypatch.setattr(lmc_ops, "_cuda_memcpy", fake_cuda_memcpy)

    key_value = torch.arange(8, dtype=torch.float32).view(1, 1, 2, 4)
    paged = torch.zeros((4, 4), dtype=torch.float32)
    key_value_ptrs = torch.tensor([paged.data_ptr()], dtype=torch.int64)
    slot_mapping = torch.tensor([2, 1], dtype=torch.int64)

    lmc_ops.multi_layer_kv_transfer(
        key_value=key_value,
        key_value_ptrs=key_value_ptrs,
        slot_mapping=slot_mapping,
        paged_memory_device=torch.device("cpu"),
        page_buffer_size=4,
        direction=lmc_ops.TransferDirection.H2D,
        gpu_kv_format=lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
    )

    assert torch.equal(paged[2], key_value[0, 0, 0])
    assert torch.equal(paged[1], key_value[0, 0, 1])
    assert len(memcpy_calls) == 2


def test_multi_layer_kv_transfer_unilateral_pointer_uses_memcpy(monkeypatch):
    memcpy_calls: list[tuple[int, int, int]] = []

    def fake_cuda_memcpy(dst_ptr: int, src_ptr: int, nbytes: int) -> None:
        memcpy_calls.append((dst_ptr, src_ptr, nbytes))
        ctypes.memmove(dst_ptr, src_ptr, nbytes)

    monkeypatch.setattr(lmc_ops, "_cuda_memcpy", fake_cuda_memcpy)

    key_value = torch.arange(16, dtype=torch.float32).view(2, 1, 2, 4)
    k_buf = torch.zeros((4, 4), dtype=torch.float32)
    v_buf = torch.zeros((4, 4), dtype=torch.float32)
    key_value_ptrs = torch.tensor([k_buf.data_ptr(), v_buf.data_ptr()], dtype=torch.int64)
    slot_mapping = torch.tensor([0, 3], dtype=torch.int64)

    lmc_ops.multi_layer_kv_transfer_unilateral(
        key_value=key_value,
        key_value_ptrs=key_value_ptrs,
        slot_mapping=slot_mapping,
        paged_memory_device=torch.device("cpu"),
        page_buffer_size=4,
        direction=lmc_ops.TransferDirection.H2D,
        gpu_kv_format=lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS,
    )

    assert torch.equal(k_buf[0], key_value[0, 0, 0])
    assert torch.equal(k_buf[3], key_value[0, 0, 1])
    assert torch.equal(v_buf[0], key_value[1, 0, 0])
    assert torch.equal(v_buf[3], key_value[1, 0, 1])
    assert len(memcpy_calls) == 4
