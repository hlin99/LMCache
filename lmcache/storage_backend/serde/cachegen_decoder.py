# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.cachegen_basics import (
    CacheGenConfig,
    CacheGenGPUBytestream,
    CacheGenGPUEncoderOutput,
)
from lmcache.storage_backend.serde.serde import Deserializer
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
import lmcache.c_ops as lmc_ops
import lmcache.storage_backend.serde.cachegen_basics as CGBasics

logger = init_logger(__name__)


@_lmcache_nvtx_annotate
def quant(bins: int, xq: torch.Tensor, max1: float):
    C = bins // 2 - 1
    x = xq / C * max1
    return x


def do_dequantize(t: torch.Tensor, bins: torch.Tensor, maxtensors: torch.Tensor):
    """
    t: [nlayers, ntokens, nchannels]
    bins: [nlayers]
    maxtensors: [nlayers, ntokens, 1]
    """
    C = (bins // 2 - 1)[:, None, None]
    t = t - C
    t = t / C
    t = t * maxtensors
    return t


@_lmcache_nvtx_annotate
def recombine_bytes(bytes_tensor, output_lengths) -> torch.Tensor:
    output_buffer_size = CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK
    offsets = output_lengths.flatten().cumsum(0).roll(1).reshape(output_lengths.shape)
    offsets[0][0] = 0
    indexes = torch.arange(output_buffer_size, device=offsets.device).tile(
        (output_lengths.shape[0], output_lengths.shape[1], 1)
    )
    final_indexes = (indexes + offsets[:, :, None]).clamp(max=len(bytes_tensor) - 1)
    return bytes_tensor[final_indexes]

import lmcache.non_cuda_equivalents as py_ops
_decode_call_count = 0

@_lmcache_nvtx_annotate
def decode_chunk(
    cdf: torch.Tensor,
    data_chunk: CacheGenGPUBytestream,
    target_buffer: torch.Tensor,
) -> None:
    """
    Write the decode output in target_buffer
    Expected shape: [nlayers (kv in total), ntokens, nchannels]
    """
    global _decode_call_count
    _decode_call_count += 1
    call_id = _decode_call_count

    bytes_tensor = data_chunk.bytestream
    length_prefsum = (
        data_chunk.bytestream_lengths.flatten()
        .cumsum(0)
        .reshape(data_chunk.bytestream_lengths.shape)
    )
    # logger.error("tony 1 length_prefsum = %s", length_prefsum)
    # length_prefsum = length_prefsum.clamp(max=bytes_tensor.shape[0] - 1)
    # logger.error("tony 2 length_prefsum = %s", length_prefsum)
#    bytes_tensor = torch.nn.functional.pad(bytes_tensor, (0, 1), value=0)

    """
    max_prefsum = int(length_prefsum.max().item())
    target_len = max_prefsum + 4
    bs_len = bytes_tensor.shape[0]
    pad_size = max(0, target_len - bs_len)
    if pad_size > 0:
        logger.error(" pad_size = %s", pad_size)
        bytes_tensor = torch.nn.functional.pad(bytes_tensor, (0, pad_size), value=0)
    """
    logger.error(f"[call#{call_id}] ========== decode_chunk start ==========")
    logger.error(f"[call#{call_id}] cdf            shape={cdf.shape}  device={cdf.device}  sum={cdf.float().sum().item():.4f}  ptr={cdf.data_ptr()}")
    logger.error(f"[call#{call_id}] bytes_tensor   shape={bytes_tensor.shape}  device={bytes_tensor.device}  sum={bytes_tensor.float().sum().item():.4f}  ptr={bytes_tensor.data_ptr()}")
    logger.error(f"[call#{call_id}] length_prefsum shape={length_prefsum.shape}  device={length_prefsum.device}  vals={length_prefsum.flatten()[:8].tolist()}")
    logger.error(f"[call#{call_id}] target_buffer  shape={target_buffer.shape}  device={target_buffer.device}  ptr={target_buffer.data_ptr()}")
    logger.error(f"[call#{call_id}] target_buffer BEFORE zero_(): sum={target_buffer.float().sum().item():.4f}")

    target_buffer.zero_()
    logger.error(f"[call#{call_id}] target_buffer AFTER  zero_(): sum={target_buffer.float().sum().item():.4f}  (should be 0)")

    # ✅ clone 输入（只读），不要动它们
    cdf_clone            = cdf.clone()
    bytes_tensor_clone   = bytes_tensor.clone()
    length_prefsum_clone = length_prefsum.clone()
    test_buf             = torch.zeros_like(target_buffer)  # ✅ output 用 zeros_like

    # lmc_ops 用原始输入
    lmc_ops.decode_fast_prefsum(cdf, bytes_tensor, length_prefsum, target_buffer)
    torch.cuda.synchronize()
    logger.error(f"[call#{call_id}] target_buffer AFTER  lmc_ops: sum={target_buffer.float().sum().item():.4f}  first8={target_buffer.flatten()[:8].tolist()}")

    # py_ops 用 clone 的输入（验证 lmc_ops 有没有意外修改输入）
    logger.error(f"[call#{call_id}] cdf_clone    after lmc_ops: sum={cdf_clone.float().sum().item():.4f}  (若与调用前不同说明lmc_ops修改了cdf)")
    logger.error(f"[call#{call_id}] bytes_clone  after lmc_ops: sum={bytes_tensor_clone.float().sum().item():.4f}")
    logger.error(f"[call#{call_id}] prefsum_clone after lmc_ops: vals={length_prefsum_clone.flatten()[:8].tolist()}")

    py_ops.decode_fast_prefsum(cdf_clone, bytes_tensor_clone, length_prefsum_clone, test_buf)
    torch.cuda.synchronize()
    logger.error(f"[call#{call_id}] test_buf     AFTER  py_ops:  sum={test_buf.float().sum().item():.4f}  first8={test_buf.flatten()[:8].tolist()}")

    # ── 验证 lmc_ops 是否修改了输入 ──────────────────────────────────
    cdf_modified        = not torch.equal(cdf, cdf_clone)
    bytes_modified      = not torch.equal(bytes_tensor, bytes_tensor_clone)
    prefsum_modified    = not torch.equal(length_prefsum, length_prefsum_clone)
    logger.error(f"[call#{call_id}] lmc_ops 是否修改了 cdf={cdf_modified}  bytes={bytes_modified}  prefsum={prefsum_modified}")

    # ── 结果对比 ──────────────────────────────────────────────────────
    is_same = torch.equal(target_buffer, test_buf)
    if not is_same:
        diff = (target_buffer.float() - test_buf.float()).abs()
        diff_count = (diff > 0).sum().item()
        max_diff   = diff.max().item()
        flat_diff  = diff.flatten()
        first_diff_idx = flat_diff.nonzero(as_tuple=False)[0].item()
        logger.error(f"[call#{call_id}] ❌ 不一致！不同元素数={diff_count}  最大差={max_diff}")
        logger.error(f"[call#{call_id}]    第一个不同位置 flat_idx={first_diff_idx}")
        logger.error(f"[call#{call_id}]    lmc_ops值={target_buffer.flatten()[first_diff_idx].item()}  py_ops值={test_buf.flatten()[first_diff_idx].item()}")
        if len(target_buffer.shape) == 3:
            nl, nt, nc = target_buffer.shape
            li = first_diff_idx // (nt * nc)
            ti = (first_diff_idx % (nt * nc)) // nc
            ci = first_diff_idx % nc
            logger.error(f"[call#{call_id}]    对应位置: layer={li}  token={ti}  channel={ci}")
            logger.error(f"[call#{call_id}]    该 layer 的 length_prefsum: {length_prefsum[li].tolist()}")
            logger.error(f"[call#{call_id}]    bytes_tensor len={bytes_tensor.shape[0]}  prefsum max={length_prefsum.max().item()}")
            # 越界检测
            if length_prefsum.max().item() >= bytes_tensor.shape[0]:
                logger.error(f"[call#{call_id}]    ⚠️  prefsum 越界！max_prefsum={length_prefsum.max().item()} >= bytes_len={bytes_tensor.shape[0]}")
    else:
        logger.error(f"[call#{call_id}] ✅ 完全一致")

    logger.error(f"[call#{call_id}] ========== decode_chunk end ==========")
    print(f"[call#{call_id}] 是否完全一致: {is_same}")




def decode_chunk1(
    cdf: torch.Tensor,
    data_chunk: CacheGenGPUBytestream,
    target_buffer: torch.Tensor,
) -> None:
    """
    Write the decode output in target_buffer
    Expected shape: [nlayers (kv in total), ntokens, nchannels]
    """
    bytes_tensor = data_chunk.bytestream
    length_prefsum = (
        data_chunk.bytestream_lengths.flatten()
        .cumsum(0)
        .reshape(data_chunk.bytestream_lengths.shape)
    )
    logger.error("decode_fast_prefsum")
    target_buffer.zero_()
    lmc_ops.decode_fast_prefsum(cdf, bytes_tensor, length_prefsum, target_buffer)

    test_buf = torch.zeros_like(target_buffer)
    py_ops.decode_fast_prefsum(cdf, bytes_tensor, length_prefsum, test_buf)
    is_same = torch.equal(target_buffer, test_buf)
    print(f"是否完全一致: {is_same}")

@_lmcache_nvtx_annotate
def decode_function_gpu(
    cdf: torch.Tensor,
    data_chunks: List[CacheGenGPUBytestream],
    layers_in_key: int,
    chunk_size: int,
    output: torch.Tensor,
):
    # TODO: dtype and shape -- still have 128 and 8
    """
    Given the path to the encoded KV bytestream, decode the KV cache

    Inputs:
        cdf: the cdf tensor, in shape [2 * nlayers, nchannels, bins + 1]
        data_chunks: the data_chunks in the encoder's output
        layers_in_key: number of layers in K (or V)
        (K/V should have the same number of layers)
        chunk_size: the chunk_size
        output: output buffer, in shape [ntokens, 2 * nlayers * nchannels]

    Outputs:
        key: the decoded key tensor in the shape of (layers, tokens, nchannels)
        value: the decoded value tensor in the shape of
        (layers, tokens, nchannels)
    """
    nlayers, nchannels, _ = cdf.shape
    output = output.reshape((nlayers, chunk_size, nchannels))

    start = 0
    for data_chunk in data_chunks:
        end = start + data_chunk.ntokens
        decode_chunk(cdf, data_chunk, output[:, start:end, :])
        start = end

    out = output.reshape((2, layers_in_key, chunk_size, nchannels))
    key, value = out.float()

    return key, value


class CacheGenDeserializer(Deserializer):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        dtype,
    ):
        self.dtype = dtype
        self.cachegen_config = CacheGenConfig.from_model_name(metadata.model_name)
        self.chunk_size = config.chunk_size
        self.output_buffer: Optional[torch.Tensor] = None
        self.key_bins = self.make_key_bins(self.cachegen_config)
        self.value_bins = self.make_value_bins(self.cachegen_config)

    def make_key_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.nlayers)
        for spec in config.kspecs:
            ret[spec.start_layer : spec.end_layer] = spec.bins
        return ret.cuda()

    def make_value_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.nlayers)
        for spec in config.vspecs:
            ret[spec.start_layer : spec.end_layer] = spec.bins
        return ret.cuda()

    def get_output_buffer(self, nlayers: int, nchannels: int, ntokens: int):
        if (
            self.output_buffer is None
            or self.output_buffer.shape[1] != 2 * nlayers * nchannels
        ):
            self.output_buffer = torch.zeros(
                (self.chunk_size, 2 * nlayers * nchannels), dtype=torch.uint8
            ).cuda()
        return self.output_buffer[:ntokens, :]

    @_lmcache_nvtx_annotate
    def from_bytes(self, bs: bytes) -> torch.Tensor:
        encoder_output = CacheGenGPUEncoderOutput.from_bytes(bs)
        encoder_output.max_tensors_key = encoder_output.max_tensors_key.cuda()
        encoder_output.max_tensors_value = encoder_output.max_tensors_value.cuda()

        ntokens = encoder_output.max_tensors_key.shape[1]
        layers_in_key = encoder_output.max_tensors_key.shape[0]
        key, value = decode_function_gpu(
            encoder_output.cdf,
            encoder_output.data_chunks,
            layers_in_key,
            ntokens,
            self.get_output_buffer(
                encoder_output.cdf.shape[0] // 2,
                encoder_output.cdf.shape[1],
                ntokens,
            ),
        )

        # Temporary fix for #83: change the device of key_bins and value_bins
        # to the device of key and value
        # This requires a long-term fix in the future. Currently,
        # CacheGenGPUEncoderOutput has implicit device in itself.
        # More specifically, if the encoder encodes the tensor on GPU0, the
        # from_bytes will also return a tensor on GPU0
        # We may want to dynamically configure the device based on config and
        # metadata in the future
        if self.key_bins.device != key.device:
            self.key_bins = self.key_bins.to(key.device)

        if self.value_bins.device != value.device:
            self.value_bins = self.value_bins.cuda()

        key = do_dequantize(key, self.key_bins, encoder_output.max_tensors_key)
        value = do_dequantize(value, self.value_bins, encoder_output.max_tensors_value)
        """ merge key and value back and reshape """
        nlayers, ntokens, nchannels = key.shape
        blob = torch.stack([key, value])  # [2, nlayers, ntokens, nchannels]
        blob = blob.reshape(
            (
                2,
                nlayers,
                ntokens,
                encoder_output.num_heads,
                encoder_output.head_size,
            )
        )

        return blob.permute((1, 0, 2, 3, 4)).to(
            self.dtype
        )  # [nlayers, 2, ntokens, num_heads, head_size]
        # huggingface
        # return blob.permute((1, 0, 3, 2, 4)).to(
        #     self.dtype
        # )  # [nlayers, 2, num_heads, ntokens, head_size]
