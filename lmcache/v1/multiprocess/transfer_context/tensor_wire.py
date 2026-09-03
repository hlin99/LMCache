# SPDX-License-Identifier: Apache-2.0
"""Safe wire format for inline engine-driven tensor transfers."""

# Standard
from typing import Annotated, Literal, TypeAlias
import math
import warnings

# Third Party
import msgspec
import torch

# First Party
from lmcache.v1.platform.base.ipc_wire import DTYPES, NonNegative

TensorShape: TypeAlias = Annotated[tuple[NonNegative, ...], msgspec.Meta(max_length=16)]
TensorBuffer: TypeAlias = Annotated[
    memoryview, msgspec.Meta(max_length=2 * 1024 * 1024 * 1024)
]

_MAX_TOTAL_TENSOR_BYTES = 16 * 1024 * 1024 * 1024
_MAX_METADATA_BYTES_PER_TENSOR = 1024
_MAX_TENSOR_COUNT = 65536
_MAX_TENSOR_BYTES = 2 * 1024 * 1024 * 1024


class TensorFrame(msgspec.Struct, forbid_unknown_fields=True):
    """Typed metadata and raw bytes for one contiguous CPU tensor."""

    dtype: str
    shape: TensorShape
    data: TensorBuffer


TensorFrames: TypeAlias = Annotated[list[TensorFrame], msgspec.Meta(max_length=65536)]


class TensorEnvelope(msgspec.Struct, forbid_unknown_fields=True):
    """Versioned collection of inline CPU tensors."""

    version: Literal[1]
    tensors: TensorFrames


TensorGroups: TypeAlias = Annotated[list[TensorFrames], msgspec.Meta(max_length=64)]


class GroupedTensorEnvelope(msgspec.Struct, forbid_unknown_fields=True):
    """Versioned group-major collection of inline CPU tensors.

    Used by multi-group (hybrid-KV) transfers, where each KV group has its own
    tensor layout. Every group is validated against its own expected layout,
    under the same per-tensor and total limits as :class:`TensorEnvelope`. The
    two envelopes have disjoint field names, so neither decodes as the other.
    """

    version: Literal[1]
    groups: TensorGroups


_MAX_GROUP_COUNT = 64


def encode_tensors(tensors: list[torch.Tensor]) -> bytes:
    """Encode CPU tensors without invoking Python object serialization.

    Args:
        tensors: Contiguous or non-contiguous CPU tensors to encode.

    Returns:
        A versioned MessagePack envelope containing typed metadata and raw bytes.

    Raises:
        ValueError: If a tensor is not on CPU, has unsupported metadata, or the
            cumulative payload exceeds the wire-format limit.
    """
    if len(tensors) > _MAX_TENSOR_COUNT:
        raise ValueError("Inline tensor count exceeds 65536")
    frames, _ = _encode_frames(tensors, 0)
    return msgspec.msgpack.encode(TensorEnvelope(version=1, tensors=frames))


def _encode_frames(
    tensors: list[torch.Tensor], total_bytes: int
) -> tuple[list[TensorFrame], int]:
    """Encode ``tensors`` as frames, adding their bytes to ``total_bytes``."""
    frames: list[TensorFrame] = []
    for tensor in tensors:
        if tensor.device.type != "cpu":
            raise ValueError("Inline tensor transfer only supports CPU tensors")
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported inline tensor dtype: {tensor.dtype}")
        shape = tuple(tensor.shape)
        if len(shape) > 16:
            raise ValueError("Inline tensor rank exceeds 16")
        raw = memoryview(
            tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
        )
        if len(raw) > _MAX_TENSOR_BYTES:
            raise ValueError("Inline tensor exceeds 2 GiB")
        total_bytes += len(raw)
        if total_bytes > _MAX_TOTAL_TENSOR_BYTES:
            raise ValueError("Inline tensor payload exceeds 16 GiB")
        frames.append(TensorFrame(dtype=dtype, shape=shape, data=raw))
    return frames, total_bytes


def decode_tensors(
    payload: bytes,
    *,
    expected_shape: torch.Size,
    expected_dtype: torch.dtype,
    expected_count: int,
) -> list[torch.Tensor]:
    """Decode and validate an inline tensor envelope.

    Args:
        payload: MessagePack bytes produced by :func:`encode_tensors`.
        expected_shape: Required shape for every decoded tensor.
        expected_dtype: Required dtype for every decoded tensor.
        expected_count: Required tensor count.

    Returns:
        CPU tensors backed directly by the immutable input payload. Callers must
        treat the returned tensors as read-only.

    Raises:
        ValueError: If the envelope or any tensor metadata, count, or byte length
            does not match the expected transfer layout.
        msgspec.DecodeError: If ``payload`` is not a valid typed envelope.
    """
    if not 0 <= expected_count <= _MAX_TENSOR_COUNT:
        raise ValueError("Expected tensor count must be between 0 and 65536")
    if len(payload) > (
        _MAX_TOTAL_TENSOR_BYTES + _MAX_METADATA_BYTES_PER_TENSOR * expected_count
    ):
        raise ValueError("Inline tensor envelope exceeds the wire-format limit")

    envelope = msgspec.msgpack.decode(payload, type=TensorEnvelope)
    if len(envelope.tensors) != expected_count:
        raise ValueError(
            "Inline tensor count mismatch: "
            f"expected {expected_count}, got {len(envelope.tensors)}"
        )

    tensors, _ = _decode_frames(envelope.tensors, expected_shape, expected_dtype, 0)
    return tensors


def _decode_frames(
    frames: list[TensorFrame],
    expected_shape: torch.Size,
    expected_dtype: torch.dtype,
    total_bytes: int,
) -> tuple[list[torch.Tensor], int]:
    """Validate ``frames`` against one layout and view them as tensors.

    Adds the frames' bytes to ``total_bytes`` and enforces the cumulative
    limit, so callers decoding several frame lists share one budget.
    """
    expected_shape_tuple = tuple(expected_shape)
    expected_dtype_name = str(expected_dtype).removeprefix("torch.")
    expected_tensor_bytes = math.prod(expected_shape_tuple) * expected_dtype.itemsize
    tensors: list[torch.Tensor] = []
    for frame in frames:
        dtype = DTYPES.get(frame.dtype)
        if dtype is None:
            raise ValueError(f"Unsupported inline tensor dtype: {frame.dtype}")
        if frame.dtype != expected_dtype_name:
            raise ValueError(
                "Inline tensor dtype mismatch: "
                f"expected {expected_dtype_name}, got {frame.dtype}"
            )
        if frame.shape != expected_shape_tuple:
            raise ValueError(
                "Inline tensor shape mismatch: "
                f"expected {expected_shape_tuple}, got {frame.shape}"
            )
        if len(frame.data) != expected_tensor_bytes:
            raise ValueError(
                "Inline tensor byte length mismatch: "
                f"expected {expected_tensor_bytes}, got {len(frame.data)}"
            )
        total_bytes += len(frame.data)
        if total_bytes > _MAX_TOTAL_TENSOR_BYTES:
            raise ValueError("Inline tensor payload exceeds 16 GiB")
        # Downstream copies or scatters from these tensors without mutating them.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            raw = torch.frombuffer(frame.data, dtype=torch.uint8)
        tensors.append(raw.view(dtype).reshape(frame.shape))
    return tensors, total_bytes


def encode_grouped_tensors(groups: list[list[torch.Tensor]]) -> bytes:
    """Encode group-major CPU tensors without Python object serialization.

    Args:
        groups: ``groups[group][chunk]`` CPU tensors; each group may have its
            own dtype and shape.

    Returns:
        A versioned MessagePack :class:`GroupedTensorEnvelope`.

    Raises:
        ValueError: If there are more than 64 groups, more than 65536 tensors
            in total, or any tensor or the cumulative payload is invalid as for
            :func:`encode_tensors`.
    """
    if len(groups) > _MAX_GROUP_COUNT:
        raise ValueError("Inline tensor group count exceeds 64")
    if sum(len(group) for group in groups) > _MAX_TENSOR_COUNT:
        raise ValueError("Inline tensor count exceeds 65536")
    total_bytes = 0
    encoded: list[list[TensorFrame]] = []
    for group in groups:
        frames, total_bytes = _encode_frames(group, total_bytes)
        encoded.append(frames)
    return msgspec.msgpack.encode(GroupedTensorEnvelope(version=1, groups=encoded))


def decode_grouped_tensors(
    payload: bytes,
    *,
    expected: list[tuple[torch.Size, torch.dtype, int]],
) -> list[list[torch.Tensor]]:
    """Decode and validate a group-major inline tensor envelope.

    Args:
        payload: MessagePack bytes produced by :func:`encode_grouped_tensors`.
        expected: One ``(shape, dtype, count)`` per group, in group order. Every
            tensor of a group must match that group's shape and dtype, and the
            group must hold exactly ``count`` tensors.

    Returns:
        ``tensors[group][chunk]`` backed directly by the immutable input
        payload. Callers must treat the returned tensors as read-only.

    Raises:
        ValueError: If the group count, any per-group tensor count, or any
            tensor metadata or byte length does not match ``expected``.
        msgspec.DecodeError: If ``payload`` is not a valid grouped envelope.
    """
    if not 0 <= len(expected) <= _MAX_GROUP_COUNT:
        raise ValueError("Expected group count must be between 0 and 64")
    expected_total = sum(count for _, _, count in expected)
    if any(count < 0 for _, _, count in expected) or not (
        0 <= expected_total <= _MAX_TENSOR_COUNT
    ):
        raise ValueError("Expected tensor count must be between 0 and 65536")
    if len(payload) > (
        _MAX_TOTAL_TENSOR_BYTES + _MAX_METADATA_BYTES_PER_TENSOR * expected_total
    ):
        raise ValueError("Inline tensor envelope exceeds the wire-format limit")

    envelope = msgspec.msgpack.decode(payload, type=GroupedTensorEnvelope)
    if len(envelope.groups) != len(expected):
        raise ValueError(
            "Inline tensor group count mismatch: "
            f"expected {len(expected)}, got {len(envelope.groups)}"
        )
    total_bytes = 0
    groups: list[list[torch.Tensor]] = []
    for gid, (frames, (shape, dtype, count)) in enumerate(
        zip(envelope.groups, expected, strict=True)
    ):
        if len(frames) != count:
            raise ValueError(
                f"Inline tensor count mismatch in group {gid}: "
                f"expected {count}, got {len(frames)}"
            )
        tensors, total_bytes = _decode_frames(frames, shape, dtype, total_bytes)
        groups.append(tensors)
    return groups
