# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional, Union
import struct

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey, parse_cache_key
from lmcache.v1.memory_management import MemoryFormat

logger = init_logger(__name__)


def _pad_shape_to_4d(shape: torch.Size) -> tuple:
    """Pad a shape to 4D by appending zeros for missing dimensions.

    The wire protocol always transmits exactly 4 dimension values.
    Shapes with fewer than 4 dimensions are right-padded with 0.

    Args:
        shape: The original tensor shape (1-D to 4-D).

    Returns:
        A 4-element tuple suitable for struct packing.
    """
    dims = list(shape)
    while len(dims) < 4:
        dims.append(0)
    return tuple(dims[:4])


def _strip_shape_trailing_zeros(shape: torch.Size) -> torch.Size:
    """Strip trailing zeros from a deserialized 4-D shape.

    Recovers the original dimensionality of shapes that were padded
    by :func:`_pad_shape_to_4d` before transmission.

    Args:
        shape: A 4-D torch.Size as received from the wire.

    Returns:
        The shape with trailing zero dimensions removed.
    """
    dims = list(shape)
    while len(dims) > 0 and dims[-1] == 0:
        dims.pop()
    return torch.Size(dims)


MAX_KEY_LENGTH = 150
REMOTE_METADATA_FMT: Optional[str] = None
REMOTE_METADATA_BYTES: Optional[int] = None


class ClientCommand(IntEnum):
    PUT = auto()
    GET = auto()
    EXIST = auto()
    LIST = auto()
    HEALTH = auto()


class ServerReturnCode(IntEnum):
    SUCCESS = 200
    FAIL = 400


DTYPE_TO_INT = {
    None: 0,
    torch.half: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
    torch.float: 4,
    torch.float32: 4,
    torch.float64: 5,
    torch.double: 5,
    torch.uint8: 6,
    torch.float8_e4m3fn: 7,
    torch.float8_e5m2: 8,
}

INT_TO_DTYPE = {
    0: None,
    1: torch.half,
    2: torch.float16,
    3: torch.bfloat16,
    4: torch.float,
    5: torch.float64,
    6: torch.uint8,
    7: torch.float8_e4m3fn,
    8: torch.float8_e5m2,
}

# TODO (Jiayi): Add more backends
LOCATION_TO_INT = {
    None: 0,
    "LocalCPUBackend": 1,
    "LocalDiskBackend": 2,
}

INT_TO_LOCATION = {
    0: None,
    1: "LocalCPUBackend",
    2: "LocalDiskBackend",
}


def init_remote_metadata_info(num_groups: int):
    global REMOTE_METADATA_FMT
    global REMOTE_METADATA_BYTES
    # length, fmt, (dtype, shape0, shape1, shape2, shape3) * num_groups
    fmt_length = 2 + 5 * num_groups
    REMOTE_METADATA_FMT = "i" * fmt_length
    REMOTE_METADATA_BYTES = 4 * fmt_length
    logger.info(
        "init remote metadata info with groups: %s, "
        "remote metadata fmt: %s, remote metadata bytes: %s",
        num_groups,
        REMOTE_METADATA_FMT,
        REMOTE_METADATA_BYTES,
    )


def get_remote_metadata_bytes():
    global REMOTE_METADATA_BYTES
    assert REMOTE_METADATA_BYTES is not None
    return REMOTE_METADATA_BYTES


@dataclass
class RemoteMetadata:
    length: int
    shapes: list[torch.Size]
    dtypes: list[torch.dtype]
    fmt: MemoryFormat

    def _prepare_params(self):
        params = [self.length, int(self.fmt.value)]
        for shape, dtype in zip(self.shapes, self.dtypes, strict=True):
            padded = _pad_shape_to_4d(shape)
            params.append(DTYPE_TO_INT[dtype])
            params.append(padded[0])
            params.append(padded[1])
            params.append(padded[2])
            params.append(padded[3])
        return params

    def serialize_into(self, buffer):
        assert REMOTE_METADATA_FMT is not None
        params = self._prepare_params()
        struct.pack_into(REMOTE_METADATA_FMT, buffer, 0, *params)

    def serialize(self) -> bytes:
        assert REMOTE_METADATA_FMT is not None
        params = self._prepare_params()
        packed_bytes = struct.pack(REMOTE_METADATA_FMT, *params)
        return packed_bytes

    @staticmethod
    def deserialize(s: bytes) -> "RemoteMetadata":
        assert REMOTE_METADATA_FMT is not None
        # length, fmt, (dtype, shape0, shape1, shape2, shape3) * num_groups
        result = struct.unpack_from(REMOTE_METADATA_FMT, s)
        length = result[0]
        memory_fmt = MemoryFormat(result[1])
        shapes = []
        dtypes = []
        for i in range(2, len(result), 5):
            raw_shape = torch.Size(result[i + 1 : i + 5])
            shapes.append(_strip_shape_trailing_zeros(raw_shape))
            dtypes.append(INT_TO_DTYPE[result[i]])

        return RemoteMetadata(
            length,
            shapes,
            dtypes,
            memory_fmt,
        )


# TODO(Jiayi): Server and client message can be merged into one.


@dataclass
class ClientMetaMessage:
    """
    Request message from LMCache workers or servers.
    """

    command: ClientCommand
    key: Union[CacheEngineKey, LayerCacheEngineKey]
    length: int
    fmt: MemoryFormat
    dtype: Optional[torch.dtype]
    shape: torch.Size
    location: Optional[str] = None

    def serialize(self) -> bytes:
        key_str = self.key.to_string()
        assert len(key_str) <= MAX_KEY_LENGTH, (
            f"Key length {len(key_str)} exceeds maximum {MAX_KEY_LENGTH}"
        )

        padded = _pad_shape_to_4d(self.shape)

        packed_bytes = struct.pack(
            f"iiiiiiiii{MAX_KEY_LENGTH}s",
            self.command.value,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            LOCATION_TO_INT[self.location],
            padded[0],
            padded[1],
            padded[2],
            padded[3],
            key_str.encode().ljust(MAX_KEY_LENGTH),
        )
        return packed_bytes

    @staticmethod
    def deserialize(s: bytes) -> "ClientMetaMessage":
        command, length, fmt, dtype, location, shape0, shape1, shape2, shape3, key = (
            struct.unpack(f"iiiiiiiii{MAX_KEY_LENGTH}s", s)
        )
        raw_shape = torch.Size([shape0, shape1, shape2, shape3])
        return ClientMetaMessage(
            ClientCommand(command),
            parse_cache_key(key.decode().strip()),
            length,
            MemoryFormat(fmt),
            INT_TO_DTYPE[dtype],
            _strip_shape_trailing_zeros(raw_shape),
            INT_TO_LOCATION[location],
        )

    @staticmethod
    def packlength() -> int:
        # NOTE: 9 is the number of integers
        return 4 * 9 + MAX_KEY_LENGTH


@dataclass
class ServerMetaMessage:
    """
    Reply message from LMCache workers or servers.
    """

    code: ServerReturnCode
    length: int
    fmt: MemoryFormat
    dtype: Optional[torch.dtype]
    shape: torch.Size
    location: Optional[str] = None

    def serialize(self) -> bytes:
        padded = _pad_shape_to_4d(self.shape)
        packed_bytes = struct.pack(
            "iiiiiiiii",
            self.code.value,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            padded[0],
            padded[1],
            padded[2],
            padded[3],
            LOCATION_TO_INT[self.location],
        )
        return packed_bytes

    @staticmethod
    def packlength() -> int:
        return 4 * 9

    @staticmethod
    def deserialize(s: bytes) -> "ServerMetaMessage":
        code, length, fmt, dtype, shape0, shape1, shape2, shape3, location = (
            struct.unpack("iiiiiiiii", s)
        )
        raw_shape = torch.Size([shape0, shape1, shape2, shape3])
        return ServerMetaMessage(
            ServerReturnCode(code),
            length,
            MemoryFormat(fmt),
            INT_TO_DTYPE[dtype],
            _strip_shape_trailing_zeros(raw_shape),
            INT_TO_LOCATION[location],
        )
