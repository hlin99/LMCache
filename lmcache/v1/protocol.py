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


def _pad_shape_to_4d(shape: torch.Size) -> torch.Size:
    """
    Pad a shape to 4D by prepending 1s.

    Args:
        shape: The input shape (can be 1D, 2D, 3D, or 4D)

    Returns:
        A 4D shape

    Examples:
        [100] -> [1, 1, 1, 100]
        [100, 200] -> [1, 1, 100, 200]
        [100, 2, 4096] -> [1, 100, 2, 4096]
        [2, 32, 100, 4096] -> [2, 32, 100, 4096]
    """
    if len(shape) == 4:
        return shape
    elif len(shape) < 4:
        # Pad with 1s at the beginning
        padding = [1] * (4 - len(shape))
        return torch.Size(padding + list(shape))
    else:
        raise ValueError(f"Shape dimension {len(shape)} is greater than 4")


@dataclass
class RemoteMetadata:
    length: int
    shapes: list[torch.Size]
    dtypes: list[torch.dtype]
    fmt: MemoryFormat

    def _prepare_params(self):
        params = [self.length, int(self.fmt.value)]
        for shape, dtype in zip(self.shapes, self.dtypes, strict=True):
            # Pad shapes to 4D if needed
            padded_shape = _pad_shape_to_4d(shape)
            params.append(DTYPE_TO_INT[dtype])
            params.append(padded_shape[0])
            params.append(padded_shape[1])
            params.append(padded_shape[2])
            params.append(padded_shape[3])
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
            shapes.append(torch.Size(result[i + 1 : i + 5]))
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

        # NOTE(Jiayi): 4 is the maximum dimension of memory object.
        # Pass in shape [x, 0, 0, 0] if it is a bytes memory object
        # Pad shape to 4D if needed
        padded_shape = _pad_shape_to_4d(self.shape)

        packed_bytes = struct.pack(
            f"iiiiiiiii{MAX_KEY_LENGTH}s",
            self.command.value,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            LOCATION_TO_INT[self.location],
            padded_shape[0],
            padded_shape[1],
            padded_shape[2],
            padded_shape[3],
            key_str.encode().ljust(MAX_KEY_LENGTH),
        )
        return packed_bytes

    @staticmethod
    def deserialize(s: bytes) -> "ClientMetaMessage":
        command, length, fmt, dtype, location, shape0, shape1, shape2, shape3, key = (
            struct.unpack(f"iiiiiiiii{MAX_KEY_LENGTH}s", s)
        )
        return ClientMetaMessage(
            ClientCommand(command),
            parse_cache_key(key.decode().strip()),
            length,
            MemoryFormat(fmt),
            INT_TO_DTYPE[dtype],
            torch.Size([shape0, shape1, shape2, shape3]),
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
        # Pad shape to 4D if needed
        padded_shape = _pad_shape_to_4d(self.shape)
        packed_bytes = struct.pack(
            "iiiiiiiii",
            self.code.value,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            padded_shape[0],
            padded_shape[1],
            padded_shape[2],
            padded_shape[3],
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
        return ServerMetaMessage(
            ServerReturnCode(code),
            length,
            MemoryFormat(fmt),
            INT_TO_DTYPE[dtype],
            torch.Size([shape0, shape1, shape2, shape3]),
            INT_TO_LOCATION[location],
        )
