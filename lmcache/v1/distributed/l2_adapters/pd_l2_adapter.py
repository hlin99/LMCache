# SPDX-License-Identifier: Apache-2.0
"""
PD (Prefetch/Decode) L2 adapter configuration.

Provides ``PdL2AdapterConfig`` for configuring a pipeline-parallel
KV-cache transfer channel in Multi-process (MP) mode.  The config
object is used to describe both the *sender* (prefill worker) and
the *receiver* (decode worker) side of the transfer.

The full PD L2Adapter implementation (store/load/networking) is
planned for a later PR.  This module registers the config class and
a stub factory so that ``"pd"`` appears in
``get_registered_l2_adapter_types()`` and the configuration layer
can be exercised and tested independently.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.config import EvictionConfig
    from lmcache.v1.distributed.internal_api import L1MemoryDesc
    from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    PersistConfig,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory

_VALID_ROLES: frozenset[str] = frozenset({"sender", "receiver"})
_REQUIRED_FIELDS: tuple[str, ...] = (
    "role",
    "peer_host",
    "peer_init_port",
    "peer_alloc_port",
)


@dataclass(frozen=True)
class PdL2AdapterConfig(L2AdapterConfigBase):
    """
    Configuration for the PD (Prefetch/Decode) L2 adapter.

    Used to set up a high-performance KV-cache transfer channel
    between a sender (prefill) and a receiver (decode) worker in
    Multi-process (MP) mode.

    Required fields:
        role: ``"sender"`` or ``"receiver"``.
        peer_host: Remote peer hostname or IP address.
        peer_init_port: Per-TP-rank NIXL initialisation ports
            (e.g. ``[9051]`` for TP=1).
        peer_alloc_port: Per-TP-rank allocation/notification ports
            (e.g. ``[9052]`` for TP=1).

    Optional fields:
        proxy_host: Proxy notification host; used by sender only
            (default: ``""``).
        proxy_port: Proxy notification port; used by sender only
            (default: ``0``).
        buffer_size: Staging buffer size in bytes per rank
            (default: ``67108864`` = 64 MB).
        buffer_device: Device for the staging buffer, ``"cpu"`` or
            ``"cuda"`` (default: ``"cpu"``).
        transfer_channel: Transfer backend, ``"nixl"`` or
            ``"mock_memory"`` (default: ``"nixl"``).
        nixl_backends: List of NIXL transport backends
            (default: ``["tcp"]``).
        eviction_config: Optional eviction config parsed from the
            ``"eviction"`` sub-dict (default: ``None``).
        persist_config: Persist config parsed from the optional
            ``"persist_enabled"`` key (default: persist enabled).
    """

    role: str
    peer_host: str
    peer_init_port: list[int]
    peer_alloc_port: list[int]
    proxy_host: str = ""
    proxy_port: int = 0
    buffer_size: int = 67108864  # 64 MB
    buffer_device: str = "cpu"
    transfer_channel: str = "nixl"
    nixl_backends: list[str] = field(default_factory=lambda: ["tcp"])
    eviction_config: EvictionConfig | None = field(
        default=None, compare=False, repr=False
    )
    persist_config: PersistConfig = field(
        default_factory=PersistConfig, compare=False, repr=False
    )

    @classmethod
    def from_dict(cls, d: dict) -> "PdL2AdapterConfig":
        """
        Build a PdL2AdapterConfig from a dict (e.g. from parsed JSON).

        Validates all required fields and the ``role`` value, then
        constructs a fully-populated frozen instance.

        Args:
            d: Adapter spec dict. Must include all required keys
                (``role``, ``peer_host``, ``peer_init_port``,
                ``peer_alloc_port``).

        Returns:
            A fully-populated PdL2AdapterConfig instance.

        Raises:
            ValueError: If a required key is missing or ``role`` is
                not ``"sender"`` or ``"receiver"``.
        """
        for key in _REQUIRED_FIELDS:
            if key not in d:
                raise ValueError(
                    f"PdL2AdapterConfig: missing required field {key!r}"
                )

        role = d["role"]
        if role not in _VALID_ROLES:
            raise ValueError(
                f"PdL2AdapterConfig: role must be 'sender' or 'receiver',"
                f" got {role!r}"
            )

        eviction_config = cls._parse_eviction_config(d)
        persist_config = cls._parse_persist_config(d)

        return cls(
            role=role,
            peer_host=d["peer_host"],
            peer_init_port=list(d["peer_init_port"]),
            peer_alloc_port=list(d["peer_alloc_port"]),
            proxy_host=d.get("proxy_host", ""),
            proxy_port=int(d.get("proxy_port", 0)),
            buffer_size=int(d.get("buffer_size", 67108864)),
            buffer_device=d.get("buffer_device", "cpu"),
            transfer_channel=d.get("transfer_channel", "nixl"),
            nixl_backends=list(d.get("nixl_backends", ["tcp"])),
            eviction_config=eviction_config,
            persist_config=persist_config,
        )

    @classmethod
    def help(cls) -> str:
        """
        Return a help string documenting all config fields.

        Returns:
            A multi-line string describing each field, its type,
            default value, and whether it is required or optional.
        """
        return (
            "PD L2 adapter config fields:\n"
            "\n"
            "Required:\n"
            "  role (str)               : 'sender' or 'receiver'\n"
            "  peer_host (str)          : remote peer hostname or IP address\n"
            "  peer_init_port (list[int]): per-TP-rank NIXL init ports"
            " (e.g. [9051] for TP=1)\n"
            "  peer_alloc_port (list[int]): per-TP-rank alloc ports"
            " (e.g. [9052] for TP=1)\n"
            "\n"
            "Optional:\n"
            "  proxy_host (str)         : proxy notification host;"
            " sender only (default: '')\n"
            "  proxy_port (int)         : proxy notification port;"
            " sender only (default: 0)\n"
            "  buffer_size (int)        : staging buffer size in bytes"
            " (default: 67108864 = 64 MB)\n"
            "  buffer_device (str)      : 'cpu' or 'cuda' (default: 'cpu')\n"
            "  transfer_channel (str)   : 'nixl' or 'mock_memory'"
            " (default: 'nixl')\n"
            "  nixl_backends (list[str]): NIXL transport backends"
            " (default: ['tcp'])\n"
        )


def _create_pd_adapter(
    config: "L2AdapterConfigBase",
    l1_memory_desc: "L1MemoryDesc | None" = None,
) -> "L2AdapterInterface":
    """
    Stub factory for the PD L2 adapter.

    The full PD L2Adapter implementation (store/load/networking) is
    planned for a later PR.  This stub registers ``"pd"`` in the
    factory registry so that ``get_registered_l2_adapter_types()``
    includes it.

    Args:
        config: The adapter config (must be a PdL2AdapterConfig).
        l1_memory_desc: Descriptor of L1 memory (unused here).

    Raises:
        NotImplementedError: Always; implementation is pending.
    """
    raise NotImplementedError(
        "PD L2Adapter implementation is pending. "
        "PdL2AdapterConfig is available for configuration only."
    )


register_l2_adapter_type("pd", PdL2AdapterConfig)
register_l2_adapter_factory("pd", _create_pd_adapter)
