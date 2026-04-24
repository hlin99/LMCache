# SPDX-License-Identifier: Apache-2.0
"""
PdL2AdapterConfig — typed config for the PD (Prefill-Decode) L2 adapter.

This module contains *only* the config dataclass and its registration.
The adapter implementation (PdL2Adapter, wire-protocol messages,
ReservationManager, etc.) lives in a subsequent PR.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)

_VALID_ROLES = ("sender", "receiver")
_VALID_BUFFER_DEVICES = ("cpu", "cuda")
_VALID_TRANSFER_CHANNELS = ("nixl", "mock_memory")


@dataclass
class PdL2AdapterConfig(L2AdapterConfigBase):
    """
    Configuration for the PD (Prefill-Decode) L2 adapter.

    Required fields:
        role: "sender" or "receiver" — which side of the PD pair this node is.
        peer_host: Hostname or IP address of the remote peer.
        peer_init_port: Per-TP-rank ports used for connection initialisation.
        peer_alloc_port: Per-TP-rank ports used for buffer allocation.

    Optional fields:
        proxy_host: Proxy notification host (default: "").
        proxy_port: Proxy notification port (default: 0, disabled).
        buffer_size: Staging buffer size in bytes (default: 67108864 = 64 MB).
        buffer_device: Device for the staging buffer — "cpu" or "cuda"
            (default: "cpu").
        transfer_channel: KV-transfer backend — "nixl" or "mock_memory"
            (default: "nixl").
        nixl_backends: NIXL transport backends (default: ["tcp"]).
    """

    role: str = field(default="")
    peer_host: str = field(default="")
    peer_init_port: list[int] = field(default_factory=list)
    peer_alloc_port: list[int] = field(default_factory=list)
    proxy_host: str = ""
    proxy_port: int = 0
    buffer_size: int = 67108864  # 64 MB
    buffer_device: str = "cpu"
    transfer_channel: str = "nixl"
    nixl_backends: list[str] = field(default_factory=lambda: ["tcp"])

    @classmethod
    def from_dict(cls, d: dict) -> "PdL2AdapterConfig":
        """
        Build a PdL2AdapterConfig from a dict (e.g. parsed JSON).

        Args:
            d: Adapter spec dict. Must include: role, peer_host,
               peer_init_port, peer_alloc_port.

        Returns:
            A PdL2AdapterConfig instance.

        Raises:
            ValueError: If required fields are missing or field values are
                invalid (bad role / buffer_device / transfer_channel).
        """
        for required in ("role", "peer_host", "peer_init_port", "peer_alloc_port"):
            if required not in d:
                raise ValueError(
                    "PdL2AdapterConfig: missing required field %r" % required
                )

        role = d["role"]
        if role not in _VALID_ROLES:
            raise ValueError(
                "PdL2AdapterConfig: role must be one of %s, got %r"
                % (list(_VALID_ROLES), role)
            )

        buffer_device = d.get("buffer_device", "cpu")
        if buffer_device not in _VALID_BUFFER_DEVICES:
            raise ValueError(
                "PdL2AdapterConfig: buffer_device must be one of %s, got %r"
                % (list(_VALID_BUFFER_DEVICES), buffer_device)
            )

        transfer_channel = d.get("transfer_channel", "nixl")
        if transfer_channel not in _VALID_TRANSFER_CHANNELS:
            raise ValueError(
                "PdL2AdapterConfig: transfer_channel must be one of %s, got %r"
                % (list(_VALID_TRANSFER_CHANNELS), transfer_channel)
            )

        cfg = cls(
            role=role,
            peer_host=d["peer_host"],
            peer_init_port=list(d["peer_init_port"]),
            peer_alloc_port=list(d["peer_alloc_port"]),
            proxy_host=d.get("proxy_host", ""),
            proxy_port=int(d.get("proxy_port", 0)),
            buffer_size=int(d.get("buffer_size", 67108864)),
            buffer_device=buffer_device,
            transfer_channel=transfer_channel,
            nixl_backends=list(d.get("nixl_backends", ["tcp"])),
        )
        cfg.eviction_config = cls._parse_eviction_config(d)
        cfg.persist_config = cls._parse_persist_config(d)
        return cfg

    @classmethod
    def help(cls) -> str:
        """
        Return a help string listing all config fields.

        Returns:
            Multi-line string with field names, types, and defaults.
        """
        return (
            "PdL2AdapterConfig fields:\n"
            "  [required]\n"
            "  - role (str): 'sender' or 'receiver'\n"
            "  - peer_host (str): remote peer hostname or IP address\n"
            "  - peer_init_port (list[int]): per-TP-rank init ports\n"
            "  - peer_alloc_port (list[int]): per-TP-rank alloc ports\n"
            "  [optional]\n"
            "  - proxy_host (str, default=''): proxy notification host\n"
            "  - proxy_port (int, default=0): proxy notification port\n"
            "  - buffer_size (int, default=67108864): staging buffer size in bytes\n"
            "  - buffer_device (str, default='cpu'): 'cpu' or 'cuda'\n"
            "  - transfer_channel (str, default='nixl'): 'nixl' or 'mock_memory'\n"
            "  - nixl_backends (list[str], default=['tcp']): NIXL transport backends\n"
            "  - persist_enabled (bool, default=True): keep data at shutdown\n"
            "  - eviction (dict, optional): L2 eviction policy config"
        )


register_l2_adapter_type("pd", PdL2AdapterConfig)
