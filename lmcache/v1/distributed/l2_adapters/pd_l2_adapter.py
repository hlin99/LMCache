# SPDX-License-Identifier: Apache-2.0
"""
PD (Prefill-Decode) L2 adapter config.

Stores the configuration needed to connect a sender (prefill) node to a
receiver (decode) node via a staging buffer and a transfer channel (NIXL or
mock_memory).  Only the config class and its registration are defined here;
the adapter implementation lives in a later PR.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)


class PdL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for the PD (Prefill-Decode) L2 adapter.

    Fields:
    - role: 'sender' (prefill) or 'receiver' (decode).
    - peer_host: hostname or IP of the remote peer.
    - peer_init_port: per-TP-rank list of init ports on the peer.
    - peer_alloc_port: per-TP-rank list of alloc ports on the peer.
    - proxy_host: proxy notification host (default: '127.0.0.1').
    - proxy_port: proxy notification port (default: 6688).
    - buffer_size: staging buffer size in bytes (default: 1 GiB).
    - buffer_device: device for the staging buffer, 'cpu' or 'cuda'
      (default: 'cpu').
    - transfer_channel: transfer backend, 'nixl' or 'mock_memory'
      (default: 'nixl').
    - nixl_backends: NIXL transport backends (default: ['tcp']).
    """

    def __init__(
        self,
        role: str,
        peer_host: str,
        peer_init_port: list[int],
        peer_alloc_port: list[int],
        proxy_host: str = "127.0.0.1",
        proxy_port: int = 6688,
        buffer_size: int = 1073741824,
        buffer_device: str = "cpu",
        transfer_channel: str = "nixl",
        nixl_backends: list[str] | None = None,
    ):
        """Initialize PdL2AdapterConfig.

        Args:
            role: 'sender' (prefill node) or 'receiver' (decode node).
            peer_host: Hostname or IP address of the remote peer.
            peer_init_port: Per-TP-rank list of initialization port numbers
                on the peer.
            peer_alloc_port: Per-TP-rank list of alloc port numbers on the
                peer.
            proxy_host: Proxy notification host (default: '127.0.0.1').
            proxy_port: Proxy notification port (default: 6688).
            buffer_size: Staging buffer size in bytes.
            buffer_device: Device for the staging buffer ('cpu' or 'cuda').
            transfer_channel: Transfer backend ('nixl' or 'mock_memory').
            nixl_backends: List of NIXL transport backend names.  Defaults
                to ``['tcp']``.
        """
        self.role = role
        self.peer_host = peer_host
        self.peer_init_port = peer_init_port
        self.peer_alloc_port = peer_alloc_port
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.buffer_size = buffer_size
        self.buffer_device = buffer_device
        self.transfer_channel = transfer_channel
        self.nixl_backends = nixl_backends or ["tcp"]

    @classmethod
    def from_dict(cls, d: dict) -> "PdL2AdapterConfig":
        """Build a PdL2AdapterConfig from a dict (e.g. parsed JSON).

        Args:
            d: Adapter spec dict.  Must include ``role``, ``peer_host``,
                ``peer_init_port``, and ``peer_alloc_port``.

        Returns:
            A new PdL2AdapterConfig instance.

        Raises:
            ValueError: If a required field is missing or a value is
                outside the allowed set.
        """
        role = d.get("role")
        if role not in ("sender", "receiver"):
            raise ValueError("role must be 'sender' or 'receiver', got %r" % role)

        peer_host = d.get("peer_host")
        if not peer_host:
            raise ValueError("peer_host is required")

        peer_init_port = d.get("peer_init_port")
        if not peer_init_port:
            raise ValueError("peer_init_port is required")

        peer_alloc_port = d.get("peer_alloc_port")
        if not peer_alloc_port:
            raise ValueError("peer_alloc_port is required")

        buffer_device = d.get("buffer_device", "cpu")
        if buffer_device not in ("cpu", "cuda"):
            raise ValueError(
                "buffer_device must be 'cpu' or 'cuda', got %r" % buffer_device
            )

        transfer_channel = d.get("transfer_channel", "nixl")
        if transfer_channel not in ("nixl", "mock_memory"):
            raise ValueError(
                "transfer_channel must be 'nixl' or 'mock_memory', got %r"
                % transfer_channel
            )

        cfg = cls(
            role=role,
            peer_host=peer_host,
            peer_init_port=list(peer_init_port),
            peer_alloc_port=list(peer_alloc_port),
            proxy_host=d.get("proxy_host", "127.0.0.1"),
            proxy_port=int(d.get("proxy_port", 6688)),
            buffer_size=int(d.get("buffer_size", 1073741824)),
            buffer_device=buffer_device,
            transfer_channel=transfer_channel,
            nixl_backends=list(d.get("nixl_backends", ["tcp"])),
        )
        return cfg

    @classmethod
    def help(cls) -> str:
        """Return a help string describing PdL2AdapterConfig fields.

        Returns:
            A multi-line string listing all config fields with types,
            default values, and whether they are required.
        """
        return (
            "PD L2 adapter config fields:\n"
            "- role (str): 'sender' or 'receiver' (required)\n"
            "- peer_host (str): remote peer hostname or IP (required)\n"
            "- peer_init_port (list[int]): per-TP-rank init ports (required)\n"
            "- peer_alloc_port (list[int]): per-TP-rank alloc ports (required)\n"
            "- proxy_host (str): proxy notification host (default: '127.0.0.1')\n"
            "- proxy_port (int): proxy notification port (default: 6688)\n"
            "- buffer_size (int): staging buffer size in bytes (default: 1073741824)\n"
            "- buffer_device (str): 'cpu' or 'cuda' (default: 'cpu')\n"
            "- transfer_channel (str): 'nixl' or 'mock_memory' (default: 'nixl')\n"
            "- nixl_backends (list[str]): NIXL transport backends (default: ['tcp'])"
        )


register_l2_adapter_type("pd", PdL2AdapterConfig)
