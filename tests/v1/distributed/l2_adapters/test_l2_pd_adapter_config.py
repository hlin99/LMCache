# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PdL2AdapterConfig.

Covers correct parsing, validation errors, help-string completeness,
factory registration, full field round-trip, and default values.
"""

# Standard
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.config import get_registered_l2_adapter_types
from lmcache.v1.distributed.l2_adapters.pd_l2_adapter import PdL2AdapterConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SENDER_REQUIRED: dict[str, Any] = {
    "role": "sender",
    "peer_host": "10.0.0.1",
    "peer_init_port": [9051],
    "peer_alloc_port": [9052],
}

_RECEIVER_REQUIRED: dict[str, Any] = {
    "role": "receiver",
    "peer_host": "10.0.0.2",
    "peer_init_port": [9051],
    "peer_alloc_port": [9052],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_minimal_sender_config() -> None:
    """Parse a dict with only required fields for the sender role."""
    cfg = PdL2AdapterConfig.from_dict(_SENDER_REQUIRED)
    assert cfg.role == "sender"
    assert cfg.peer_host == "10.0.0.1"
    assert cfg.peer_init_port == [9051]
    assert cfg.peer_alloc_port == [9052]


def test_parse_minimal_receiver_config() -> None:
    """Parse a dict with only required fields for the receiver role."""
    cfg = PdL2AdapterConfig.from_dict(_RECEIVER_REQUIRED)
    assert cfg.role == "receiver"
    assert cfg.peer_host == "10.0.0.2"
    assert cfg.peer_init_port == [9051]
    assert cfg.peer_alloc_port == [9052]


def test_fail_on_missing_required() -> None:
    """Omitting any required field must raise ValueError."""
    for key in ("role", "peer_host", "peer_init_port", "peer_alloc_port"):
        d = dict(_SENDER_REQUIRED)
        del d[key]
        with pytest.raises(ValueError, match=key):
            PdL2AdapterConfig.from_dict(d)


def test_fail_on_invalid_role() -> None:
    """A role value that is not 'sender' or 'receiver' must raise ValueError."""
    d = dict(_SENDER_REQUIRED)
    d["role"] = "invalid"
    with pytest.raises(ValueError):
        PdL2AdapterConfig.from_dict(d)


def test_help_contains_all_field_names() -> None:
    """The help() output must mention every public field name."""
    help_text = PdL2AdapterConfig.help()
    for field_name in (
        "role",
        "peer_host",
        "peer_init_port",
        "peer_alloc_port",
        "proxy_host",
        "proxy_port",
        "buffer_size",
        "buffer_device",
        "transfer_channel",
        "nixl_backends",
    ):
        assert field_name in help_text, (
            f"Field {field_name!r} not found in PdL2AdapterConfig.help()"
        )


def test_registered_in_factory() -> None:
    """The 'pd' type must appear in the L2 adapter type registry."""
    assert "pd" in get_registered_l2_adapter_types()


def test_all_fields_round_trip() -> None:
    """Pass all supported fields and verify each is reflected on the instance."""
    d = {
        "role": "sender",
        "peer_host": "192.168.1.1",
        "peer_init_port": [9051, 9053],
        "peer_alloc_port": [9052, 9054],
        "proxy_host": "proxy.example.com",
        "proxy_port": 8080,
        "buffer_size": 134217728,
        "buffer_device": "cuda",
        "transfer_channel": "mock_memory",
        "nixl_backends": ["tcp", "rdma"],
    }
    cfg = PdL2AdapterConfig.from_dict(d)
    assert cfg.role == "sender"
    assert cfg.peer_host == "192.168.1.1"
    assert cfg.peer_init_port == [9051, 9053]
    assert cfg.peer_alloc_port == [9052, 9054]
    assert cfg.proxy_host == "proxy.example.com"
    assert cfg.proxy_port == 8080
    assert cfg.buffer_size == 134217728
    assert cfg.buffer_device == "cuda"
    assert cfg.transfer_channel == "mock_memory"
    assert cfg.nixl_backends == ["tcp", "rdma"]


def test_defaults_applied() -> None:
    """Optional fields must take their documented default values when omitted."""
    cfg = PdL2AdapterConfig.from_dict(_SENDER_REQUIRED)
    assert cfg.proxy_host == ""
    assert cfg.proxy_port == 0
    assert cfg.buffer_size == 67108864  # 64 MB
    assert cfg.buffer_device == "cpu"
    assert cfg.transfer_channel == "nixl"
    assert cfg.nixl_backends == ["tcp"]
    assert cfg.eviction_config is None
