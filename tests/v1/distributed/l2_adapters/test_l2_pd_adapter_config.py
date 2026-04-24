# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PdL2AdapterConfig.

Covers:
- Minimal sender / receiver round-trips
- Missing required fields raise ValueError
- Invalid role / buffer_device / transfer_channel raise ValueError
- help() output contains every field name
- "pd" is present in the registered adapter type list
- All fields survive a full round-trip
- Default values are applied when optional fields are omitted
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
    "peer_host": "192.168.1.10",
    "peer_init_port": [9000, 9001],
    "peer_alloc_port": [9100, 9101],
}

_RECEIVER_REQUIRED: dict[str, Any] = {
    "role": "receiver",
    "peer_host": "192.168.1.20",
    "peer_init_port": [9000, 9001],
    "peer_alloc_port": [9100, 9101],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_minimal_sender_config() -> None:
    """Only required fields — sender role."""
    cfg = PdL2AdapterConfig.from_dict(_SENDER_REQUIRED)
    assert cfg.role == "sender"
    assert cfg.peer_host == "192.168.1.10"
    assert cfg.peer_init_port == [9000, 9001]
    assert cfg.peer_alloc_port == [9100, 9101]


def test_parse_minimal_receiver_config() -> None:
    """Only required fields — receiver role."""
    cfg = PdL2AdapterConfig.from_dict(_RECEIVER_REQUIRED)
    assert cfg.role == "receiver"
    assert cfg.peer_host == "192.168.1.20"
    assert cfg.peer_init_port == [9000, 9001]
    assert cfg.peer_alloc_port == [9100, 9101]


def test_fail_on_missing_required() -> None:
    """Each required field, when absent, must raise ValueError."""
    required_fields = ("role", "peer_host", "peer_init_port", "peer_alloc_port")
    for field in required_fields:
        incomplete = {k: v for k, v in _SENDER_REQUIRED.items() if k != field}
        with pytest.raises(ValueError, match=field):
            PdL2AdapterConfig.from_dict(incomplete)


def test_fail_on_invalid_role() -> None:
    """An unrecognised role must raise ValueError."""
    bad = {**_SENDER_REQUIRED, "role": "invalid"}
    with pytest.raises(ValueError, match="role"):
        PdL2AdapterConfig.from_dict(bad)


def test_help_contains_all_field_names() -> None:
    """help() output must mention every field name."""
    text = PdL2AdapterConfig.help()
    expected_fields = [
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
        "persist_enabled",
        "eviction",
    ]
    for name in expected_fields:
        assert name in text, f"help() is missing field {name!r}"


def test_registered_in_factory() -> None:
    """'pd' must appear in the registered adapter type list."""
    assert "pd" in get_registered_l2_adapter_types()


def test_all_fields_round_trip() -> None:
    """All fields, including optional ones, survive from_dict."""
    full: dict[str, Any] = {
        "role": "sender",
        "peer_host": "10.0.0.1",
        "peer_init_port": [8000],
        "peer_alloc_port": [8100],
        "proxy_host": "proxy.local",
        "proxy_port": 5555,
        "buffer_size": 134217728,  # 128 MB
        "buffer_device": "cuda",
        "transfer_channel": "mock_memory",
        "nixl_backends": ["ucx", "tcp"],
        "persist_enabled": False,
    }
    cfg = PdL2AdapterConfig.from_dict(full)
    assert cfg.role == "sender"
    assert cfg.peer_host == "10.0.0.1"
    assert cfg.peer_init_port == [8000]
    assert cfg.peer_alloc_port == [8100]
    assert cfg.proxy_host == "proxy.local"
    assert cfg.proxy_port == 5555
    assert cfg.buffer_size == 134217728
    assert cfg.buffer_device == "cuda"
    assert cfg.transfer_channel == "mock_memory"
    assert cfg.nixl_backends == ["ucx", "tcp"]
    assert cfg.persist_config.persist_enabled is False


def test_defaults_applied() -> None:
    """Optional fields default to documented values when omitted."""
    cfg = PdL2AdapterConfig.from_dict(_SENDER_REQUIRED)
    assert cfg.proxy_host == ""
    assert cfg.proxy_port == 0
    assert cfg.buffer_size == 67108864
    assert cfg.buffer_device == "cpu"
    assert cfg.transfer_channel == "nixl"
    assert cfg.nixl_backends == ["tcp"]
    assert cfg.eviction_config is None
    assert cfg.persist_config.persist_enabled is True


def test_fail_on_invalid_buffer_device() -> None:
    """buffer_device='gpu' (not 'cuda') must raise ValueError."""
    bad = {**_SENDER_REQUIRED, "buffer_device": "gpu"}
    with pytest.raises(ValueError, match="buffer_device"):
        PdL2AdapterConfig.from_dict(bad)


def test_fail_on_invalid_transfer_channel() -> None:
    """An unrecognised transfer_channel must raise ValueError."""
    bad = {**_SENDER_REQUIRED, "transfer_channel": "invalid"}
    with pytest.raises(ValueError, match="transfer_channel"):
        PdL2AdapterConfig.from_dict(bad)
