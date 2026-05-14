# SPDX-License-Identifier: Apache-2.0
"""Regression tests for optional imports in ``lmcache bench kvcache``."""

# Standard
import importlib


def test_cli_main_imports_without_torch() -> None:
    """CLI module import must not require the full runtime package."""
    assert importlib.import_module("lmcache.cli.main") is not None
