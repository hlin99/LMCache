# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import store_timer
from lmcache.v1.multiprocess.store_timer import StoreTimer


def test_set_path_updates_emitted_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_timer.logger, "isEnabledFor", lambda _level: True)
    debug = MagicMock()
    monkeypatch.setattr(store_timer.logger, "debug", debug)

    timer = StoreTimer("req-1", path="data")
    timer.set_path("pickle")
    timer.mark("fwd_return")
    timer.emit()

    assert debug.call_count == 1
    assert debug.call_args.args[2] == "pickle"
