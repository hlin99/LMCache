# SPDX-License-Identifier: Apache-2.0
"""Unit tests for lmcache.perf_step_timer.PerfStepTimer."""

# Standard
import logging
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.perf_step_timer import PerfStepTimer

_LOGGER_NAME = "lmcache.perf_step_timer"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_perf_step_timer_debug():
    """Set the perf_step_timer logger to DEBUG for every test in this module.

    This ensures PerfStepTimer() constructed during a test has _enabled=True.
    The original level is restored after each test.
    """
    store_logger = logging.getLogger(_LOGGER_NAME)
    orig_level = store_logger.level
    store_logger.setLevel(logging.DEBUG)
    yield
    store_logger.setLevel(orig_level)


# ---------------------------------------------------------------------------
# Basic mark / emit behaviour (unchanged from before)
# ---------------------------------------------------------------------------


class TestMarkEmit:
    def test_emit_produces_log(self, caplog):
        timer = PerfStepTimer(prefix="req-1")
        t0 = 1000.0
        with patch("lmcache.perf_step_timer.time") as mock_time:
            mock_time.perf_counter.side_effect = [t0, t0 + 0.003, t0 + 0.005]
            timer.mark("op", "start")
            timer.mark("op", "mid")
            timer.mark("op", "end")

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            timer.emit("op")

        assert "[PERF-STEP-TIMING]" in caplog.text
        assert "prefix=req-1" in caplog.text
        assert "name=op" in caplog.text
        assert "total=" in caplog.text

    def test_emit_consumes_group(self, caplog):
        """A second emit() after the first produces no additional log line."""
        timer = PerfStepTimer()
        t0 = 1000.0
        with patch("lmcache.perf_step_timer.time") as mock_time:
            mock_time.perf_counter.side_effect = [t0, t0 + 0.001]
            timer.mark("op", "a")
            timer.mark("op", "b")

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            timer.emit("op")
            first_count = caplog.text.count("[PERF-STEP-TIMING]")
            timer.emit("op")  # group already consumed — should be a no-op
            second_count = caplog.text.count("[PERF-STEP-TIMING]")

        assert first_count == 1
        assert second_count == 1  # no additional log line

    def test_emit_unknown_name_is_noop(self, caplog):
        timer = PerfStepTimer()
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            timer.emit("nonexistent")
        assert "[PERF-STEP-TIMING]" not in caplog.text

    def test_emit_single_step_is_noop(self, caplog):
        timer = PerfStepTimer()
        with patch("lmcache.perf_step_timer.time") as mock_time:
            mock_time.perf_counter.return_value = 1000.0
            timer.mark("op", "only")
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            timer.emit("op")
        assert "[PERF-STEP-TIMING]" not in caplog.text

    def test_emit_all_emits_multiple_groups(self, caplog):
        timer = PerfStepTimer()
        t0 = 1000.0
        with patch("lmcache.perf_step_timer.time") as mock_time:
            mock_time.perf_counter.side_effect = [t0, t0 + 0.001, t0, t0 + 0.002]
            timer.mark("a", "start")
            timer.mark("a", "end")
            timer.mark("b", "start")
            timer.mark("b", "end")

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            timer.emit_all()
            first_count = caplog.text.count("[PERF-STEP-TIMING]")
            timer.emit_all()  # all groups consumed — should be a no-op
            second_count = caplog.text.count("[PERF-STEP-TIMING]")

        assert first_count == 2
        assert second_count == 2  # no additional log lines

    def test_no_prefix_in_log(self, caplog):
        timer = PerfStepTimer()
        t0 = 1000.0
        with patch("lmcache.perf_step_timer.time") as mock_time:
            mock_time.perf_counter.side_effect = [t0, t0 + 0.001]
            timer.mark("op", "a")
            timer.mark("op", "b")
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            timer.emit("op")
        assert "prefix=" not in caplog.text


# ---------------------------------------------------------------------------
# TTL auto-emit behaviour
# ---------------------------------------------------------------------------


class TestTTLAutoEmit:
    def test_stale_group_auto_emitted_on_next_mark(self, caplog):
        """A group not marked for > ttl seconds is auto-emitted by the next mark."""
        timer = PerfStepTimer(ttl=5.0)
        t0 = 1000.0

        with patch("lmcache.perf_step_timer.time") as mock_time:
            mock_time.perf_counter.side_effect = [t0, t0 + 0.001]
            timer.mark("stale", "a")
            timer.mark("stale", "b")

        # Trigger auto-emit with a mark on a different group 6 seconds later
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            with patch("lmcache.perf_step_timer.time") as mock_time:
                mock_time.perf_counter.return_value = t0 + 6.0  # > ttl=5s
                timer.mark("fresh", "x")

        assert "[PERF-STEP-TIMING]" in caplog.text
        assert "name=stale" in caplog.text

        # Verify the group was consumed: a second explicit emit produces nothing
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            count_before = caplog.text.count("[PERF-STEP-TIMING]")
            timer.emit("stale")
            assert caplog.text.count("[PERF-STEP-TIMING]") == count_before

    def test_non_stale_group_not_auto_emitted(self, caplog):
        """A group within the TTL window is NOT auto-emitted."""
        timer = PerfStepTimer(ttl=5.0)
        t0 = 1000.0

        with patch("lmcache.perf_step_timer.time") as mock_time:
            mock_time.perf_counter.side_effect = [t0, t0 + 0.001]
            timer.mark("recent", "a")
            timer.mark("recent", "b")

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            with patch("lmcache.perf_step_timer.time") as mock_time:
                mock_time.perf_counter.return_value = t0 + 3.0  # < ttl=5s
                timer.mark("other", "x")

        assert "name=recent" not in caplog.text

        # The group must still be present — explicit emit should produce a log line
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            timer.emit("recent")
        assert "name=recent" in caplog.text

    def test_current_group_never_auto_emitted_by_own_mark(self, caplog):
        """The group being marked is never considered stale in the same call."""
        timer = PerfStepTimer(ttl=5.0)
        t0 = 1000.0

        with patch("lmcache.perf_step_timer.time") as mock_time:
            mock_time.perf_counter.side_effect = [t0]
            timer.mark("op", "first")

        # 6 seconds later, mark the SAME group again — should NOT auto-emit it
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            with patch("lmcache.perf_step_timer.time") as mock_time:
                mock_time.perf_counter.return_value = t0 + 6.0
                timer.mark("op", "second")

        assert "name=op" not in caplog.text

        # The group must still be present — explicit emit should produce a log line
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            timer.emit("op")
        assert "name=op" in caplog.text

    def test_custom_ttl_respected(self, caplog):
        """A custom ttl value controls when auto-emit fires."""
        timer = PerfStepTimer(ttl=2.0)
        t0 = 1000.0

        with patch("lmcache.perf_step_timer.time") as mock_time:
            mock_time.perf_counter.side_effect = [t0, t0 + 0.001]
            timer.mark("old", "a")
            timer.mark("old", "b")

        # 3 seconds later — stale for ttl=2.0 but NOT for default ttl=5.0
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            with patch("lmcache.perf_step_timer.time") as mock_time:
                mock_time.perf_counter.return_value = t0 + 3.0
                timer.mark("new", "x")

        assert "[PERF-STEP-TIMING]" in caplog.text
        assert "name=old" in caplog.text

    def test_auto_emit_log_format_matches_explicit_emit(self, caplog):
        """Auto-emitted log lines use the same [PERF-STEP-TIMING] format."""
        timer = PerfStepTimer(prefix="p", ttl=5.0)
        t0 = 1000.0

        with patch("lmcache.perf_step_timer.time") as mock_time:
            mock_time.perf_counter.side_effect = [t0, t0 + 0.005]
            timer.mark("g", "start")
            timer.mark("g", "end")

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            with patch("lmcache.perf_step_timer.time") as mock_time:
                mock_time.perf_counter.return_value = t0 + 6.0
                timer.mark("other", "x")

        assert "[PERF-STEP-TIMING]" in caplog.text
        assert "prefix=p" in caplog.text
        assert "name=g" in caplog.text
        assert "start -> end=" in caplog.text
        assert "total=" in caplog.text

    def test_multiple_stale_groups_all_auto_emitted(self, caplog):
        """All stale groups are auto-emitted in a single mark() call."""
        timer = PerfStepTimer(ttl=5.0)
        t0 = 1000.0

        with patch("lmcache.perf_step_timer.time") as mock_time:
            mock_time.perf_counter.side_effect = [
                t0,
                t0 + 0.001,  # group "a"
                t0,
                t0 + 0.002,  # group "b"
            ]
            timer.mark("a", "s1")
            timer.mark("a", "s2")
            timer.mark("b", "s1")
            timer.mark("b", "s2")

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            with patch("lmcache.perf_step_timer.time") as mock_time:
                mock_time.perf_counter.return_value = t0 + 10.0
                timer.mark("fresh", "x")

        assert caplog.text.count("[PERF-STEP-TIMING]") == 2

        # Both stale groups consumed; "fresh" still present
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            count_before = caplog.text.count("[PERF-STEP-TIMING]")
            timer.emit("a")  # already consumed — no-op
            timer.emit("b")  # already consumed — no-op
            assert caplog.text.count("[PERF-STEP-TIMING]") == count_before

            timer.emit("fresh")  # only 1 step, so emit is a no-op
            assert caplog.text.count("[PERF-STEP-TIMING]") == count_before


# ---------------------------------------------------------------------------
# Disabled timer
# ---------------------------------------------------------------------------


class TestDisabledTimer:
    def test_disabled_timer_mark_is_noop(self):
        """mark() on a disabled timer raises no error."""
        logging.getLogger(_LOGGER_NAME).setLevel(logging.WARNING)
        timer = PerfStepTimer()
        assert not timer.is_enabled
        timer.mark("op", "step")  # must not raise AttributeError

    def test_disabled_timer_emit_is_noop(self):
        """emit() on a disabled timer raises no error."""
        logging.getLogger(_LOGGER_NAME).setLevel(logging.WARNING)
        timer = PerfStepTimer()
        assert not timer.is_enabled
        timer.emit("op")  # must not raise

    def test_default_constructor_disabled_when_not_debug(self):
        """PerfStepTimer constructed at WARNING level is disabled."""
        logging.getLogger(_LOGGER_NAME).setLevel(logging.WARNING)
        timer = PerfStepTimer()
        assert not timer.is_enabled
