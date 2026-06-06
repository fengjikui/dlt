"""Tests for interval computation and scheduling primitives."""

from datetime import datetime, timezone  # noqa: I251
from typing import Optional
from zoneinfo import ZoneInfo

import pytest

from dlt.common.configuration.container import Container
from dlt.common.pendulum import pendulum
from dlt.common.time import ensure_pendulum_datetime_utc
from dlt.common.typing import TTimeInterval

from dlt._workspace.deployment.exceptions import InvalidTrigger
from dlt._workspace.deployment.interval import (
    compute_run_interval,
    cron_floor,
    cron_lag,
    full_days_interval,
    lag_interval,
    next_scheduled_run,
    resolve_interval_spec,
)
from dlt._workspace.deployment.typing import (
    TIntervalSpec,
    TTrigger,
)
from dlt.extract.incremental.context import (
    TimeIntervalContext,
    interval as interval_accessor,
)


def test_resolve_with_explicit_end() -> None:
    spec: TIntervalSpec = {"start": "2024-01-01T00:00:00Z", "end": "2024-01-05T00:00:00Z"}
    result = resolve_interval_spec(spec, "0 0 * * *")
    assert result == (
        ensure_pendulum_datetime_utc("2024-01-01"),
        ensure_pendulum_datetime_utc("2024-01-05"),
    )


def test_resolve_open_ended_uses_last_cron_tick() -> None:
    """Open-ended spec resolves end to the last elapsed cron tick."""
    spec: TIntervalSpec = {"start": "2020-01-01T00:00:00Z"}
    start, end = resolve_interval_spec(spec, "0 0 * * *")
    assert start == ensure_pendulum_datetime_utc("2020-01-01")
    assert end <= pendulum.now("UTC")
    assert end.hour == 0 and end.minute == 0


def test_resolve_interval_spec_snaps_start() -> None:
    """When start is between cron ticks, it snaps backward."""
    spec: TIntervalSpec = {"start": "2024-01-01T06:30:00Z", "end": "2024-01-05T00:00:00Z"}
    start, end = resolve_interval_spec(spec, "0 0 * * *")
    assert start == ensure_pendulum_datetime_utc("2024-01-01")
    assert end == ensure_pendulum_datetime_utc("2024-01-05")


def test_resolve_explicit_end_snapped() -> None:
    """Explicit end between cron ticks snaps backward."""
    spec: TIntervalSpec = {"start": "2024-01-01T00:00:00Z", "end": "2024-01-05T06:30:00Z"}
    start, end = resolve_interval_spec(spec, "0 0 * * *")
    assert start == ensure_pendulum_datetime_utc("2024-01-01")
    assert end == ensure_pendulum_datetime_utc("2024-01-05")


@pytest.mark.parametrize(
    "cron_expr,dt,expected",
    [
        ("0 0 * * *", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
        ("0 0 * * *", "2024-01-01T06:30:00Z", "2024-01-01T00:00:00Z"),
        ("*/3 * * * *", "2024-01-01T11:40:00Z", "2024-01-01T11:39:00Z"),
        ("*/3 * * * *", "2024-01-01T11:42:00Z", "2024-01-01T11:42:00Z"),
        ("* * * * *", "2024-01-01T11:40:00Z", "2024-01-01T11:40:00Z"),
    ],
    ids=["daily-aligned", "daily-misaligned", "3min-misaligned", "3min-aligned", "1min-aligned"],
)
def test_cron_floor(cron_expr: str, dt: str, expected: str) -> None:
    assert cron_floor(cron_expr, ensure_pendulum_datetime_utc(dt)) == ensure_pendulum_datetime_utc(
        expected
    )


@pytest.mark.parametrize(
    "tz,cron,spec_start,spec_end,expected_start,expected_end",
    [
        # 08:00 CET (winter, +01:00) → 07:00 UTC
        (
            "Europe/Berlin",
            "0 8 * * *",
            "2024-01-01T00:00:00Z",
            "2024-01-05T00:00:00Z",
            "2023-12-31T07:00:00Z",
            "2024-01-04T07:00:00Z",
        ),
        # 08:00 CEST (summer, +02:00) → 06:00 UTC
        (
            "Europe/Berlin",
            "0 8 * * *",
            "2024-06-01T00:00:00Z",
            "2024-06-05T00:00:00Z",
            "2024-05-31T06:00:00Z",
            "2024-06-04T06:00:00Z",
        ),
        # 08:00 EDT (summer, -04:00) → 12:00 UTC
        (
            "America/New_York",
            "0 8 * * *",
            "2024-07-01T00:00:00Z",
            "2024-07-05T00:00:00Z",
            "2024-06-30T12:00:00Z",
            "2024-07-04T12:00:00Z",
        ),
        # UTC baseline — no shift
        (
            "UTC",
            "0 0 * * *",
            "2024-01-01T00:00:00Z",
            "2024-01-05T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "2024-01-05T00:00:00Z",
        ),
    ],
    ids=["berlin-winter", "berlin-summer", "new-york-summer", "utc-baseline"],
)
def test_resolve_interval_spec_returns_utc(
    tz: str,
    cron: str,
    spec_start: str,
    spec_end: str,
    expected_start: str,
    expected_end: str,
) -> None:
    """Non-UTC tz drives cron alignment; returned datetimes are always UTC."""
    spec: TIntervalSpec = {"start": spec_start, "end": spec_end}
    start, end = resolve_interval_spec(spec, cron, tz=tz)
    assert start.tzinfo == timezone.utc
    assert end.tzinfo == timezone.utc
    assert start == ensure_pendulum_datetime_utc(expected_start)
    assert end == ensure_pendulum_datetime_utc(expected_end)


@pytest.mark.parametrize(
    "tz,year,month,day,hour",
    [
        ("Europe/Berlin", 2024, 3, 31, 8),  # DST-transition day
        ("Europe/Berlin", 2024, 6, 15, 8),  # CEST
        ("Europe/Berlin", 2024, 12, 15, 8),  # CET
        ("America/New_York", 2024, 3, 10, 8),  # DST-transition day
        ("Pacific/Auckland", 2024, 9, 29, 8),  # southern-hemisphere DST
        ("UTC", 2024, 1, 1, 0),
    ],
    ids=["berlin-dst-day", "berlin-cest", "berlin-cet", "ny-dst-day", "auckland", "utc"],
)
def test_interval_utc_round_trip_through_launcher_boundary(
    tz: str, year: int, month: int, day: int, hour: int
) -> None:
    """UTC ISO → parse → re-apply IANA tz equals original local datetime."""
    target_tz = ZoneInfo(tz)
    local = datetime(year, month, day, hour, tzinfo=target_tz)
    iso_utc = local.astimezone(timezone.utc).isoformat()
    restored = ensure_pendulum_datetime_utc(iso_utc).astimezone(target_tz)
    assert restored == local
    assert isinstance(restored.tzinfo, ZoneInfo)
    assert restored.tzinfo.key == tz


@pytest.mark.parametrize(
    "trigger,tz,prev,now_ref,expected_at",
    [
        # schedule: next cron tick after now
        (
            "schedule:0 0 * * *",
            "UTC",
            None,
            "2024-06-15T12:00:00Z",
            "2024-06-16T00:00:00Z",
        ),
        # schedule: exactly on a tick — next is the following tick
        (
            "schedule:0 0 * * *",
            "UTC",
            None,
            "2024-06-15T00:00:00Z",
            "2024-06-16T00:00:00Z",
        ),
        # schedule: with timezone
        (
            "schedule:0 8 * * *",
            "US/Eastern",
            None,
            "2024-06-15T11:00:00Z",  # 07:00 ET, before 08:00
            "2024-06-15T12:00:00Z",  # 08:00 ET
        ),
        # every: first run (no prev), waits one period
        (
            "every:1h",
            "UTC",
            None,
            "2024-06-15T10:30:00Z",
            "2024-06-15T11:30:00Z",
        ),
        # every: prev exists, next = prev + period
        (
            "every:1h",
            "UTC",
            "2024-06-15T10:00:00Z",
            "2024-06-15T10:30:00Z",
            "2024-06-15T11:00:00Z",
        ),
        # every: prev + period is in the past — clamp to now + period
        (
            "every:1h",
            "UTC",
            "2024-06-15T08:00:00Z",
            "2024-06-15T12:00:00Z",
            "2024-06-15T13:00:00Z",
        ),
        # once: future datetime
        (
            "once:2024-12-31T23:59:59Z",
            "UTC",
            None,
            "2024-06-15T00:00:00Z",
            "2024-12-31T23:59:59Z",
        ),
        # once: past datetime — clamp to now
        (
            "once:2024-01-01T00:00:00Z",
            "UTC",
            None,
            "2024-06-15T00:00:00Z",
            "2024-06-15T00:00:00Z",
        ),
    ],
    ids=[
        "schedule-midday",
        "schedule-on-tick",
        "schedule-timezone",
        "every-first-waits-period",
        "every-with-prev",
        "every-clamp-to-now-plus-period",
        "once-future",
        "once-past-clamped",
    ],
)
def test_next_scheduled_run(
    trigger: str,
    tz: str,
    prev: Optional[str],
    now_ref: str,
    expected_at: str,
) -> None:
    scheduled_at = next_scheduled_run(
        TTrigger(trigger),
        ensure_pendulum_datetime_utc(now_ref),
        tz=tz,
        prev_scheduled_run=ensure_pendulum_datetime_utc(prev) if prev else None,
    )
    assert scheduled_at == ensure_pendulum_datetime_utc(expected_at)


def test_next_scheduled_run_returns_utc() -> None:
    """All returned datetimes are UTC regardless of tz parameter."""
    scheduled_at = next_scheduled_run(
        TTrigger("schedule:0 8 * * *"),
        ensure_pendulum_datetime_utc("2024-06-15T11:00:00Z"),
        tz="US/Eastern",
    )
    assert scheduled_at.tzname() == "UTC"

    scheduled_every = next_scheduled_run(
        TTrigger("every:1h"),
        ensure_pendulum_datetime_utc("2024-06-15T10:00:00Z"),
    )
    assert scheduled_every.tzname() == "UTC"

    scheduled_once = next_scheduled_run(
        TTrigger("once:2025-01-01T00:00:00Z"),
        ensure_pendulum_datetime_utc("2024-06-15T00:00:00Z"),
    )
    assert scheduled_once.tzname() == "UTC"


def test_next_scheduled_run_rejects_non_timed() -> None:
    """Non-timed triggers raise InvalidTrigger."""
    with pytest.raises(InvalidTrigger, match="not a timed trigger"):
        next_scheduled_run(
            TTrigger("manual:jobs.mod.a"),
            ensure_pendulum_datetime_utc("2024-06-15T00:00:00Z"),
        )


@pytest.mark.parametrize(
    "trigger,prev,now,expected_start,expected_end",
    [
        # ── schedule, prev=None: most recently ELAPSED cron interval ──
        # now mid-period with sub-second precision → [prev_tick, floor(now))
        (
            "schedule:0 * * * *",
            None,
            "2024-06-15T12:37:42.123456Z",
            "2024-06-15T11:00:00Z",
            "2024-06-15T12:00:00Z",
        ),
        # now exactly on a tick → floor includes equality, interval is [prev_tick, now)
        (
            "schedule:0 * * * *",
            None,
            "2024-06-15T12:00:00Z",
            "2024-06-15T11:00:00Z",
            "2024-06-15T12:00:00Z",
        ),
        # now 1μs after a tick → same floor, same result
        (
            "schedule:0 * * * *",
            None,
            "2024-06-15T12:00:00.000001Z",
            "2024-06-15T11:00:00Z",
            "2024-06-15T12:00:00Z",
        ),
        # now 1μs BEFORE a tick → floor is the previous tick, not the upcoming one
        (
            "schedule:0 * * * *",
            None,
            "2024-06-15T11:59:59.999999Z",
            "2024-06-15T10:00:00Z",
            "2024-06-15T11:00:00Z",
        ),
        # ── schedule, prev set: overrides natural start; end is floor(now) ──
        # steady state: prev ON a cron tick, equal to natural_start → same result
        (
            "schedule:0 * * * *",
            "2024-06-15T11:00:00Z",
            "2024-06-15T12:37:42.123Z",
            "2024-06-15T11:00:00Z",
            "2024-06-15T12:00:00Z",
        ),
        # gap-fill: prev OFF a tick (missed cascade / non-aligned watermark)
        (
            "schedule:0 * * * *",
            "2024-06-15T09:15:23.500Z",
            "2024-06-15T12:37:42.123Z",
            "2024-06-15T09:15:23.500Z",
            "2024-06-15T12:00:00Z",
        ),
        # gap-fill with prev on a much earlier cron tick → large window
        (
            "schedule:0 * * * *",
            "2024-06-15T06:00:00Z",
            "2024-06-15T12:37:42.123Z",
            "2024-06-15T06:00:00Z",
            "2024-06-15T12:00:00Z",
        ),
        # now exactly on a tick with prev set → end = now, not next tick
        (
            "schedule:0 * * * *",
            "2024-06-15T09:00:00Z",
            "2024-06-15T12:00:00Z",
            "2024-06-15T09:00:00Z",
            "2024-06-15T12:00:00Z",
        ),
        # ── every, prev=None ──
        (
            "every:1h",
            None,
            "2024-06-15T12:30:00Z",
            "2024-06-15T11:30:00Z",
            "2024-06-15T12:30:00Z",
        ),
        (
            "every:5m",
            None,
            "2024-06-15T12:30:00Z",
            "2024-06-15T12:25:00Z",
            "2024-06-15T12:30:00Z",
        ),
        # ── once, prev=None ──
        (
            "once:2030-01-01T00:00:00Z",
            None,
            "2024-06-15T12:30:00Z",
            "2030-01-01T00:00:00Z",
            "2030-01-01T00:00:00Z",
        ),
        # ── event-like triggers, prev=None ──
        (
            "manual:jobs.mod.a",
            None,
            "2024-06-15T12:30:00Z",
            "2024-06-15T12:30:00Z",
            "2024-06-15T12:30:00Z",
        ),
        (
            "tag:nightly",
            None,
            "2024-06-15T12:30:00Z",
            "2024-06-15T12:30:00Z",
            "2024-06-15T12:30:00Z",
        ),
        (
            "job.success:jobs.mod.upstream",
            None,
            "2024-06-15T12:30:00Z",
            "2024-06-15T12:30:00Z",
            "2024-06-15T12:30:00Z",
        ),
        # ── every, prev set: continuity — [prev, now) ──
        (
            "every:1h",
            "2024-06-15T10:00:00Z",
            "2024-06-15T12:30:00Z",
            "2024-06-15T10:00:00Z",
            "2024-06-15T12:30:00Z",
        ),
        # ── point-in-time triggers with prev set: prev is IGNORED ──
        # once: always [once, once) regardless of prev
        (
            "once:2030-01-01T00:00:00Z",
            "2024-06-15T10:00:00Z",
            "2024-06-15T12:30:00Z",
            "2030-01-01T00:00:00Z",
            "2030-01-01T00:00:00Z",
        ),
        # manual: always [now, now) regardless of prev
        (
            "manual:jobs.mod.a",
            "2024-06-15T10:00:00Z",
            "2024-06-15T12:30:00Z",
            "2024-06-15T12:30:00Z",
            "2024-06-15T12:30:00Z",
        ),
        # tag: always [now, now) regardless of prev
        (
            "tag:nightly",
            "2024-06-15T10:00:00Z",
            "2024-06-15T12:30:00Z",
            "2024-06-15T12:30:00Z",
            "2024-06-15T12:30:00Z",
        ),
        # job.success: always [now, now) regardless of prev
        (
            "job.success:jobs.mod.upstream",
            "2024-06-15T10:00:00Z",
            "2024-06-15T12:30:00Z",
            "2024-06-15T12:30:00Z",
            "2024-06-15T12:30:00Z",
        ),
    ],
    ids=[
        "schedule-prev=None-midperiod-subsec",
        "schedule-prev=None-on-tick",
        "schedule-prev=None-1us-after-tick",
        "schedule-prev=None-1us-before-tick",
        "schedule-prev-on-tick-steady-state",
        "schedule-prev-off-tick-gap-fill",
        "schedule-prev-early-tick-large-gap",
        "schedule-prev-set-now-on-tick",
        "every-prev=None-1h",
        "every-prev=None-5m",
        "once-prev=None",
        "manual-prev=None",
        "tag-prev=None",
        "job.success-prev=None",
        "every-prev-set-continuity",
        "once-prev-set-ignored",
        "manual-prev-set-ignored",
        "tag-prev-set-ignored",
        "job.success-prev-set-ignored",
    ],
)
def test_compute_run_interval(
    trigger: str,
    prev: Optional[str],
    now: str,
    expected_start: str,
    expected_end: str,
) -> None:
    iv = compute_run_interval(
        TTrigger(trigger),
        ensure_pendulum_datetime_utc(now),
        prev_interval_end=ensure_pendulum_datetime_utc(prev) if prev else None,
    )
    assert iv == (
        ensure_pendulum_datetime_utc(expected_start),
        ensure_pendulum_datetime_utc(expected_end),
    )


def test_compute_run_interval_returns_utc() -> None:
    """All returned datetimes are UTC."""
    iv = compute_run_interval(
        TTrigger("schedule:0 8 * * *"),
        ensure_pendulum_datetime_utc("2024-06-15T11:00:00Z"),
        prev_interval_end=None,
        tz="US/Eastern",
    )
    assert iv[0].tzname() == "UTC"
    assert iv[1].tzname() == "UTC"


def test_compute_run_interval_schedule_with_timezone() -> None:
    """Schedule interval is the most recently ELAPSED cron period in target tz.

    08:00 ET = 12:00 UTC (EDT). At now = 2024-06-15T11:00Z (07:00 ET on 06-15),
    the last elapsed 08:00-ET tick is 2024-06-14T08:00 ET = 06-14T12:00Z;
    the tick before that is 2024-06-13T08:00 ET = 06-13T12:00Z.
    """
    iv = compute_run_interval(
        TTrigger("schedule:0 8 * * *"),
        ensure_pendulum_datetime_utc("2024-06-15T11:00:00Z"),
        prev_interval_end=None,
        tz="US/Eastern",
    )
    assert iv == (
        ensure_pendulum_datetime_utc("2024-06-13T12:00:00Z"),
        ensure_pendulum_datetime_utc("2024-06-14T12:00:00Z"),
    )


def test_compute_run_interval_invalid_trigger() -> None:
    """Unparseable triggers raise InvalidTrigger."""
    with pytest.raises(InvalidTrigger):
        compute_run_interval(
            TTrigger("not-a-trigger"),
            ensure_pendulum_datetime_utc("2024-06-15T12:00:00Z"),
            prev_interval_end=None,
        )


@pytest.mark.parametrize(
    "cron,dt,count,expected",
    [
        # daily cron, off-tick value
        ("0 0 * * *", "2024-01-15T14:00:00Z", 0, "2024-01-15T00:00:00Z"),
        ("0 0 * * *", "2024-01-15T14:00:00Z", 1, "2024-01-14T00:00:00Z"),
        ("0 0 * * *", "2024-01-15T14:00:00Z", 3, "2024-01-12T00:00:00Z"),
        ("0 0 * * *", "2024-01-15T14:00:00Z", -1, "2024-01-16T00:00:00Z"),
        ("0 0 * * *", "2024-01-15T14:00:00Z", -2, "2024-01-17T00:00:00Z"),
        # daily cron, exactly on a tick
        ("0 0 * * *", "2024-01-15T00:00:00Z", 0, "2024-01-15T00:00:00Z"),
        ("0 0 * * *", "2024-01-15T00:00:00Z", 1, "2024-01-14T00:00:00Z"),
        ("0 0 * * *", "2024-01-15T00:00:00Z", -1, "2024-01-16T00:00:00Z"),
        # hourly cron
        ("0 * * * *", "2024-01-15T14:30:00Z", 2, "2024-01-15T12:00:00Z"),
        ("0 * * * *", "2024-01-15T14:30:00Z", -1, "2024-01-15T15:00:00Z"),
    ],
    ids=[
        "daily-floor",
        "daily-back-1",
        "daily-back-3",
        "daily-forward-1",
        "daily-forward-2",
        "on-tick-floor",
        "on-tick-back-1",
        "on-tick-forward-1",
        "hourly-back-2",
        "hourly-forward-1",
    ],
)
def test_cron_lag(cron: str, dt: str, count: int, expected: str) -> None:
    assert cron_lag(cron, ensure_pendulum_datetime_utc(dt), count) == ensure_pendulum_datetime_utc(
        expected
    )


def test_cron_lag_preserves_timezone() -> None:
    """Ticks are computed on the wall clock of the input timezone."""
    berlin = ZoneInfo("Europe/Berlin")
    dt = datetime(2024, 1, 15, 14, 0, tzinfo=berlin)
    lagged = cron_lag("0 0 * * *", dt, 1)
    assert lagged == datetime(2024, 1, 14, 0, 0, tzinfo=berlin)
    assert lagged.tzinfo is berlin


def _iv(start: str, end: str) -> TTimeInterval:
    return TTimeInterval(ensure_pendulum_datetime_utc(start), ensure_pendulum_datetime_utc(end))


def test_lag_interval() -> None:
    iv = _iv("2024-01-13T07:00:00Z", "2024-01-15T14:00:00Z")

    # bare cron lags the start, end untouched
    assert lag_interval(iv, "0 0 * * *", 3) == _iv("2024-01-10T00:00:00Z", "2024-01-15T14:00:00Z")
    # schedule: trigger form, count 0 floors the start
    assert lag_interval(iv, "schedule:0 0 * * *", 0) == _iv(
        "2024-01-13T00:00:00Z", "2024-01-15T14:00:00Z"
    )
    # lag_end drops the incomplete trailing day
    assert lag_interval(iv, "0 0 * * *", 0, lag_end=True) == _iv(
        "2024-01-13T07:00:00Z", "2024-01-15T00:00:00Z"
    )
    # negative count moves the bound into the future
    assert lag_interval(iv, "0 0 * * *", -1, lag_end=True) == _iv(
        "2024-01-13T07:00:00Z", "2024-01-16T00:00:00Z"
    )
    # every: trigger shifts by period, count 0 is a no-op
    assert lag_interval(iv, "every:1h", 2) == _iv("2024-01-13T05:00:00Z", "2024-01-15T14:00:00Z")
    assert lag_interval(iv, "every:1h", 0) == iv

    # lagging the end below the start raises
    with pytest.raises(ValueError, match="empty or negative"):
        lag_interval(iv, "0 0 * * *", 3, lag_end=True)
    # period-less triggers raise
    with pytest.raises(InvalidTrigger):
        lag_interval(iv, "http:", 1)


def test_full_days_interval() -> None:
    # mid-day bounds widened to cover both days fully
    assert full_days_interval(_iv("2024-01-13T07:00:00Z", "2024-01-15T14:00:00Z")) == _iv(
        "2024-01-13T00:00:00Z", "2024-01-16T00:00:00Z"
    )
    # exact-midnight end still extends by one day
    assert full_days_interval(_iv("2024-01-15T00:00:00Z", "2024-01-16T00:00:00Z")) == _iv(
        "2024-01-15T00:00:00Z", "2024-01-17T00:00:00Z"
    )
    # bounds floor on the wall clock of their own timezone
    berlin = ZoneInfo("Europe/Berlin")
    widened = full_days_interval(
        TTimeInterval(
            datetime(2024, 1, 15, 7, 0, tzinfo=berlin), datetime(2024, 1, 15, 14, 0, tzinfo=berlin)
        )
    )
    assert widened == TTimeInterval(
        datetime(2024, 1, 15, 0, 0, tzinfo=berlin), datetime(2024, 1, 16, 0, 0, tzinfo=berlin)
    )


def test_interval_accessor_apply_lag_and_full_days() -> None:
    """`dlt.current.interval` mutators chain and replicate the manual lag-and-widen pattern."""
    iv = _iv("2024-01-13T07:00:00Z", "2024-01-15T14:00:00Z")
    with Container().injectable_context(TimeIntervalContext(interval=iv)):
        assert not interval_accessor.is_empty
        # mutators return the accessor so calls chain; result reads via the call form
        assert interval_accessor.apply_full_days().apply_lag("0 0 * * *", 3)() == _iv(
            "2024-01-10T00:00:00Z", "2024-01-16T00:00:00Z"
        )
        # mutators set the active interval
        assert interval_accessor() == _iv("2024-01-10T00:00:00Z", "2024-01-16T00:00:00Z")

    # no active interval raises
    with Container().injectable_context(TimeIntervalContext()):
        with pytest.raises(RuntimeError, match="no active interval"):
            interval_accessor.apply_lag("0 0 * * *")
        with pytest.raises(RuntimeError, match="no active interval"):
            interval_accessor.apply_full_days()
