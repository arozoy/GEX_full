"""NYSE trading calendar: list trading days and compute snapshot times.

Uses the `exchange_calendars` library so we do not depend on a live holidays API.
The snapshot is defined as `MINUTES_BEFORE_CLOSE` before the session close,
which automatically adjusts for early-close days (e.g., day after Thanksgiving).
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import exchange_calendars as xcals

from config import MARKET_TZ, MINUTES_BEFORE_CLOSE

_NYSE = xcals.get_calendar("XNYS")

# Polygon flat files for trading day D are published by ~11 AM ET on the
# next calendar day. We use this cutoff to cap END_DATE automatically.
_FLATFILE_PUBLISH_HOUR_ET = 11


def list_trading_days(start: date, end: date) -> list[date]:
    """Return every NYSE trading day in [start, end] inclusive."""
    sessions = _NYSE.sessions_in_range(str(start), str(end))
    return [s.date() for s in sessions]


def market_close_et(trading_day: date) -> datetime:
    """Session close in `America/New_York`. Honors early-close days."""
    close_utc = _NYSE.session_close(str(trading_day))
    return close_utc.to_pydatetime().astimezone(MARKET_TZ)


def snapshot_time_et(trading_day: date) -> datetime:
    """`MINUTES_BEFORE_CLOSE` before session close (ET)."""
    return market_close_et(trading_day) - timedelta(minutes=MINUTES_BEFORE_CLOSE)


def option_expiration_datetime_et(trading_day: date) -> datetime:
    """Moment an equity option expires (4:00 PM ET on the expiration date).

    For 0DTE contracts we price at the snapshot time; this function gives the
    terminal moment used for the time-to-expiry calculation.
    """
    return market_close_et(trading_day)


def time_to_expiry_years(now_et: datetime, expiry_et: datetime) -> float:
    """Year-fraction between two timezone-aware ET datetimes. 365-day basis."""
    delta = expiry_et - now_et
    return max(delta.total_seconds() / (365.0 * 24 * 3600), 0.0)


def latest_published_flatfile_day() -> date:
    """Most recent NYSE trading day whose flat file has been published.

    Flat files for trading day D are published by ~11 AM ET on the day
    following D. Before that cutoff, only D-2's file and earlier are safe
    to assume published.
    """
    now_et = datetime.now(MARKET_TZ)
    # Start from yesterday; if it's before the 11 AM ET cutoff, step one
    # more day back (yesterday's file isn't guaranteed published yet).
    cutoff = now_et.date() - timedelta(days=1)
    if now_et.time() < time(_FLATFILE_PUBLISH_HOUR_ET, 0):
        cutoff -= timedelta(days=1)
    # Walk backwards until we land on a valid NYSE session.
    while not _NYSE.is_session(cutoff.isoformat()):
        cutoff -= timedelta(days=1)
    return cutoff
