"""🚀 Momentum Analysis over HTTP: a price window with its indicators, and a divergence read.

Two routes because they change at different rates — the window when the ticker
or period moves, the divergence on every anchor change. Both rebuild the frame
from the same cached fetch, so the second costs no request to Sharadar.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query, status

from app.api.contract import Contract, finite_list
from app.data import sharadar_client
from app.logic import momentum_analysis as ma

router = APIRouter(prefix="/momentum", tags=["momentum"])

#: The lookbacks, an enum in the schema, built from the logic layer's table so
#: the two cannot drift.
Period = Literal[tuple(ma.PERIOD_DAYS)]  # type: ignore[valid-type]
AnchorKind = Literal["Lows", "Highs"]
Ticker = Annotated[str, Path(pattern=r"^[A-Za-z0-9.\-]{1,10}$")]

#: How far back the first default anchor sits, in sessions.
DEFAULT_ANCHOR_SESSIONS = 20


class Bars(Contract):
    """One entry per session, column by column — the shape a chart trace takes.

    Plain ``float`` throughout: :func:`app.logic.momentum_analysis.indicator_frame`
    drops every session whose indicators are still warming up.
    """

    sessions: list[date]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    macd: list[float]
    signal: list[float]
    histogram: list[float]
    rsi: list[float]


class Scanners(Contract):
    macd_momentum: str
    macd_trend: str
    rsi_trend: str
    rsi_failure_swings: str


class AnchorDates(Contract):
    first: date
    second: date


class MomentumWindow(Contract):
    ticker: str
    period: Period
    window_start: date
    bars: Bars
    scanners: Scanners
    #: The session ``DEFAULT_ANCHOR_SESSIONS`` back and the last one.
    default_anchors: AnchorDates


class Anchor(Contract):
    session: date
    close: float
    macd: float
    rsi: float


class Reading(Contract):
    state: str
    bias: str


class Divergence(Contract):
    first: Anchor
    second: Anchor
    macd: Reading
    rsi: Reading


def _frame(ticker: str, period: str) -> pd.DataFrame:
    end = date.today()
    window_start = end - timedelta(days=ma.PERIOD_DAYS[period])
    fetch_start = window_start - timedelta(days=ma.INDICATOR_WARMUP_DAYS)
    ohlcv = sharadar_client.fetch_ohlcv(ticker, fetch_start.isoformat(), end.isoformat())
    if ohlcv.empty:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No data found for this ticker.")
    frame = ma.indicator_frame(ohlcv, window_start)
    if frame.empty:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Not enough price history for this ticker to compute indicators.",
        )
    return frame


def _anchor(point: ma.AnchorPoint) -> Anchor:
    return Anchor(
        session=point.session.date(), close=point.close, macd=point.macd, rsi=point.rsi
    )


def _sessions(index: pd.Index) -> list[date]:
    return [stamp.date() for stamp in index]


@router.get("/{ticker}")
def momentum_window(
    ticker: Ticker, period: Annotated[Period, Query()] = "1Y"
) -> MomentumWindow:
    """The window's bars with RSI and MACD, and the four scanner readings."""
    ticker = ticker.upper()
    frame = _frame(ticker, period)
    macd_momentum, macd_trend = ma.detect_macd_crossovers(frame["MACD"], frame["MACD_Hist"])
    sessions = _sessions(frame.index)
    return MomentumWindow(
        ticker=ticker,
        period=period,
        window_start=date.today() - timedelta(days=ma.PERIOD_DAYS[period]),
        bars=Bars(
            sessions=sessions,
            open=finite_list(frame["Open"]),
            high=finite_list(frame["High"]),
            low=finite_list(frame["Low"]),
            close=finite_list(frame["Close"]),
            macd=finite_list(frame["MACD"]),
            signal=finite_list(frame["Signal"]),
            histogram=finite_list(frame["MACD_Hist"]),
            rsi=finite_list(frame["RSI"]),
        ),
        scanners=Scanners(
            macd_momentum=macd_momentum,
            macd_trend=macd_trend,
            rsi_trend=ma.detect_rsi_trend(frame["RSI"]),
            rsi_failure_swings=ma.detect_failure_swings(frame["RSI"]),
        ),
        default_anchors=AnchorDates(
            first=sessions[-min(DEFAULT_ANCHOR_SESSIONS, len(sessions))],
            second=sessions[-1],
        ),
    )


@router.get("/{ticker}/divergence")
def divergence(
    ticker: Ticker,
    first: Annotated[date, Query()],
    second: Annotated[date, Query()],
    anchor: Annotated[AnchorKind, Query()] = "Lows",
    period: Annotated[Period, Query()] = "1Y",
) -> Divergence:
    """MACD and RSI against price between two anchor sessions.

    Each anchor snaps to the last session on or before it, as the chart's own
    hover does; one that precedes the window is refused rather than moved.
    """
    frame = _frame(ticker.upper(), period)
    points = ma.anchor_point(frame, first), ma.anchor_point(frame, second)
    if None in points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Ensure both anchor dates fall within the selected time period.",
        )
    a, b = points
    macd = ma.analyze_divergence(a.close, b.close, a.macd, b.macd, anchor)
    rsi = ma.analyze_divergence(a.close, b.close, a.rsi, b.rsi, anchor)
    return Divergence(
        first=_anchor(a),
        second=_anchor(b),
        macd=Reading(state=macd[0], bias=macd[1]),
        rsi=Reading(state=rsi[0], bias=rsi[1]),
    )
