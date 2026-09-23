"""The Summary tab: key statistics and price-window slicing.

Pure — no Streamlit, no network, no global state — so the statistics can be
unit tested against fixture frames and reused by a future screener. The fetch
that feeds them lives in :mod:`app.logic.research.dataset`.

Two conventions from the DCF engine carry over unchanged:

* **No silent fallbacks.** A figure that could not be obtained becomes
  ``Sourced.unavailable`` with a reason, never zero and never blank.
* **Every displayed number carries an origin**, and where it was computed, the
  formula that produced it.

The one place those two rules genuinely collide is the dividend. Sharadar fills
``dps`` and ``divyield`` with ``0`` both for companies that pay nothing and for
figures absent from a filing (see ``vocabulary.ZERO_FILLED_INDICATORS``),
so a bare ``0.00`` is ambiguous. :func:`_dividend_tile` resolves it by
cross-checking the ACTIONS table, which records an actual dividend payment or
records nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.data.sharadar import vocabulary
from app.logic.dcf import wacc as wacc_lib
from app.logic.research.formatting import (
    format_count,
    format_money,
    format_pct,
    format_range,
    format_ratio,
    is_missing,
)
from app.provenance import Origin, Sourced

#: Price windows offered above the chart, in display order. ``5D`` counts
#: trading sessions because that is what it means on a quote page; everything
#: else is a calendar offset, so ``1M`` spans a month of the market's time
#: rather than 21 sessions of it.
PERIODS: tuple[str, ...] = ("5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "ALL")

#: The key-statistics table, column by column — four columns of four rows, each
#: column a theme: what the stock did, what the company is worth, what you pay
#: per unit of it, and everything with a date on it.
#:
#: Labels here must match the ones :func:`build_key_stats` produces exactly;
#: ``test_stock_research`` asserts the two agree, so a renamed tile fails a test
#: rather than silently vanishing from the page.
STAT_COLUMNS: tuple[tuple[str, ...], ...] = (
    ("Day's Range", "52 Week Range", "Volume", "Avg Volume (3M)"),
    ("Market Cap", "Enterprise Value", "Beta (5Y monthly vs SPY)", "Shares Outstanding"),
    ("P/E", "P/B", "P/S", "EPS (TTM, diluted)"),
    ("EV/EBITDA", "Dividend & Yield (TTM)", "Ex-Dividend Date", "Next Report (est.)"),
)

#: Shown above the chart as the headline metric rather than in the table, but
#: still listed in the sources panel because it is displayed on the page.
HEADLINE_STAT = "Previous Close"

#: Sessions averaged for the "Avg Volume (3M)" tile. A quarter is ~63 trading
#: days; using sessions rather than a date window keeps the average stable
#: across holiday-heavy quarters.
_AVG_VOLUME_SESSIONS = 63

#: Quarters of filing history used to estimate the next reporting date.
_FILING_LAG_QUARTERS = 8
#: Benchmark for the beta regression. Sharadar carries no index levels, so the
#: S&P 500 stand-in is the ETF — same convention as the DCF page.
BENCHMARK = "SPY"

BETA_YEARS = 5


# ---------------------------------------------------------------------------
# Display carrier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatTile:
    """One key-statistic cell: a formatted value plus its provenance.

    Distinct from :class:`~app.provenance.Sourced` because ``Sourced.value`` is
    a single ``float``, and three of the tiles on this page are not one: the
    two ranges are a pair of numbers and the reporting dates are dates. Rather
    than widen a type the whole DCF engine depends on, numbers stay in
    ``Sourced`` and :meth:`from_sourced` projects them here for display;
    ranges and dates build a tile directly.

    ``display`` is ``None`` exactly when the figure is unavailable — the view
    renders "not reported" rather than an empty cell.
    """

    label: str
    display: str | None
    origin: Origin
    source: str
    formula: str | None = None
    as_of: date | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.display is not None

    def describe(self, include_origin: bool = False) -> str:
        """One-line provenance string, mirroring ``Sourced.describe``."""
        bits = [self.origin.label] if include_origin else []
        bits.append(self.source)
        if self.formula:
            bits.append(self.formula)
        if self.as_of:
            bits.append(f"as of {self.as_of.isoformat()}")
        if self.detail:
            bits.append(self.detail)
        return " · ".join(b for b in bits if b)

    @classmethod
    def from_sourced(cls, label: str, sourced: Sourced, formatter) -> "StatTile":
        """Projects a ``Sourced`` number into a tile using ``formatter``."""
        return cls(
            label=label,
            display=formatter(sourced.value) if sourced.ok else None,
            origin=sourced.origin,
            source=sourced.source,
            formula=sourced.formula,
            as_of=sourced.as_of,
            detail=sourced.detail,
        )

    @classmethod
    def unavailable(cls, label: str, source: str, reason: str) -> "StatTile":
        return cls(
            label=label,
            display=None,
            origin=Origin.UNAVAILABLE,
            source=source,
            detail=reason,
        )


# ---------------------------------------------------------------------------
# Price windows
# ---------------------------------------------------------------------------


def slice_period(frame: pd.DataFrame, period: str, today: date | None = None) -> pd.DataFrame:
    """Trims a date-indexed price frame to one of :data:`PERIODS`.

    The full history is fetched once when the user searches; every window is a
    slice of that frame, so switching periods never touches the network.

    Anchored on the last bar rather than on ``today`` for the calendar windows,
    so a stale or delisted series still shows a full window instead of an empty
    one. ``YTD`` is the exception — it means this calendar year, and anchoring
    it to the data would silently redefine it.
    """
    if frame.empty or period == "ALL":
        return frame
    if period == "5D":
        return frame.tail(5)

    index = pd.DatetimeIndex(frame.index)
    last = index.max()
    reference = pd.Timestamp(today) if today is not None else last

    offsets = {
        "1M": pd.DateOffset(months=1),
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1),
        "5Y": pd.DateOffset(years=5),
    }
    if period == "YTD":
        start = pd.Timestamp(year=reference.year, month=1, day=1)
    elif period in offsets:
        start = last - offsets[period]
    else:
        return frame

    return frame.loc[index >= start]


def period_change(frame: pd.DataFrame) -> Sourced:
    """Percentage change in close across whatever window ``frame`` covers."""
    if frame.empty or "Close" not in frame.columns:
        return Sourced.unavailable("Sharadar SEP", "No price history in this window.")
    closes = frame["Close"].dropna()
    if len(closes) < 2:
        return Sourced.unavailable(
            "Sharadar SEP", "Fewer than two sessions in this window."
        )
    first, last = float(closes.iloc[0]), float(closes.iloc[-1])
    if first == 0:
        return Sourced.unavailable("Sharadar SEP", "Opening close of the window is zero.")
    return Sourced.derived(
        value=(last - first) / first,
        source="Sharadar SEP",
        formula=f"({last:,.2f} − {first:,.2f}) ÷ {first:,.2f}",
        as_of=pd.Timestamp(frame.index.max()).date(),
    )


# ---------------------------------------------------------------------------
# Individual statistics
# ---------------------------------------------------------------------------


def _last_row(frame: pd.DataFrame) -> pd.Series | None:
    return None if frame.empty else frame.iloc[-1]


def _statement_value(frame: pd.DataFrame, indicator: str) -> tuple[float | None, date | None]:
    """Newest reported value of an SF1 indicator, with its report date.

    Indexed by indicator code rather than by Sharadar's published caption, so a
    caption change upstream cannot silently return the wrong line.
    """
    caption = vocabulary.SF1_CAPTIONS.get(indicator)
    if not caption or frame.empty or caption not in frame.columns:
        return None, None
    row = frame.iloc[-1]
    value = row.get(caption)
    as_of = row.get("Report Date")
    return (
        None if is_missing(value) else float(value),
        None if is_missing(as_of) else pd.Timestamp(as_of).date(),
    )


def days_range(ohlcv: pd.DataFrame, currency: str) -> StatTile:
    row = _last_row(ohlcv)
    if row is None or is_missing(row.get("Low")) or is_missing(row.get("High")):
        return StatTile.unavailable(
            "Day's Range", "Sharadar SEP", "No high/low on the latest session."
        )
    as_of = pd.Timestamp(ohlcv.index.max()).date()
    return StatTile(
        label="Day's Range",
        display=format_range(row["Low"], row["High"], currency),
        origin=Origin.REPORTED,
        source="Sharadar SEP",
        as_of=as_of,
    )


def fifty_two_week_range(ohlcv: pd.DataFrame, currency: str) -> StatTile:
    if ohlcv.empty:
        return StatTile.unavailable(
            "52 Week Range", "Sharadar SEP", "No price history."
        )
    index = pd.DatetimeIndex(ohlcv.index)
    window = ohlcv.loc[index >= index.max() - pd.DateOffset(years=1)]
    lows, highs = window.get("Low"), window.get("High")
    if lows is None or highs is None or lows.dropna().empty or highs.dropna().empty:
        return StatTile.unavailable(
            "52 Week Range", "Sharadar SEP", "No high/low in the trailing year."
        )
    return StatTile(
        label="52 Week Range",
        display=format_range(float(lows.min()), float(highs.max()), currency),
        origin=Origin.DERIVED,
        source="Sharadar SEP",
        formula=f"min low / max high over {len(window)} sessions",
        as_of=pd.Timestamp(index.max()).date(),
    )


def _dividend_tile(
    ttm_pl: pd.DataFrame,
    actions: pd.DataFrame,
    last_close: float | None,
    currency: str,
    today: date,
) -> StatTile:
    """TTM dividend with its yield in brackets, zero-fill ambiguity resolved.

    ``dps`` is one of the indicators Sharadar zero-fills when an item is absent
    from a filing, so ``0.00`` means either "pays nothing" or "not disclosed".
    ACTIONS settles it: a company that paid a dividend has a row for it. Where
    the two disagree the figure is reported unavailable rather than shown as
    zero — that disagreement is exactly the case the no-silent-fallbacks rule
    exists for.

    The yield is *derived* rather than read from SF1's ``divyield``, which
    Sharadar rounds to three decimals — 0.4% for a figure that is really 0.34%.
    Dividing the dividend already on show by the price already on show also
    keeps the bracket consistent with the number in front of it.
    """
    label = "Dividend & Yield (TTM)"
    dps, dps_as_of = _statement_value(ttm_pl, "dps")
    recent = _recent_dividends(actions, today)

    if dps is None:
        return StatTile.unavailable(
            label,
            "Sharadar SF1 (MRT)",
            "Dividends per share not reported for the TTM period.",
        )

    if dps == 0 and not recent.empty:
        latest = pd.Timestamp(recent["date"].max()).date()
        return StatTile.unavailable(
            label,
            "Sharadar SF1 (MRT) · dps",
            f"SF1 reports 0 but ACTIONS shows {len(recent)} dividend(s) since "
            f"{pd.Timestamp(recent['date'].min()).date().isoformat()} "
            f"(latest {latest.isoformat()}). Sharadar writes 0 where an item is "
            f"absent from a filing, so this zero cannot be trusted.",
        )

    detail = (
        "No dividends in the trailing 12 months (confirmed against ACTIONS)."
        if dps == 0
        else ""
    )

    if not last_close:
        return StatTile(
            label=label,
            display=format_money(dps, currency),
            origin=Origin.REPORTED,
            source="Sharadar SF1 (MRT) · dps",
            as_of=dps_as_of,
            detail=" ".join(filter(None, [detail, "No closing price, so no yield."])),
        )

    return StatTile(
        label=label,
        display=f"{format_money(dps, currency)} ({format_pct(dps / last_close)})",
        # Reported dividend, derived yield: the weaker of the two origins wins,
        # because the cell is only as trustworthy as its least direct part.
        origin=Origin.DERIVED,
        source="Sharadar SF1 (MRT) · dps, Sharadar SEP",
        formula=f"yield = {dps:,.4f} ÷ close {last_close:,.2f}",
        as_of=dps_as_of,
        detail=detail,
    )


def _recent_dividends(actions: pd.DataFrame, today: date, days: int = 366) -> pd.DataFrame:
    """Dividend rows with an ex-date inside the trailing window."""
    if actions.empty or "action" not in actions.columns:
        return pd.DataFrame(columns=["date", "value"])
    dividends = actions[actions["action"] == vocabulary.DIVIDEND_ACTION]
    if dividends.empty:
        return pd.DataFrame(columns=["date", "value"])
    cutoff = pd.Timestamp(today) - pd.Timedelta(days=days)
    return dividends[
        (dividends["date"] >= cutoff) & (dividends["date"] <= pd.Timestamp(today))
    ]


def _ex_dividend_tile(actions: pd.DataFrame, currency: str, today: date) -> StatTile:
    label = "Ex-Dividend Date"
    if actions.empty or "action" not in actions.columns:
        return StatTile.unavailable(
            label, "Sharadar ACTIONS", "No corporate actions on record."
        )
    dividends = actions[actions["action"] == vocabulary.DIVIDEND_ACTION]
    if dividends.empty:
        return StatTile.unavailable(
            label, "Sharadar ACTIONS", "This company has never paid a dividend."
        )

    row = dividends.iloc[-1]
    ex_date = pd.Timestamp(row["date"]).date()
    amount = row.get("value")
    detail = "" if is_missing(amount) else f"{format_money(amount, currency)} per share"
    # Sharadar carries declared ex-dates ahead of time, so the newest row is
    # sometimes in the future. Say which it is rather than implying it passed.
    if ex_date > today:
        detail = " · ".join(filter(None, ["upcoming", detail]))
    return StatTile(
        label=label,
        display=ex_date.isoformat(),
        origin=Origin.REPORTED,
        source="Sharadar ACTIONS",
        as_of=ex_date,
        detail=detail,
    )


def _next_report_tile(quarterly_ar: pd.DataFrame, today: date) -> StatTile:
    """An estimate of the next filing date, from the company's own cadence.

    The filing date is only real under the ``AR`` dimensions: under ``MR``,
    ``datekey`` equals the report period, so the lag is uniformly zero and any
    estimate built on it would be wrong by a month. Confirmed against live
    responses — AAPL files a steady 34 days after period end, KO 26–51.

    Sharadar publishes nothing forward-looking, so this is an estimate and is
    labelled as one. It is not an announced earnings date and must not be shown
    as one. The last real filing date rides along in the detail, since it is
    what the estimate is built from.
    """
    label = "Next Report (est.)"
    source = "Sharadar SF1 (ARQ)"
    if quarterly_ar.empty or "Publish Date" not in quarterly_ar.columns:
        return StatTile.unavailable(label, source, "No quarterly filings on record.")

    recent = quarterly_ar.dropna(subset=["Publish Date", "Report Date"]).tail(
        _FILING_LAG_QUARTERS
    )
    if recent.empty:
        return StatTile.unavailable(label, source, "No dated quarterly filings.")

    last = recent.iloc[-1]
    published = pd.Timestamp(last["Publish Date"]).date()
    period_end = pd.Timestamp(last["Report Date"]).date()
    fiscal = str(last.get("Fiscal Period") or "").strip()
    year = last.get("Fiscal Year")
    tag = f"{'' if pd.isna(year) else int(year)} {fiscal}".strip()
    filed = (
        f"Last filed {published.isoformat()} "
        f"({tag} · period ended {period_end.isoformat()})."
    )

    lags = (recent["Publish Date"] - recent["Report Date"]).dt.days.dropna()
    if lags.empty:
        return StatTile.unavailable(
            label, source, f"{filed} Filing lag could not be measured."
        )

    median_lag = int(round(float(lags.median())))
    next_period_end = (pd.Timestamp(period_end) + pd.DateOffset(months=3)).date()
    estimate = (pd.Timestamp(next_period_end) + pd.Timedelta(days=median_lag)).date()

    detail = f"{filed} Projected from filing cadence — Sharadar publishes no earnings calendar."
    if estimate < today:
        # The projected date has already passed, which means the filing is late
        # or Sharadar has not ingested it yet. Rolling the estimate forward
        # would quietly skip a quarter that should already have been reported,
        # so the stale date stands and says why.
        detail = (
            f"This date has passed — the filing for the period ending "
            f"{next_period_end.isoformat()} is not yet in Sharadar. " + detail
        )

    return StatTile(
        label=label,
        display=estimate.isoformat(),
        origin=Origin.ESTIMATE,
        source=source,
        formula=(
            f"next period end {next_period_end.isoformat()} "
            f"+ median filing lag {median_lag}d over {len(lags)} quarters"
        ),
        detail=detail,
    )


def _beta_tile(
    stock_closes: pd.Series, index_closes: pd.Series, benchmark: str = BENCHMARK
) -> StatTile:
    label = f"Beta (5Y monthly vs {benchmark})"
    source = f"Sharadar SEP + SFP ({benchmark})"
    if stock_closes.empty or index_closes.empty:
        return StatTile.unavailable(
            label, source, "No overlapping price history to regress."
        )
    try:
        result = wacc_lib.regress_beta(
            stock_closes, index_closes, frequency="monthly", index_ticker=benchmark
        )
    except wacc_lib.InsufficientObservationsError as error:
        return StatTile.unavailable(label, source, str(error))

    # `result.end` is the resampled month-end label, which lands in the future
    # for any month still in progress. The last actual price is the honest date.
    return StatTile(
        label=label,
        display=format_ratio(result.beta),
        origin=Origin.DERIVED,
        source=source,
        formula=result.formula,
        as_of=pd.Timestamp(stock_closes.index.max()).date(),
        detail=f"R² {result.r_squared:.2f}",
    )


# ---------------------------------------------------------------------------
# The tile grid
# ---------------------------------------------------------------------------


def build_key_stats(
    *,
    ohlcv: pd.DataFrame,
    daily: pd.DataFrame,
    ttm_pl: pd.DataFrame,
    quarterly_ar: pd.DataFrame,
    actions: pd.DataFrame,
    stock_closes: pd.Series,
    index_closes: pd.Series,
    currency: str = "",
    today: date | None = None,
) -> list[StatTile]:
    """The whole key-statistics grid, in display order.

    Every argument is a frame that has already been fetched; this does no I/O.
    A tile whose inputs are missing reports itself unavailable — none of them
    fall back to a neighbouring source or to zero.
    """
    today = today or date.today()
    price_row = _last_row(ohlcv)
    daily_row = _last_row(daily)
    price_date = None if ohlcv.empty else pd.Timestamp(ohlcv.index.max()).date()

    close = None if price_row is None or is_missing(price_row.get("Close")) else float(
        price_row["Close"]
    )

    def from_daily(label: str, column: str, formatter, note: str = "") -> StatTile:
        if daily_row is None or is_missing(daily_row.get(column)):
            return StatTile.unavailable(
                label, "Sharadar DAILY", f"{column} not reported for the latest session."
            )
        return StatTile(
            label=label,
            display=formatter(daily_row[column]),
            origin=Origin.REPORTED,
            source=f"Sharadar DAILY · {column}",
            as_of=pd.Timestamp(daily_row["date"]).date()
            if "date" in daily_row and not is_missing(daily_row["date"])
            else price_date,
            detail=note,
        )

    tiles: list[StatTile] = []

    # --- Price -------------------------------------------------------------
    tiles.append(
        StatTile.from_sourced(
            "Previous Close",
            Sourced.reported(close, "Sharadar SEP · close", as_of=price_date)
            if close is not None
            else Sourced.unavailable("Sharadar SEP", "No closing price on record."),
            lambda v: format_money(v, currency),
        )
    )
    tiles.append(days_range(ohlcv, currency))
    tiles.append(fifty_two_week_range(ohlcv, currency))

    # --- Volume ------------------------------------------------------------
    tiles.append(
        StatTile.from_sourced(
            "Volume",
            Sourced.reported(
                float(price_row["Volume"]), "Sharadar SEP · volume", as_of=price_date
            )
            if price_row is not None and not is_missing(price_row.get("Volume"))
            else Sourced.unavailable("Sharadar SEP", "No volume on the latest session."),
            format_count,
        )
    )
    volumes = (
        pd.Series(dtype="float64")
        if ohlcv.empty or "Volume" not in ohlcv.columns
        else ohlcv["Volume"].dropna().tail(_AVG_VOLUME_SESSIONS)
    )
    tiles.append(
        StatTile.from_sourced(
            "Avg Volume (3M)",
            Sourced.derived(
                value=float(volumes.mean()),
                source="Sharadar SEP",
                formula=f"mean volume over {len(volumes)} sessions",
                as_of=price_date,
            )
            if not volumes.empty
            else Sourced.unavailable("Sharadar SEP", "No volume history."),
            format_count,
        )
    )

    # --- Size and multiples ------------------------------------------------
    # DAILY publishes market cap and EV in millions; the client has already
    # rescaled them to raw units, so these are formatted like any other money.
    tiles.append(from_daily("Market Cap", "marketcap", lambda v: format_money(v, currency)))
    tiles.append(from_daily("Enterprise Value", "ev", lambda v: format_money(v, currency)))
    tiles.append(from_daily("P/E", "pe", format_ratio, note="Trailing twelve months."))
    tiles.append(from_daily("P/B", "pb", format_ratio))
    tiles.append(from_daily("P/S", "ps", format_ratio))
    tiles.append(from_daily("EV/EBITDA", "evebitda", format_ratio))

    # --- Earnings ----------------------------------------------------------
    eps, eps_as_of = _statement_value(ttm_pl, "epsdil")
    tiles.append(
        StatTile.from_sourced(
            "EPS (TTM, diluted)",
            Sourced.reported(eps, "Sharadar SF1 (MRT) · epsdil", as_of=eps_as_of)
            if eps is not None
            else Sourced.unavailable(
                "Sharadar SF1 (MRT)", "Diluted EPS not reported for the TTM period."
            ),
            lambda v: format_money(v, currency),
        )
    )

    # --- Dividends ---------------------------------------------------------
    tiles.append(_dividend_tile(ttm_pl, actions, close, currency, today))
    tiles.append(_ex_dividend_tile(actions, currency, today))

    # --- Risk --------------------------------------------------------------
    tiles.append(_beta_tile(stock_closes, index_closes))

    # --- Shares ------------------------------------------------------------
    if daily_row is not None and not is_missing(daily_row.get("marketcap")) and close:
        tiles.append(
            StatTile.from_sourced(
                "Shares Outstanding",
                Sourced.derived(
                    value=float(daily_row["marketcap"]) / close,
                    source="Sharadar DAILY + SEP",
                    formula=f"market cap ÷ close {close:,.2f}",
                    as_of=price_date,
                ),
                format_count,
            )
        )
    else:
        tiles.append(
            StatTile.unavailable(
                "Shares Outstanding",
                "Sharadar DAILY + SEP",
                "Needs both a market cap and a closing price.",
            )
        )

    # --- Reporting calendar ------------------------------------------------
    tiles.append(_next_report_tile(quarterly_ar, today))
    return tiles


def stats_by_label(stats: list[StatTile]) -> dict[str, StatTile]:
    """Tiles keyed by label, for laying the table out by :data:`STAT_COLUMNS`."""
    return {tile.label: tile for tile in stats}

