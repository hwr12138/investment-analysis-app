"""``SEP``/``SFP`` prices, ``DAILY`` valuation ratios, and benchmark series."""

from __future__ import annotations

from functools import partial

import pandas as pd

from app.data import cache, parallel
from app.data.sharadar import reference, transport
from app.data.sharadar.transport import TickerNotFoundError

#: DAILY reports these in millions; SF1 reports the same quantities raw.
_DAILY_MILLIONS = ("marketcap", "ev")

#: Every numeric column DAILY carries. Coerced together so the per-ticker and
#: universe-wide fetches cannot drift apart on which ones are numbers.
_DAILY_NUMERIC = ("ev", "evebit", "evebitda", "marketcap", "pb", "pe", "ps")


_PRICE_NUMERIC = ("open", "high", "low", "close", "volume", "closeadj", "closeunadj")


def _fetch_price_rows(
    ticker: str, *, table: str, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    params: dict = {"ticker": ticker}
    if start:
        params["date.gte"] = start
    if end:
        params["date.lte"] = end

    frame = transport._request_all(table, params, ticker=ticker)
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in _PRICE_NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


@cache.data(ttl=3600)
def _price_rows(ticker: str) -> pd.DataFrame:
    """Every SEP/SFP row for one ticker — the one entry both price shapes slice.

    ``fetch_ohlcv`` (the research page, from 1990) and ``fetch_prices`` (the DCF,
    from the start of its beta window) used to ask for overlapping ranges under
    separate cache keys, so every company searched paid for its price history
    twice. A full history is ~7,100 rows for a company listed since the late
    1990s — one page — so fetching all of it once and slicing costs nothing the
    narrower request used to save.
    """
    return _fetch_price_rows(ticker, table=_price_table_for(ticker))


def _between(rows: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """The rows whose ``date`` falls in ``[start, end]``, either bound optional."""
    if rows.empty or "date" not in rows.columns:
        return rows
    keep = pd.Series(True, index=rows.index)
    if start:
        keep &= rows["date"] >= pd.Timestamp(start)
    if end:
        keep &= rows["date"] <= pd.Timestamp(end)
    return rows[keep].reset_index(drop=True)


def _price_table_for(ticker: str) -> str:
    """Which of SEP/SFP carries this ticker. Defaults to SEP for equities."""
    return reference.fetch_price_tables((ticker.strip().upper(),)).get(
        ticker.strip().upper(), "SEP"
    )


def _normalise_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Types a raw DAILY response and rescales market cap and EV to raw units.

    DAILY publishes ``marketcap`` and ``ev`` in millions while SF1 publishes the
    same quantities raw. Everything downstream — the market-cap cross-check, the
    equity bridge, the valuation tab's peer medians — works in raw units, and a
    missed conversion looks entirely plausible while being wrong by a factor of
    a million.

    Shared by the per-ticker and universe-wide fetchers rather than inlined in
    either. There is now more than one DAILY call site, so the conversion has to
    live somewhere neither of them can forget it.
    """
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame()

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in _DAILY_NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in _DAILY_MILLIONS:
        if column in frame.columns:
            frame[column] = frame[column] * 1_000_000.0
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _fetch_daily_metrics(ticker: str, start: str | None = None) -> pd.DataFrame:
    """One company's daily valuation ratios, oldest first."""
    params: dict = {"ticker": ticker}
    if start:
        params["date.gte"] = start
    return _normalise_daily(transport._request_all("DAILY", params, ticker=ticker))


@cache.data(ttl=60 * 60 * 12)
def fetch_daily_metrics(ticker: str, start: str | None = None) -> pd.DataFrame:
    """Daily valuation ratios: ``ev evebit evebitda marketcap pb pe ps``.

    ``fetch_prices`` merges only ``marketcap``, ``ev`` and ``evebitda`` into the
    price frame because that is all the DCF needs; this exposes the whole thing for
    callers that want the multiples. Both go through the same private fetcher,
    which is the point — the millions-to-raw conversion for ``marketcap`` and
    ``ev`` happens in exactly one place, and a second call site doing its own
    DAILY request would be a factor-of-a-million bug waiting to happen.

    Columns keep Sharadar's own names and the frame is sorted oldest-first.
    """
    return _fetch_daily_metrics(ticker, start)


def _fetch_prices(ticker: str, start: str | None = None) -> pd.DataFrame:
    prices = _between(_price_rows(ticker.strip().upper()), start, None)
    if prices.empty:
        raise TickerNotFoundError(
            f"Sharadar returned no price history for {ticker}.",
            table="SEP",
            ticker=ticker,
        )

    out = pd.DataFrame(
        {
            "Date": prices["date"],
            "Last Closing Price": prices.get("closeunadj"),
            "Adjusted Closing Price": prices.get("closeadj"),
            "Split-Adjusted Close": prices.get("close"),
            "Volume": prices.get("volume"),
        }
    )

    # The cached full series, sliced — the same entry the research page's
    # snapshot fills, so the DCF's DAILY request is a cache hit.
    daily = _between(fetch_daily_metrics(ticker), start, None)
    if not daily.empty:
        merged = out.merge(
            daily[[c for c in ("date", "marketcap", "ev", "evebitda") if c in daily.columns]],
            left_on="Date",
            right_on="date",
            how="left",
        ).drop(columns=["date"])
        merged = merged.rename(
            columns={
                "marketcap": "Market-Cap",
                "ev": "Enterprise Value",
                "evebitda": "EV/EBITDA - ttm",
            }
        )
        # Sharadar's market cap is sharesbas × split-adjusted close × sharefactor,
        # so dividing it back out recovers the share count on the same basis.
        # This is a cross-check against the SF1 count, not a substitute for it.
        if {"Market-Cap", "Split-Adjusted Close"}.issubset(merged.columns):
            merged["Common Shares Outstanding"] = (
                merged["Market-Cap"] / merged["Split-Adjusted Close"]
            )
        out = merged

    return out.sort_values("Date").reset_index(drop=True)


@cache.data(ttl=60 * 60 * 12)
def fetch_prices(ticker: str, start: str | None = None) -> pd.DataFrame:
    """Daily prices and valuation ratios for one company.

    ``Adjusted Closing Price`` is Sharadar's ``closeadj`` — adjusted for splits,
    stock dividends, cash dividends and spinoffs — and is what the beta
    regression uses. ``Last Closing Price`` is ``closeunadj``, the raw traded
    price, and is only correct for the current quote and market cap.
    """
    return _fetch_prices(ticker, start)


def _fetch_price_history(
    tickers: tuple[str, ...], start_date: str, end_date: str
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    wanted = tuple(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    if not wanted:
        return pd.DataFrame()

    # One request per ticker, sent together: Sharadar's answer time varies by
    # several seconds, and in sequence four tickers took 28s against 15s for the
    # slowest alone.
    routes = reference.fetch_price_tables(wanted)
    fetched = parallel.gather({
        ticker: partial(
            _fetch_price_rows,
            ticker,
            table=routes.get(ticker, "SEP"),
            start=start_date,
            end=end_date,
        )
        for ticker in wanted
    })
    columns: dict[str, pd.Series] = {}
    for ticker in wanted:
        rows = fetched[ticker]
        if rows.empty or "closeadj" not in rows.columns:
            continue
        columns[ticker] = rows.set_index("date")["closeadj"].dropna()

    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns).sort_index().dropna(how="all")


@cache.data(ttl=3600)
def fetch_price_history(
    tickers: tuple[str, ...], start_date: str, end_date: str
) -> pd.DataFrame:
    """Adjusted closes for several tickers, one column each.

    Tickers are routed to SEP or SFP automatically, so a mixed list of stocks
    and ETFs works. Uses ``closeadj`` (total return) throughout, so every column
    is on one adjustment convention — which is what makes cross-ticker
    correlations and betas comparable.
    """
    return _fetch_price_history(tickers, start_date, end_date)


def fetch_index_prices(index_ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Adjusted closes for a benchmark, as a Series.

    The benchmark is an ETF from SFP rather than an index, because Sharadar
    carries no index levels. ``SPY`` is the default stand-in for the S&P 500;
    its ``closeadj`` is a total-return series, which matches the convention on
    the company side of the regression.

    A slice of the benchmark's cached full history, like the company's own
    prices: the DCF line on the summary chart regresses a beta at every filing
    date, and a range-keyed fetch would have paid one request per date.
    """
    rows = _between(_price_rows(index_ticker.strip().upper()), start_date, end_date)
    if rows.empty or "closeadj" not in rows.columns:
        return pd.Series(dtype="float64", name=index_ticker)
    return rows.set_index("date")["closeadj"].dropna().rename(index_ticker)


def _fetch_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    rows = _between(_price_rows(ticker), start_date, end_date)
    if rows.empty:
        return pd.DataFrame()

    out = rows.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "closeadj": "Adj Close",
        }
    )
    keep = [c for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume") if c in out]
    return out.set_index("Date")[keep]


@cache.data(ttl=3600)
def fetch_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Full OHLCV for one ticker, indexed by date.

    ``Open``/``High``/``Low``/``Close`` are adjusted for splits and stock
    dividends but not cash dividends, so the four stay mutually consistent and
    a candlestick drawn from them is honest. ``Adj Close`` is the fully adjusted
    series for anything that needs total return.
    """
    return _fetch_ohlcv(ticker, start_date, end_date)
