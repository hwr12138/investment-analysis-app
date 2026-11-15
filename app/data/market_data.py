"""Rates and the equity risk premium.

The two market inputs that do not come from a securities feed: Treasury yields
proxied by the CBOE yield indices on Yahoo Finance, and the implied equity risk
premium scraped from Damodaran. Prices and fundamentals live in the sibling
``sharadar_client``.

Every fetcher here returns :class:`~app.provenance.Sourced` rather than a bare
float: a scrape or a third-party rate can fail, and a page must be able to tell
"4.18% scraped from Damodaran on 1 July 2026" apart from "the scrape broke and
you are looking at a constant". Nothing here substitutes a default on failure.

**Failures are never cached.** Each cached function *raises* :class:`FetchFailed`
on failure — :mod:`app.data.cache` does not store an exception — and a thin
uncached wrapper turns it into ``Sourced.unavailable``. Returning the failure
from the cached function instead, as this module once did, cached a transient
outage for the full day.
"""

import re
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from app.data import cache
from app.provenance import Sourced

#: An implied ERP outside this band is not credible and is treated as a failed
#: scrape rather than passed through. Consistent with the DCF spec's 2-10% band.
ERP_PLAUSIBLE_BAND = (0.02, 0.10)

DAMODARAN_URL = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm"


class FetchFailed(Exception):
    """A market-data fetch that must not be cached. Carries the reason."""


def _yahoo_last_close(symbol: str) -> tuple[float, date | None]:
    """The latest populated daily close for a Yahoo Finance symbol.

    Raises :class:`FetchFailed` with the reason. Several trading days are
    requested and the last populated one taken, so a weekend or a market
    holiday is not mistaken for an outage.
    """
    try:
        history = yf.Ticker(symbol).history(
            period="5d", interval="1d", auto_adjust=False, timeout=10
        )
    except Exception:
        raise FetchFailed(f"Could not reach Yahoo Finance for {symbol}") from None

    # yfinance absorbs an unknown symbol, a rate limit, and an outage alike into
    # an empty frame rather than raising, so empty is the failure path here and
    # not an edge case.
    if history is None or history.empty or "Close" not in history:
        raise FetchFailed(f"Yahoo Finance returned no quotes for {symbol}")
    closes = history["Close"].dropna()
    if closes.empty:
        raise FetchFailed(f"Yahoo Finance returned no closing level for {symbol}")

    stamp = closes.index[-1]
    return float(closes.iloc[-1]), (stamp.date() if hasattr(stamp, "date") else None)


#: How far before the requested day the last close may lie and still be read as
#: that day's level: a long weekend plus a holiday, not a gap in the series.
HISTORY_MAX_STALENESS_DAYS = 7


@cache.data(ttl=3600 * 24)
def _cached_yahoo_history(symbol: str) -> pd.Series:
    """Every daily close Yahoo Finance has for ``symbol``, indexed by date.

    One entry per symbol, whatever day is later asked for, so reading a level
    at each of twenty filing dates costs one request. Raises
    :class:`FetchFailed`, like the latest-close fetch.
    """
    try:
        history = yf.Ticker(symbol).history(
            period="max", interval="1d", auto_adjust=False, timeout=10
        )
    except Exception:
        raise FetchFailed(f"Could not reach Yahoo Finance for {symbol}") from None
    if history is None or history.empty or "Close" not in history:
        raise FetchFailed(f"Yahoo Finance returned no history for {symbol}")
    closes = history["Close"].dropna()
    if closes.empty:
        raise FetchFailed(f"Yahoo Finance returned no closing levels for {symbol}")
    closes.index = pd.DatetimeIndex([pd.Timestamp(t).date() for t in closes.index])
    return closes.sort_index()


def _yahoo_close_on(symbol: str, day: date) -> tuple[float, date]:
    """The close on ``day``, or the last one before it within a week.

    Raises :class:`FetchFailed` when the history cannot be read or has no close
    near enough — a stale level is not passed off as that day's.
    """
    closes = _cached_yahoo_history(symbol)
    known = closes[closes.index <= pd.Timestamp(day)]
    if known.empty:
        raise FetchFailed(f"Yahoo Finance has no {symbol} close on or before {day}")
    stamp = known.index[-1].date()
    if (day - stamp) > timedelta(days=HISTORY_MAX_STALENESS_DAYS):
        raise FetchFailed(
            f"The last {symbol} close before {day} is from {stamp}, more than "
            f"{HISTORY_MAX_STALENESS_DAYS} days earlier"
        )
    return float(known.iloc[-1]), stamp


# --- Treasury Rates (Yahoo Finance yield indices) ---

#: CBOE 13-week Treasury bill index — the short-term risk-free proxy. It quotes
#: the bill's *discount* rate, which sits a few basis points under the
#: bond-equivalent yield; close enough for a Sharpe-ratio input, and the
#: shortest tenor Yahoo publishes.
IRX_SYMBOL = "^IRX"

#: CBOE 10-year Treasury note yield index — the constant-maturity 10-year.
TNX_SYMBOL = "^TNX"

#: Both indices quote in percentage points, so 4.63 means 4.63%. A reading
#: outside this band means the quoting convention moved under us — Yahoo
#: published ^TNX at ten times the yield for years — and a rate wrong by a
#: factor of ten passes every downstream sanity check, so it is treated as a
#: failed fetch rather than passed through.
YIELD_PLAUSIBLE_BAND_PCT = (-1.0, 25.0)


@cache.data(ttl=3600 * 24)
def _cached_yahoo_yield(symbol: str) -> tuple[float, date | None]:
    """The cache entry: a plausible yield in percent and its date, or a raise."""
    rate_pct, as_of = _yahoo_last_close(symbol)
    low, high = YIELD_PLAUSIBLE_BAND_PCT
    if not low <= rate_pct <= high:
        raise FetchFailed(
            f"{symbol} quoted {rate_pct:.2f}, outside the plausible "
            f"{low:g}% to {high:g}% band, so the quoting convention is assumed "
            "to have changed"
        )
    return rate_pct, as_of


def _fetch_yahoo_yield(symbol: str) -> tuple[float | None, date | None, str]:
    """Fetches the latest daily close of a Yahoo Finance Treasury-yield index.

    Returns ``(rate_percent, as_of, error)``. ``rate_percent`` is ``None``
    exactly when the fetch failed, and ``error`` then says why. The reason is
    what lets callers that must not fabricate a rate explain themselves;
    callers that tolerate a default can ignore it. Uncached, so a failure is
    retried on the next call rather than remembered for a day.
    """
    try:
        rate_pct, as_of = _cached_yahoo_yield(symbol)
    except FetchFailed as failure:
        return None, None, str(failure)
    return rate_pct, as_of, ""


def fetch_current_risk_free_rate() -> Sourced:
    """Returns the current 13-week Treasury bill yield (``^IRX``) with its provenance.

    The short rate the portfolio pages use for Sharpe ratios. On failure the
    returned :class:`Sourced` has ``value=None`` and carries the reason, on the
    same contract as :func:`fetch_10y_treasury_rate`. It used to fall back to a
    hardcoded 4%, which a reader could not tell apart from a live quote.
    """
    rate_pct, as_of, error = _fetch_yahoo_yield(IRX_SYMBOL)
    if rate_pct is None:
        return Sourced.unavailable(
            source=f"Yahoo Finance ({IRX_SYMBOL}, 13-week Treasury bill)",
            reason=error or f"Yahoo Finance did not return a quote for {IRX_SYMBOL}",
        )

    return Sourced.reported(
        value=rate_pct / 100.0,
        source=f"Yahoo Finance {IRX_SYMBOL} — 13-week Treasury bill discount rate",
        as_of=as_of,
    )


def fetch_10y_treasury_rate() -> Sourced:
    """Returns the current 10-year Treasury rate (``^TNX``) with its provenance.

    On failure the returned :class:`Sourced` has ``value=None`` and carries the
    reason. Nothing is substituted — a DCF discounted at a made-up risk-free
    rate is worse than one the user had to fill in by hand.
    """
    rate_pct, as_of, error = _fetch_yahoo_yield(TNX_SYMBOL)
    if rate_pct is None:
        return Sourced.unavailable(
            source=f"Yahoo Finance ({TNX_SYMBOL}, 10-year Treasury yield)",
            reason=error or f"Yahoo Finance did not return a quote for {TNX_SYMBOL}",
        )

    return Sourced.reported(
        value=rate_pct / 100.0,
        source=f"Yahoo Finance {TNX_SYMBOL} — 10-year Treasury constant maturity",
        as_of=as_of,
    )


def fetch_10y_treasury_rate_on(day: date) -> Sourced:
    """The 10-year Treasury rate at the close on ``day`` (or just before it).

    For valuing a company as of a past filing date. Same source, band and
    contract as :func:`fetch_10y_treasury_rate`: unavailable, never substituted.
    """
    source = f"Yahoo Finance {TNX_SYMBOL} — 10-year Treasury constant maturity"
    try:
        rate_pct, stamp = _yahoo_close_on(TNX_SYMBOL, day)
    except FetchFailed as failure:
        return Sourced.unavailable(source=source, reason=str(failure))
    low, high = YIELD_PLAUSIBLE_BAND_PCT
    if not low <= rate_pct <= high:
        return Sourced.unavailable(
            source=source,
            reason=f"{TNX_SYMBOL} quoted {rate_pct:.2f} on {stamp}, outside the plausible "
            f"{low:g}% to {high:g}% band",
        )
    return Sourced.reported(value=rate_pct / 100.0, source=source, as_of=stamp)


# --- Equity Risk Premium & Beta ---


@cache.data(ttl=3600 * 24)
def _cached_erp() -> tuple[float, date | None]:
    """The cache entry: the scraped ERP and its date, or a :class:`FetchFailed`."""
    try:
        response = requests.get(DAMODARAN_URL, timeout=10)
    except Exception:
        raise FetchFailed("Could not reach the Damodaran page") from None

    if response.status_code != 200:
        raise FetchFailed(f"The Damodaran page returned HTTP {response.status_code}")

    # Tags are stripped to nothing, not to a space: the page splits the figure
    # across elements as `4.<span></span><span>18%`, so anything inserted
    # between them turns 4.18 into a 4.00 that looks perfectly reasonable.
    text = re.sub(r"<[^>]+>", "", response.text)
    text = text.replace("&nbsp;", " ")

    # The trailing % is required, not optional. Without it a truncated parse of
    # "4." matches happily and silently reports 4.00%.
    dated = re.search(
        r"Implied\s+ERP\s+on\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})\s*=\s*(\d+(?:\.\d+)?)\s*%",
        text,
        re.IGNORECASE,
    )
    as_of: date | None = None
    if dated:
        raw_value = dated.group(2)
        try:
            as_of = datetime.strptime(
                re.sub(r"\s+", " ", dated.group(1)).strip(), "%B %d, %Y"
            ).date()
        except ValueError:
            as_of = None
    else:
        # The number without a date is still usable, but say the date is missing.
        undated = re.search(
            r"Implied\s+ERP.*?=\s*(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE
        )
        if not undated:
            raise FetchFailed(
                "The 'Implied ERP' figure was not found — the page layout has changed"
            )
        raw_value = undated.group(1)

    try:
        erp = float(raw_value) / 100.0
    except ValueError:
        raise FetchFailed(f"Could not read '{raw_value}' as an ERP percentage") from None

    low, high = ERP_PLAUSIBLE_BAND
    if not low <= erp <= high:
        raise FetchFailed(
            f"Scraped ERP of {erp:.2%} is outside the plausible "
            f"{low:.0%}-{high:.0%} band, so the scrape is assumed broken"
        )
    return erp, as_of


def fetch_erp() -> Sourced:
    """Scrapes Aswath Damodaran's site for the current Implied Equity Risk Premium.

    The page states it as ``Implied ERP on July 1, 2026 = 4.18%``, so the as-of
    date comes off the same line as the number and both are reported together.

    Scraping is fragile by nature. If the request fails, the pattern stops
    matching, or the value lands outside :data:`ERP_PLAUSIBLE_BAND`, this
    returns an unavailable :class:`Sourced` with the reason. It never
    substitutes a constant — the caller is expected to fall through to manual
    entry and say why.
    """
    try:
        erp, as_of = _cached_erp()
    except FetchFailed as failure:
        return Sourced.unavailable(source=DAMODARAN_URL, reason=str(failure))
    return Sourced.reported(
        value=erp,
        source="Damodaran — Implied Equity Risk Premium (S&P 500)",
        as_of=as_of,
    )


# --- Exchange rates ---

#: A quote outside this band — USD per unit of a currency — is a broken or
#: misread quote rather than a rate. Wide on purpose: it spans the Kuwaiti dinar
#: (~3.3) to the Iranian rial (~0.00002).
FX_PLAUSIBLE_BAND = (1e-6, 1e3)


@cache.data(ttl=3600 * 6)
def _cached_fx_rate(currency: str) -> tuple[float, date | None, str]:
    """The cache entry: USD per unit of ``currency``, its date and the symbol read.

    Tries ``{CCY}USD=X`` (USD per unit) first, then ``USD{CCY}=X`` inverted —
    Yahoo lists some pairs only one way. Raises :class:`FetchFailed` naming both
    attempts when neither yields a plausible rate.
    """
    reasons = []
    for symbol, invert in ((f"{currency}USD=X", False), (f"USD{currency}=X", True)):
        try:
            close, as_of = _yahoo_last_close(symbol)
        except FetchFailed as failure:
            reasons.append(str(failure))
            continue
        if close <= 0:
            reasons.append(f"{symbol} quoted {close}")
            continue
        rate = 1.0 / close if invert else close
        low, high = FX_PLAUSIBLE_BAND
        if not low <= rate <= high:
            reasons.append(f"{symbol} implies {rate:g} USD per {currency}, not plausible")
            continue
        return rate, as_of, symbol
    raise FetchFailed("; ".join(reasons))


def fetch_fx_rate(currency: str) -> Sourced:
    """Spot US dollars per one unit of ``currency``, with its provenance.

    Unavailable — never substituted — when Yahoo has no usable quote either
    way round. Cached six hours on success; a failure is retried next call.
    """
    currency = currency.strip().upper()
    try:
        rate, as_of, symbol = _cached_fx_rate(currency)
    except FetchFailed as failure:
        return Sourced.unavailable(
            source=f"Yahoo Finance ({currency}USD=X / USD{currency}=X)",
            reason=str(failure),
        )
    inverted = symbol.startswith("USD")
    return Sourced.reported(
        value=rate,
        source=f"Yahoo Finance {symbol}" + (" (inverted)" if inverted else "")
        + f" — USD per {currency}",
        as_of=as_of,
    )



def fetch_fx_rate_on(currency: str, day: date) -> Sourced:
    """US dollars per one unit of ``currency`` at the close on ``day``.

    The as-of counterpart of :func:`fetch_fx_rate`, for converting a past
    filing's base period at the rate of its own date. Tries the same two pairs,
    the same way round.
    """
    currency = currency.strip().upper()
    reasons = []
    for symbol, invert in ((f"{currency}USD=X", False), (f"USD{currency}=X", True)):
        try:
            close, stamp = _yahoo_close_on(symbol, day)
        except FetchFailed as failure:
            reasons.append(str(failure))
            continue
        if close <= 0:
            reasons.append(f"{symbol} quoted {close} on {stamp}")
            continue
        rate = 1.0 / close if invert else close
        low, high = FX_PLAUSIBLE_BAND
        if not low <= rate <= high:
            reasons.append(f"{symbol} implies {rate:g} USD per {currency}, not plausible")
            continue
        return Sourced.reported(
            value=rate,
            source=f"Yahoo Finance {symbol}" + (" (inverted)" if invert else "")
            + f" — USD per {currency}",
            as_of=stamp,
        )
    return Sourced.unavailable(
        source=f"Yahoo Finance ({currency}USD=X / USD{currency}=X)", reason="; ".join(reasons)
    )
