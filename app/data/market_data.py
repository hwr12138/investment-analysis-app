"""Rates and the equity risk premium.

The two market inputs that do not come from a securities feed: Treasury yields
proxied by the CBOE yield indices on Yahoo Finance, and the implied equity risk
premium scraped from Damodaran. Prices and fundamentals live in the sibling
``sharadar_client``.

Every fetcher here returns :class:`~app.provenance.Sourced` rather than a bare
float: a scrape or a third-party rate can fail, and a page must be able to tell
"4.18% scraped from Damodaran on 1 July 2026" apart from "the scrape broke and
you are looking at a constant". Nothing here substitutes a default on failure.
"""

import re
from datetime import date, datetime

import requests
import streamlit as st
import yfinance as yf

from app.provenance import Sourced

#: An implied ERP outside this band is not credible and is treated as a failed
#: scrape rather than passed through. Consistent with the DCF spec's 2-10% band.
ERP_PLAUSIBLE_BAND = (0.02, 0.10)

DAMODARAN_URL = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm"


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


@st.cache_data(show_spinner=False, ttl=3600 * 24)
def _fetch_yahoo_yield(symbol: str) -> tuple[float | None, date | None, str]:
    """Fetches the latest daily close of a Yahoo Finance Treasury-yield index.

    Returns ``(rate_percent, as_of, error)``. ``rate_percent`` is ``None``
    exactly when the fetch failed, and ``error`` then says why. The reason is
    what lets callers that must not fabricate a rate explain themselves;
    callers that tolerate a default can ignore it.
    """
    try:
        history = yf.Ticker(symbol).history(
            period="5d", interval="1d", auto_adjust=False, timeout=10
        )
    except Exception:
        return None, None, f"Could not reach Yahoo Finance for {symbol}"

    # yfinance absorbs an unknown symbol, a rate limit, and an outage alike into
    # an empty frame rather than raising, so empty is the failure path here and
    # not an edge case.
    if history is None or history.empty or "Close" not in history:
        return None, None, f"Yahoo Finance returned no quotes for {symbol}"

    # Several trading days are requested and the last populated one taken, so a
    # weekend or a market holiday is not mistaken for an outage.
    closes = history["Close"].dropna()
    if closes.empty:
        return None, None, f"Yahoo Finance returned no closing level for {symbol}"

    rate_pct = float(closes.iloc[-1])
    low, high = YIELD_PLAUSIBLE_BAND_PCT
    if not low <= rate_pct <= high:
        return None, None, (
            f"{symbol} quoted {rate_pct:.2f}, outside the plausible "
            f"{low:g}% to {high:g}% band, so the quoting convention is assumed "
            "to have changed"
        )

    stamp = closes.index[-1]
    as_of = stamp.date() if hasattr(stamp, "date") else None
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


# --- Equity Risk Premium & Beta ---


@st.cache_data(show_spinner=False, ttl=3600 * 24)
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
        response = requests.get(DAMODARAN_URL, timeout=10)
    except Exception:
        return Sourced.unavailable(
            source=DAMODARAN_URL, reason="Could not reach the Damodaran page"
        )

    if response.status_code != 200:
        return Sourced.unavailable(
            source=DAMODARAN_URL,
            reason=f"The Damodaran page returned HTTP {response.status_code}",
        )

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
            return Sourced.unavailable(
                source=DAMODARAN_URL,
                reason="The 'Implied ERP' figure was not found — the page layout has changed",
            )
        raw_value = undated.group(1)

    try:
        erp = float(raw_value) / 100.0
    except ValueError:
        return Sourced.unavailable(
            source=DAMODARAN_URL,
            reason=f"Could not read '{raw_value}' as an ERP percentage",
        )

    low, high = ERP_PLAUSIBLE_BAND
    if not low <= erp <= high:
        return Sourced.unavailable(
            source=DAMODARAN_URL,
            reason=(
                f"Scraped ERP of {erp:.2%} is outside the plausible "
                f"{low:.0%}-{high:.0%} band, so the scrape is assumed broken"
            ),
        )

    return Sourced.reported(
        value=erp,
        source="Damodaran — Implied Equity Risk Premium (S&P 500)",
        as_of=as_of,
    )
