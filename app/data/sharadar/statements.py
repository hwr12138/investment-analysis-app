"""``SF1`` for one company: the four statement frames, and views over them."""

from __future__ import annotations

import pandas as pd

from app.data import cache
from app.data.sharadar import reference, transport
from app.data.sharadar.transport import TickerNotFoundError
from app.data.sharadar.vocabulary import (
    MINORITY_INTEREST_CAPTION,
    SF1_CAPTIONS,
    SF1_LAYOUT,
    SIGN_FLIPPED,
    StatementSet,
)

_STATEMENT_KEYS = ("PL", "BS", "CF", "DERIVED")


def dimension_for(*, annual: bool, ttm: bool, asreported: bool) -> str:
    """Builds the SF1 ``dimension`` code.

    ``AR`` excludes restatements and time-indexes to the SEC filing date; ``MR``
    includes them and time-indexes to the report period. The ``asreported``
    toggle maps onto that pair exactly.
    """
    base = "AR" if asreported else "MR"
    if ttm:
        return f"{base}T"
    return f"{base}Y" if annual else f"{base}Q"


def _split_fiscal_period(value: object) -> tuple[int | None, str | None]:
    """``"2025-FY"`` → ``(2025, "FY")``; ``"2026-Q3"`` → ``(2026, "Q3")``."""
    text = str(value or "").strip()
    if "-" not in text:
        return None, None
    year, _, period = text.partition("-")
    try:
        return int(year), period.upper()
    except ValueError:
        return None, period.upper() or None


def _sf1_frames(
    rows: pd.DataFrame, *, source_link: str, restated: bool
) -> dict[str, pd.DataFrame]:
    """Partitions the wide SF1 result into the four statement frames.

    Every frame carries the same metadata columns, because every frame comes
    from the same SF1 row. That is why an annual request never splits a fiscal
    year across two half-populated rows.
    """
    numeric = rows.copy()
    for indicator in SF1_CAPTIONS:
        if indicator in numeric.columns:
            numeric[indicator] = pd.to_numeric(numeric[indicator], errors="coerce")

    for indicator in SIGN_FLIPPED:
        if indicator in numeric.columns:
            numeric[indicator] = -numeric[indicator]

    parsed = [_split_fiscal_period(v) for v in numeric.get("fiscalperiod", [])]
    meta = pd.DataFrame(
        {
            "Fiscal Year": pd.array([p[0] for p in parsed], dtype="Int64"),
            "Fiscal Period": [p[1] for p in parsed],
            "Report Date": pd.to_datetime(numeric.get("reportperiod"), errors="coerce"),
            "Publish Date": pd.to_datetime(numeric.get("datekey"), errors="coerce"),
            "Calendar Date": pd.to_datetime(numeric.get("calendardate"), errors="coerce"),
            "Dimension": numeric.get("dimension"),
            "Restated": "Yes" if restated else "No",
            "Source": source_link or None,
            "Value Check": None,
        },
        index=numeric.index,
    )

    # SF1 has no balance-sheet non-controlling interest line, but its `equity`
    # is explicitly "attributable to parent", so the residual recovers it.
    derived_nci: pd.Series | None = None
    if {"assets", "liabilities", "equity"}.issubset(numeric.columns):
        derived_nci = numeric["assets"] - numeric["liabilities"] - numeric["equity"]

    frames: dict[str, pd.DataFrame] = {}
    for statement in _STATEMENT_KEYS:
        captions = {
            indicator: caption
            for indicator, caption, target in SF1_LAYOUT
            if target == statement and indicator in numeric.columns
        }
        body = numeric[list(captions)].rename(columns=captions)
        if statement == "BS" and derived_nci is not None:
            body = body.assign(**{MINORITY_INTEREST_CAPTION: derived_nci})
        frames[statement] = pd.concat([meta, body], axis=1).reset_index(drop=True)

    return frames


def _order_statement_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Sorts chronologically and drops superseded filings.

    Sharadar returns rows unsorted by contract. The ``AR`` dimensions can also
    emit more than one row for a fiscal period when a company files twice in a
    quarter; the row with the later filing date supersedes.
    """
    if frame.empty or "Report Date" not in frame.columns:
        return frame

    ordered = frame.sort_values(
        [c for c in ("Report Date", "Publish Date") if c in frame.columns]
    )
    if {"Fiscal Year", "Fiscal Period"}.issubset(ordered.columns):
        ordered = ordered.drop_duplicates(
            subset=["Fiscal Year", "Fiscal Period"], keep="last"
        )
    return ordered.reset_index(drop=True)


def _fetch_statements(
    ticker: str,
    *,
    annual: bool = True,
    ttm: bool = False,
    asreported: bool = False,
) -> StatementSet:
    dimension = dimension_for(annual=annual, ttm=ttm, asreported=asreported)
    # The cached lookup, not the bare one: every dimension asks, and the research
    # page, the DCF and the tabs ask for four of them per company.
    meta = reference.fetch_ticker_meta(ticker)

    rows = transport._request_all(
        "SF1", {"ticker": ticker, "dimension": dimension}, ticker=ticker
    )
    if rows.empty:
        raise TickerNotFoundError(
            f"Sharadar returned no {dimension} fundamentals for {ticker}.",
            table="SF1",
            ticker=ticker,
        )

    frames = {
        name: _order_statement_rows(frame)
        for name, frame in _sf1_frames(
            rows,
            source_link=str(meta.get("secfilings") or ""),
            restated=not asreported,
        ).items()
    }

    sic_code = meta.get("siccode")
    return StatementSet(
        ticker=ticker.strip().upper(),
        permaticker=int(meta.get("permaticker") or 0),
        name=str(meta.get("name") or ""),
        currency=str(meta.get("currency") or ""),
        # `template` is the generic classification string the schema resolver
        # and the unsupported-sector error report back to the user.
        template=str(meta.get("sector") or meta.get("sicsector") or ""),
        dimension=dimension,
        period="FY" if annual and not ttm else "Q1,Q2,Q3,Q4",
        ttm=ttm,
        asreported=asreported,
        sector=str(meta.get("sector") or ""),
        industry=str(meta.get("industry") or ""),
        sic_code=int(sic_code) if sic_code is not None and not pd.isna(sic_code) else None,
        frames=frames,
    )


@cache.data(ttl=60 * 60 * 24 * 3)
def _cached_statements(
    ticker: str, annual: bool, ttm: bool, asreported: bool
) -> StatementSet:
    """The cache entry. Positional only — see :func:`fetch_statements`."""
    return _fetch_statements(ticker, annual=annual, ttm=ttm, asreported=asreported)


def fetch_statements(
    ticker: str,
    *,
    annual: bool = True,
    ttm: bool = False,
    asreported: bool = False,
) -> StatementSet:
    """Fundamentals for one company, sorted chronologically and de-duplicated.

    ``asreported=True`` selects Sharadar's ``AR`` dimensions (excludes
    restatements, indexed to the SEC filing date); the default selects ``MR``
    (includes restatements, indexed to the report period).

    The cache sits on ``_cached_statements`` rather than here, and this wrapper
    exists only to fill the defaults in before the key is built. The cache
    (:mod:`app.data.cache`) keys on the arguments *as passed*, without applying
    defaults: ``fetch_statements(t, annual=True)`` and
    ``fetch_statements(t, annual=True, ttm=False)`` are the same request but
    hashed differently, and each would fetch the same ``MRY`` rows separately.
    That is not hypothetical — the research page passed all three keywords and
    the DCF omitted ``ttm``, so one search fetched ``MRY`` and ``MRQ`` twice
    each. Forwarding positionally makes every call form collapse onto one key.
    """
    return _cached_statements(ticker, annual, ttm, asreported)


# ---------------------------------------------------------------------------
# Views over a StatementSet
# ---------------------------------------------------------------------------
#
# Sharadar carries share counts and the filing date on the SF1 row itself, so
# these are projections of a frame already in hand rather than network calls.
# They keep the column names their consumers expect, and they keep the knowledge
# of which Sharadar indicator means what inside this module.


def common_shares_frame(statements: StatementSet) -> pd.DataFrame:
    """Point-in-time shares outstanding, as ``endDate``/``value``.

    Sharadar's ``sharesbas`` is the count stated on the cover of the 10-K or
    10-Q, adjusted for splits. ``sharefactor`` is applied because it is what
    Sharadar itself multiplies through when computing market cap — it corrects
    for ADRs representing a fraction or multiple of an underlying share, and
    for multi-class structures.
    """
    empty = pd.DataFrame(columns=["endDate", "value"])
    frame = statements.frame("DERIVED")
    caption = SF1_CAPTIONS["sharesbas"]
    if frame.empty or caption not in frame.columns:
        return empty

    factor = frame.get(SF1_CAPTIONS["sharefactor"])
    shares = pd.to_numeric(frame[caption], errors="coerce")
    if factor is not None:
        shares = shares * pd.to_numeric(factor, errors="coerce").fillna(1.0)

    out = pd.DataFrame({"endDate": frame["Report Date"], "value": shares})
    return out.dropna(subset=["endDate", "value"]).sort_values("endDate").reset_index(
        drop=True
    )


def share_factor(statements: StatementSet) -> float | None:
    """The latest ``sharefactor`` — ADR shares per underlying share, and the like.

    :func:`common_shares_frame` multiplies it through already. Weighted counts
    and per-share figures (EPS) are published per *underlying* share, so any
    count compared with a price per ADR has to be multiplied by it too. ``None``
    when the column is absent or empty, which callers read as a factor of one.
    """
    frame = statements.frame("DERIVED")
    caption = SF1_CAPTIONS["sharefactor"]
    if frame.empty or caption not in frame.columns:
        return None
    ordered = frame.sort_values("Report Date") if "Report Date" in frame.columns else frame
    values = pd.to_numeric(ordered[caption], errors="coerce").dropna()
    values = values[values > 0]
    return float(values.iloc[-1]) if not values.empty else None


def weighted_shares_frame(statements: StatementSet) -> pd.DataFrame:
    """Weighted-average basic and diluted counts per fiscal period."""
    columns = ["endDate", "fyear", "period", "basic", "diluted"]
    frame = statements.frame("PL")
    basic_col = SF1_CAPTIONS["shareswa"]
    diluted_col = SF1_CAPTIONS["shareswadil"]
    if frame.empty or basic_col not in frame.columns:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(
        {
            "endDate": frame["Report Date"],
            "fyear": frame["Fiscal Year"],
            "period": frame["Fiscal Period"],
            "basic": pd.to_numeric(frame[basic_col], errors="coerce"),
            "diluted": pd.to_numeric(frame.get(diluted_col), errors="coerce"),
        }
    )
    return out.dropna(subset=["endDate"]).sort_values("endDate").reset_index(drop=True)


def filings_frame(statements: StatementSet) -> pd.DataFrame:
    """The filing behind each period, for the provenance panel.

    Sharadar has no per-filing URL, so ``sourceLink`` is the company's EDGAR
    filing index — enough to reach the document in one click, but not a direct
    link to it. Under the ``AR`` dimensions ``datekey`` is the SEC filing date;
    under ``MR`` it is the report period, so the filing date is only meaningful
    for as-reported runs and is labelled as such upstream.
    """
    columns = ["fyear", "period", "filingDate", "filingType", "sourceLink"]
    frame = statements.frame("PL")
    if frame.empty or "Fiscal Year" not in frame.columns:
        return pd.DataFrame(columns=columns)

    periods = frame["Fiscal Period"]
    out = pd.DataFrame(
        {
            "fyear": frame["Fiscal Year"],
            "period": periods,
            "filingDate": frame["Publish Date"],
            "filingType": ["10-K" if p == "FY" else "10-Q" for p in periods],
            "sourceLink": frame["Source"],
        }
    )
    return out.dropna(subset=["fyear"]).sort_values("filingDate").reset_index(drop=True)


def fiscal_year_end_month(statements: StatementSet) -> int | None:
    """The month a fiscal year ends in, from the latest annual report period.

    Sharadar publishes no fiscal-calendar attribute, so it is read off the
    data. Only annual statements carry it — a quarterly set would report
    whichever quarter came last.
    """
    frame = statements.frame("PL")
    if frame.empty or "Report Date" not in frame.columns:
        return None

    annual = frame
    if "Fiscal Period" in frame.columns:
        annual = frame[frame["Fiscal Period"] == "FY"]
        if annual.empty:
            return None

    dates = pd.to_datetime(annual["Report Date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return int(dates.iloc[-1].month)
