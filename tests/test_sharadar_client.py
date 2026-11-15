"""Datatable parsing, sign and unit normalisation, and the API key never leaking."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data import settings
from app.data import sharadar_client as sc
from app.data.sharadar import actions as sh_actions
from app.data.sharadar import insiders as sh_insiders
from app.data.sharadar import prices as sh_prices
from app.data.sharadar import reference as sh_reference
from app.data.sharadar import statements as sh_statements
from app.data.sharadar import transport as sh_transport
from tests.conftest import SF1_COLUMNS, sf1_payload, tickers_payload


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parses_the_datatable_envelope():
    payload = {
        "datatable": {
            "columns": [{"name": "ticker"}, {"name": "revenue"}],
            "data": [["AAPL", 100], ["MSFT", 120]],
        },
        "meta": {"next_cursor_id": None},
    }
    frame = sc.parse_datatable(payload)
    assert list(frame.columns) == ["ticker", "revenue"]
    assert frame["revenue"].tolist() == [100, 120]


def test_column_order_may_differ_between_calls():
    """Sharadar appends indicators over time, so nothing may depend on position."""
    first = sc.parse_datatable(
        {"datatable": {"columns": [{"name": n} for n in ("ticker", "revenue", "netinc")],
                       "data": [["TEST", 1200, 240]]}}
    )
    second = sc.parse_datatable(
        {"datatable": {"columns": [{"name": n} for n in ("netinc", "ticker", "revenue")],
                       "data": [[240, "TEST", 1200]]}}
    )
    for column in ("ticker", "revenue", "netinc"):
        assert first[column].iloc[0] == second[column].iloc[0]


def test_empty_payload_yields_an_empty_frame():
    assert sc.parse_datatable({}).empty
    assert sc.parse_datatable({"datatable": {"columns": [], "data": []}}).empty


def test_rows_are_sorted_chronologically(monkeypatch, statements_payload, tickers_response):
    """The fixture rows arrive 2025, 2023, 2024 — the client must reorder them."""
    _stub_tables(monkeypatch, sf1=statements_payload, tickers=tickers_response)
    statements = sh_statements._fetch_statements("TEST")
    years = statements.frame("PL")["Fiscal Year"].tolist()
    assert years == [2023, 2024, 2025]


def test_duplicate_period_keeps_the_later_filing(monkeypatch, annual_rows, tickers_response):
    """AR dimensions can emit two rows for one period when a company refiles."""
    restated = list(annual_rows[0])
    restated[SF1_COLUMNS.index("datekey")] = "2026-02-01"
    restated[SF1_COLUMNS.index("revenue")] = 1250
    _stub_tables(
        monkeypatch,
        sf1=sf1_payload(annual_rows + [restated]),
        tickers=tickers_response,
    )
    frame = sh_statements._fetch_statements("TEST").frame("PL")
    assert len(frame) == 3
    assert frame["Revenues"].iloc[-1] == 1250


# ---------------------------------------------------------------------------
# Normalisation the engine depends on
# ---------------------------------------------------------------------------


def test_expense_lines_are_negated_on_ingest(monkeypatch, statements_payload, tickers_response):
    """Sharadar publishes expenses positive; the engine's contract is negative."""
    _stub_tables(monkeypatch, sf1=statements_payload, tickers=tickers_response)
    pl = sh_statements._fetch_statements("TEST").frame("PL")
    latest = pl.iloc[-1]
    assert latest["Cost of Revenue"] == -700
    assert latest["Operating Expenses"] == -190
    assert latest["Income Tax Expense"] == -60


def test_capex_is_not_negated(monkeypatch, statements_payload, tickers_response):
    """Sharadar already signs capex negative — flipping it would invert FCFF."""
    _stub_tables(monkeypatch, sf1=statements_payload, tickers=tickers_response)
    cf = sh_statements._fetch_statements("TEST").frame("CF")
    assert cf["Capital Expenditure"].iloc[-1] == -100


def test_income_statement_identity_holds_after_sign_flips(
    monkeypatch, statements_payload, tickers_response
):
    """Revenue + cost of revenue + operating expenses = operating income."""
    _stub_tables(monkeypatch, sf1=statements_payload, tickers=tickers_response)
    row = sh_statements._fetch_statements("TEST").frame("PL").iloc[-1]
    computed = row["Revenues"] + row["Cost of Revenue"] + row["Operating Expenses"]
    assert computed == pytest.approx(row["Operating Income"])


def test_minority_interest_is_derived_as_the_equity_residual(
    monkeypatch, annual_rows, tickers_response
):
    """Sharadar's equity is parent-only, so NCI is assets − liabilities − equity."""
    row = list(annual_rows[0])
    # Push equity below the residual so a non-zero NCI has to appear.
    row[SF1_COLUMNS.index("equity")] = 300.0  # assets 1060 − liabilities 700 = 360
    _stub_tables(monkeypatch, sf1=sf1_payload([row]), tickers=tickers_response)
    bs = sh_statements._fetch_statements("TEST").frame("BS")
    assert bs[sc.MINORITY_INTEREST_CAPTION].iloc[-1] == pytest.approx(60.0)


def test_fiscal_period_is_split_into_year_and_period(
    monkeypatch, statements_payload, tickers_response
):
    _stub_tables(monkeypatch, sf1=statements_payload, tickers=tickers_response)
    pl = sh_statements._fetch_statements("TEST").frame("PL")
    assert pl["Fiscal Year"].tolist() == [2023, 2024, 2025]
    assert set(pl["Fiscal Period"]) == {"FY"}


def test_every_statement_frame_carries_the_same_metadata(
    monkeypatch, statements_payload, tickers_response
):
    """All four frames come off one SF1 row, so the merge key is identical."""
    _stub_tables(monkeypatch, sf1=statements_payload, tickers=tickers_response)
    statements = sh_statements._fetch_statements("TEST")
    reference = statements.frame("PL")[["Fiscal Year", "Fiscal Period", "Report Date"]]
    for code in ("BS", "CF", "DERIVED"):
        other = statements.frame(code)[["Fiscal Year", "Fiscal Period", "Report Date"]]
        pd.testing.assert_frame_equal(reference, other)


def test_daily_market_cap_and_ev_are_rescaled_to_raw_units(monkeypatch):
    """DAILY publishes millions; SF1 and everything downstream use raw units."""
    payload = {
        "datatable": {
            "columns": [{"name": n} for n in ("ticker", "date", "marketcap", "ev", "evebitda")],
            "data": [["AAPL", "2026-08-03", 4428166.1, 4472966.1, 26.5]],
        },
        "meta": {"next_cursor_id": None},
    }
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: payload)
    frame = sh_prices._fetch_daily_metrics("AAPL")
    assert frame["marketcap"].iloc[0] == pytest.approx(4.4281661e12)
    assert frame["ev"].iloc[0] == pytest.approx(4.4729661e12)
    # Ratios are dimensionless and must not be touched.
    assert frame["evebitda"].iloc[0] == pytest.approx(26.5)


def test_daily_metrics_exposes_the_multiples_that_fetch_prices_drops(monkeypatch):
    """``fetch_prices`` keeps only marketcap and evebitda; the research page needs the rest."""
    payload = {
        "datatable": {
            "columns": [
                {"name": n}
                for n in ("ticker", "date", "ev", "evebit", "evebitda", "marketcap", "pb", "pe", "ps")
            ],
            "data": [["AAPL", "2026-08-05", 4583590.0, 29.4, 27.1, 4538790.0, 42.2, 35.2, 9.7]],
        },
        "meta": {"next_cursor_id": None},
    }
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: payload)
    frame = sh_prices._fetch_daily_metrics("AAPL")
    for column in ("pe", "pb", "ps", "evebit"):
        assert column in frame.columns
    assert frame["pe"].iloc[0] == pytest.approx(35.2)
    # Still raw units, because the wrapper shares one fetcher with fetch_prices.
    assert frame["marketcap"].iloc[0] == pytest.approx(4.53879e12)


# ---------------------------------------------------------------------------
# Corporate actions
# ---------------------------------------------------------------------------


def test_actions_are_sorted_oldest_first(monkeypatch, actions_response):
    """ACTIONS arrives newest-first — the reverse of every other table here."""
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: actions_response)
    frame = sh_actions._fetch_actions("TEST")
    dates = frame["date"].tolist()
    assert dates == sorted(dates)
    assert frame["date"].iloc[-1] == pd.Timestamp("2025-11-10")


def test_actions_normalise_the_na_contra_columns(monkeypatch, actions_response):
    """Sharadar writes the string "N/A", which would otherwise render as a ticker."""
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: actions_response)
    frame = sh_actions._fetch_actions("TEST")
    dividends = frame[frame["action"] == sc.DIVIDEND_ACTION]
    assert dividends["contraticker"].isna().all()
    # A real counterparty survives.
    change = frame[frame["action"] == "tickerchangeto"]
    assert change["contraticker"].iloc[0] == "OLDT"


def test_actions_dividend_rows_carry_the_ex_date_and_amount(monkeypatch, actions_response):
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: actions_response)
    frame = sh_actions._fetch_actions("TEST")
    latest = frame[frame["action"] == sc.DIVIDEND_ACTION].iloc[-1]
    assert latest["date"] == pd.Timestamp("2025-11-10")
    assert latest["value"] == pytest.approx(0.27)


def test_no_actions_returns_an_empty_frame_rather_than_raising(monkeypatch):
    """A company that has never paid a dividend is ordinary, not an error."""
    empty = {"datatable": {"columns": [], "data": []}, "meta": {"next_cursor_id": None}}
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: empty)
    assert sh_actions._fetch_actions("TEST").empty


# ---------------------------------------------------------------------------
# Insider transactions
# ---------------------------------------------------------------------------


def test_insider_rows_are_sorted_by_filing_date(monkeypatch, insiders_response):
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: insiders_response)
    frame = sh_insiders._fetch_insider_transactions("TEST")
    dates = frame["filingdate"].tolist()
    assert dates == sorted(dates)


def test_two_lines_in_one_filing_are_ordered_by_row_number(
    monkeypatch, insiders_response
):
    """Each line carries a running balance; only the last one is end-of-day."""
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: insiders_response)
    frame = sh_insiders._fetch_insider_transactions("TEST")
    same_day = frame[
        (frame["ownername"] == "JANE OFFICER")
        & (frame["filingdate"] == pd.Timestamp("2026-07-06"))
    ]
    assert same_day["rownum"].tolist() == [1, 2, 3]
    assert same_day["sharesownedfollowingtransaction"].iloc[-1] == 72_000


def test_insider_dates_and_numbers_are_typed(monkeypatch, insiders_response):
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: insiders_response)
    frame = sh_insiders._fetch_insider_transactions("TEST")
    assert str(frame["filingdate"].dtype).startswith("datetime64")
    assert str(frame["transactiondate"].dtype).startswith("datetime64")
    # Numeric, but not necessarily float: ``transactionshares`` comes back
    # integral when no row in the response is null, and float when one is.
    assert frame["transactionshares"].dtype.kind in "if"
    assert frame["transactionpricepershare"].dtype.kind == "f"


def test_a_restated_filing_supersedes_the_row_it_corrects(
    monkeypatch, insiders_response
):
    """Ninety of 142 live restated rows repeated an existing trade verbatim."""
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: insiders_response)
    frame = sh_insiders._fetch_insider_transactions("TEST")
    sale = frame[
        (frame["ownername"] == "LATE FILER")
        & (frame["transactiondate"] == pd.Timestamp("2026-06-30"))
    ]
    assert len(sale) == 1
    assert sale["formtype"].iloc[0] == "RESTATED - 4"


def test_holdings_rows_are_never_deduplicated_against_each_other(
    monkeypatch, insiders_response
):
    """Every field of the transaction key is null or zero on a Form 3 line, so
    one blanket dedup would collapse a company's whole holdings history."""
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: insiders_response)
    frame = sh_insiders._fetch_insider_transactions("TEST")
    quiet = frame[frame["ownername"] == "QUIET HOLDER"]
    assert len(quiet) == 2
    assert set(quiet["securityadcode"]) == {"N", "D"}


def test_the_transaction_value_column_is_left_unsigned(
    monkeypatch, insiders_response
):
    """The client does not repair it. Quirk 6 documents it; the logic layer
    ignores the column entirely and multiplies shares by price instead."""
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: insiders_response)
    frame = sh_insiders._fetch_insider_transactions("TEST")
    sale = frame[
        (frame["ownername"] == "JANE OFFICER")
        & (frame["transactioncode"] == "S")
        & (frame["transactionshares"] == -60_000)
    ].iloc[0]
    assert sale["transactionshares"] == -60_000
    assert sale["transactionvalue"] == 6_000_000  # positive, as sent


def test_sf2_columns_are_not_in_sign_flipped():
    """That set negates positive magnitudes; ``transactionshares`` is already
    correct and negating it would invert every trade."""
    assert not sc.SIGN_FLIPPED & {"transactionshares", "transactionvalue"}


def test_insider_codes_are_upper_cased_on_ingest(monkeypatch):
    """Consumers test frozenset membership, which cannot miss on case."""
    from tests.conftest import SF2_COLUMNS, SF2_ROWS, sf2_payload

    lowered = [list(row) for row in SF2_ROWS]
    code = SF2_COLUMNS.index("transactioncode")
    for row in lowered:
        if row[code]:
            row[code] = row[code].lower()
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: sf2_payload(lowered))
    frame = sh_insiders._fetch_insider_transactions("TEST")
    assert "p" not in set(frame["transactioncode"].dropna())
    assert "P" in set(frame["transactioncode"].dropna())


def test_the_insider_fetch_filters_on_the_filing_date(monkeypatch, insiders_response):
    """``transactiondate`` is not one of SF2's filterable parameters, so the
    window has to be requested on the filing date and trimmed downstream."""
    seen: list[dict] = []

    def capture(table, query, **kwargs):
        seen.append(query)
        return insiders_response

    monkeypatch.setattr(sh_transport, "_get", capture)
    sh_insiders._fetch_insider_transactions("TEST", "2021-08-05")
    assert seen[0]["filingdate.gte"] == "2021-08-05"
    assert "transactiondate.gte" not in seen[0]


def test_the_insider_projection_omits_the_always_null_issuer_name(
    monkeypatch, insiders_response
):
    """Null on 0 of 31,724 live rows; ``fetch_ticker_meta`` is the name source."""
    seen: list[dict] = []

    def capture(table, query, **kwargs):
        seen.append(query)
        return insiders_response

    monkeypatch.setattr(sh_transport, "_get", capture)
    sh_insiders._fetch_insider_transactions("TEST")
    projection = seen[0]["qopts.columns"]
    assert "issuername" not in projection
    assert "rownum" in projection  # part of the primary key, kept deliberately


def test_no_insider_filings_returns_an_empty_frame_rather_than_raising(monkeypatch):
    """A company whose insiders have never filed is ordinary, not an error."""
    empty = {"datatable": {"columns": [], "data": []}, "meta": {"next_cursor_id": None}}
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: empty)
    assert sh_insiders._fetch_insider_transactions("TEST").empty


# ---------------------------------------------------------------------------
# Zero-versus-missing
# ---------------------------------------------------------------------------


def test_zero_filled_indicators_are_recognised_by_caption():
    assert sc.is_zero_filled("Interest Expense")
    assert sc.is_zero_filled("Capital Expenditure")
    # Not a zero-filled indicator: Sharadar always computes it.
    assert not sc.is_zero_filled("Operating Income")
    # Not a Sharadar indicator at all.
    assert not sc.is_zero_filled("Nonsense Line")


def test_zeros_are_passed_through_rather_than_blanked(
    monkeypatch, statements_payload, tickers_response
):
    """The client must not guess; a 0 stays a 0 and the validation layer flags it."""
    _stub_tables(monkeypatch, sf1=statements_payload, tickers=tickers_response)
    pl = sh_statements._fetch_statements("TEST").frame("PL")
    assert pl["Interest Expense"].iloc[-1] == 0.0
    assert pl["Interest Expense"].notna().all()


# ---------------------------------------------------------------------------
# Sector gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sic_code, sector, expected",
    [
        (6021, "Financial Services", True),      # bank
        (6311, "Financial Services", True),      # life insurance
        (6999, "Anything", True),                # top of SIC division H
        (5999, "Retail", False),                 # just below
        (7000, "Services", False),               # just above
        (3571, "Technology", False),             # AAPL
        (None, "Financial Services", True),      # no SIC, vendor label catches it
        (None, "Technology", False),
    ],
)
def test_financial_sector_detection(sic_code, sector, expected):
    statements = sc.StatementSet(
        ticker="X", permaticker=1, name="X", currency="USD", template=sector,
        dimension="MRY", period="FY", ttm=False, asreported=False,
        sic_code=sic_code, frames={},
    )
    assert statements.is_financial_template is expected


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "annual, ttm, asreported, expected",
    [
        (True, False, False, "MRY"),
        (False, False, False, "MRQ"),
        (False, True, False, "MRT"),
        (True, False, True, "ARY"),
        (False, False, True, "ARQ"),
        (False, True, True, "ART"),
    ],
)
def test_dimension_codes(annual, ttm, asreported, expected):
    assert sc.dimension_for(annual=annual, ttm=ttm, asreported=asreported) == expected


def test_as_reported_and_restated_are_labelled_distinctly(
    monkeypatch, statements_payload, tickers_response
):
    _stub_tables(monkeypatch, sf1=statements_payload, tickers=tickers_response)
    assert sh_statements._fetch_statements("TEST").frame("PL")["Restated"].iloc[0] == "Yes"
    assert (
        sh_statements._fetch_statements("TEST", asreported=True).frame("PL")["Restated"].iloc[0]
        == "No"
    )


# ---------------------------------------------------------------------------
# The statement cache key
# ---------------------------------------------------------------------------


def test_omitting_a_default_keyword_still_hits_the_cache(monkeypatch):
    """One request, however the caller spells the defaults.

    The cache keys on the arguments as passed and does *not* apply defaults
    (it kept ``st.cache_data``'s rule when it replaced it), so ``fetch_statements(t, annual=True)`` and
    ``fetch_statements(t, annual=True, ttm=False)`` would hash differently and
    fetch the same ``MRY`` rows twice. The research page spells all three
    keywords and the DCF spells two, so this cost one duplicate SF1 request per
    dimension on every search until `fetch_statements` began forwarding
    positionally to a private cached function.
    """
    calls = []

    def spy(ticker, *, annual=True, ttm=False, asreported=False):
        calls.append((ticker, annual, ttm, asreported))
        return sc.StatementSet(
            ticker=ticker, permaticker=1, name=ticker, currency="USD", template="",
            dimension=sc.dimension_for(annual=annual, ttm=ttm, asreported=asreported),
            period="FY", ttm=ttm, asreported=asreported, frames={},
        )

    monkeypatch.setattr(sh_statements, "_fetch_statements", spy)
    sh_statements._cached_statements.clear()

    sc.fetch_statements("TEST", annual=True, ttm=False, asreported=False)
    sc.fetch_statements("TEST", annual=True, asreported=False)
    sc.fetch_statements("TEST", annual=True)
    sc.fetch_statements("TEST")

    assert calls == [("TEST", True, False, False)]


def test_a_genuinely_different_dimension_is_still_its_own_entry():
    """The collapse above must not swallow a real difference."""
    sh_statements._cached_statements.clear()
    calls = []

    def spy(ticker, *, annual=True, ttm=False, asreported=False):
        calls.append(sc.dimension_for(annual=annual, ttm=ttm, asreported=asreported))
        return sc.StatementSet(
            ticker=ticker, permaticker=1, name=ticker, currency="USD", template="",
            dimension=calls[-1], period="FY", ttm=ttm, asreported=asreported, frames={},
        )

    original = sh_statements._fetch_statements
    sh_statements._fetch_statements = spy
    try:
        sc.fetch_statements("TEST", annual=True)
        sc.fetch_statements("TEST", annual=False)
        sc.fetch_statements("TEST", annual=False, ttm=True)
        sc.fetch_statements("TEST", annual=False, asreported=True)
    finally:
        sh_statements._fetch_statements = original
        sh_statements._cached_statements.clear()

    assert calls == ["MRY", "MRQ", "MRT", "ARQ"]


# ---------------------------------------------------------------------------
# Views over a StatementSet
# ---------------------------------------------------------------------------


def test_share_frames_are_projected_off_the_statements(
    monkeypatch, statements_payload, tickers_response
):
    _stub_tables(monkeypatch, sf1=statements_payload, tickers=tickers_response)
    statements = sh_statements._fetch_statements("TEST")

    common = sc.common_shares_frame(statements)
    assert list(common.columns) == ["endDate", "value"]
    assert common["value"].iloc[-1] == 1000.0

    weighted = sc.weighted_shares_frame(statements)
    assert weighted["basic"].iloc[-1] == 1000.0
    assert weighted["diluted"].iloc[-1] == 1010.0
    assert weighted["endDate"].iloc[-1] == pd.Timestamp("2025-09-30")


def test_share_factor_is_applied_to_the_point_in_time_count(
    monkeypatch, annual_rows, tickers_response
):
    """ADRs and multi-class structures need the factor Sharadar applies itself."""
    row = list(annual_rows[0])
    row[SF1_COLUMNS.index("sharefactor")] = 0.5
    _stub_tables(monkeypatch, sf1=sf1_payload([row]), tickers=tickers_response)
    common = sc.common_shares_frame(sh_statements._fetch_statements("TEST"))
    assert common["value"].iloc[-1] == 500.0


def test_filings_frame_labels_annual_rows_as_10k(
    monkeypatch, statements_payload, tickers_response
):
    _stub_tables(monkeypatch, sf1=statements_payload, tickers=tickers_response)
    filings = sc.filings_frame(sh_statements._fetch_statements("TEST"))
    assert filings["filingType"].tolist() == ["10-K", "10-K", "10-K"]
    assert filings["sourceLink"].iloc[-1].startswith("https://www.sec.gov/")


def test_fiscal_year_end_month_is_read_off_the_report_period(
    monkeypatch, statements_payload, tickers_response
):
    _stub_tables(monkeypatch, sf1=statements_payload, tickers=tickers_response)
    assert sc.fiscal_year_end_month(sh_statements._fetch_statements("TEST")) == 9


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def test_error_never_carries_the_api_key(monkeypatch):
    """Nasdaq only takes the key as a query param, so no surface may echo params."""
    secret = "SUPERSECRETKEY123"
    monkeypatch.setattr(sh_transport, "_api_key", lambda: secret)
    monkeypatch.setattr(sh_transport.requests, "get", lambda *a, **k: _FakeResponse(403))

    with pytest.raises(sc.SharadarError) as caught:
        sh_transport._get("SF1", {"ticker": "AAPL"})

    rendered = f"{caught.value} {caught.value.args!r} {caught.value.__dict__!r}"
    assert secret not in rendered
    assert "api_key" not in rendered
    assert caught.value.status == 403


@pytest.mark.parametrize("status", [400, 403, 404])
def test_each_status_gets_a_distinct_message(monkeypatch, status):
    monkeypatch.setattr(sh_transport, "_api_key", lambda: "k")
    monkeypatch.setattr(sh_transport.requests, "get", lambda *a, **k: _FakeResponse(status))
    with pytest.raises(sc.SharadarError) as caught:
        sh_transport._get("SF1", {})
    assert str(status) in caught.value.message


def test_rate_limit_is_retried_then_raised(monkeypatch):
    calls = {"n": 0}

    def _get(*args, **kwargs):
        calls["n"] += 1
        return _FakeResponse(429)

    monkeypatch.setattr(sh_transport, "_api_key", lambda: "k")
    monkeypatch.setattr(sh_transport.requests, "get", _get)
    monkeypatch.setattr(sh_transport.time, "sleep", lambda *_: None)

    with pytest.raises(sc.SharadarError) as caught:
        sh_transport._get("SF1", {})
    assert calls["n"] == sh_transport._MAX_ATTEMPTS
    assert caught.value.status == 429


def test_rate_limit_recovers_on_retry(monkeypatch):
    responses = [_FakeResponse(429), _FakeResponse(200, {"ok": True})]
    monkeypatch.setattr(sh_transport, "_api_key", lambda: "k")
    monkeypatch.setattr(sh_transport.requests, "get", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(sh_transport.time, "sleep", lambda *_: None)
    assert sh_transport._get("SF1", {}) == {"ok": True}


def test_missing_key_raises_before_any_request(monkeypatch):
    # conftest points the secrets file at nothing; the environment is ours to clear.
    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)

    def _fail(*args, **kwargs):
        raise AssertionError("a request was made without an API key")

    monkeypatch.setattr(sh_transport.requests, "get", _fail)
    with pytest.raises(sc.SharadarError, match="SHARADAR_API_KEY"):
        sh_transport._get("SF1", {})


def test_the_environment_supplies_the_key_where_there_is_no_secrets_file(monkeypatch):
    """How Cloud Run hands a Secret Manager secret to the API service."""
    monkeypatch.setenv("SHARADAR_API_KEY", "from-env")
    assert sh_transport._api_key() == "from-env"


def test_the_secrets_file_supplies_the_key_locally(monkeypatch, tmp_path):
    """Local development: ``.streamlit/secrets.toml``, read when the environment is silent."""
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('SHARADAR_API_KEY = "from-file"\n')
    monkeypatch.setattr(settings, "SECRETS_FILE", secrets)
    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    assert sh_transport._api_key() == "from-file"
    monkeypatch.setenv("SHARADAR_API_KEY", "from-env")
    assert sh_transport._api_key() == "from-env"  # the environment wins


def test_a_malformed_secrets_file_is_named_and_never_quoted(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('SHARADAR_API_KEY = "SUPERSECRETKEY123\n')  # unterminated
    monkeypatch.setattr(settings, "SECRETS_FILE", secrets)
    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    with pytest.raises(sc.SharadarError) as caught:
        sh_transport._api_key()
    rendered = f"{caught.value} {caught.value.args!r} {caught.value.__dict__!r}"
    assert str(secrets) in rendered
    assert "SUPERSECRETKEY123" not in rendered
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_pagination_follows_the_cursor_until_exhausted(monkeypatch, annual_rows):
    """Responses truncate at 10,000 rows; silently dropping the rest loses history."""
    pages = [
        sf1_payload(annual_rows[:1], cursor="abc"),
        sf1_payload(annual_rows[1:], cursor=None),
    ]
    seen: list[dict] = []

    def _get(table, params, ticker=None):
        seen.append(dict(params))
        return pages[len(seen) - 1]

    monkeypatch.setattr(sh_transport, "_get", _get)
    frame = sh_transport._request_all("SF1", {"ticker": "TEST"})

    assert len(frame) == 3
    assert len(seen) == 2
    assert "qopts.cursor_id" not in seen[0]
    assert seen[1]["qopts.cursor_id"] == "abc"


def test_pagination_stops_at_the_runaway_guard(monkeypatch, annual_rows):
    """A server that never clears the cursor must not loop forever."""
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: sf1_payload(annual_rows[:1], cursor="always")
    )
    frame = sh_transport._request_all("SF1", {"ticker": "TEST"})
    assert len(frame) == sh_transport._MAX_PAGES


def test_unknown_ticker_raises_rather_than_returning_empty(monkeypatch, tickers_response):
    _stub_tables(monkeypatch, sf1=sf1_payload([]), tickers=tickers_response)
    with pytest.raises(sc.TickerNotFoundError):
        sh_statements._fetch_statements("NOPE")


# ---------------------------------------------------------------------------
# Price-table routing
# ---------------------------------------------------------------------------


def test_tickers_are_routed_to_sep_or_sfp(monkeypatch):
    """Nothing in a ticker string says whether it is an equity or a fund."""
    payload = {
        "datatable": {
            "columns": [{"name": "table"}, {"name": "ticker"}],
            "data": [["SF1", "AAPL"], ["SEP", "AAPL"], ["SFP", "SPY"]],
        },
        "meta": {"next_cursor_id": None},
    }
    monkeypatch.setattr(sh_transport, "_get", lambda *a, **k: payload)
    routes = sh_reference._fetch_price_tables(("AAPL", "SPY"))
    assert routes == {"AAPL": "SEP", "SPY": "SFP"}


def _stub_tables(monkeypatch, *, sf1: dict, tickers: dict) -> None:
    """Routes ``_get`` to the right canned payload by table name."""
    def _get(table, params, ticker=None):
        if table == "TICKERS":
            return tickers
        if table == "SF1":
            return sf1
        raise AssertionError(f"unexpected table {table}")

    monkeypatch.setattr(sh_transport, "_get", _get)


def test_a_price_history_sends_its_tickers_requests_together(monkeypatch):
    """Each stubbed request waits until every one has started. In sequence the
    first would wait alone and the barrier would time out."""
    import threading

    tickers = ("QQQ", "GLD", "XLE", "SPY")
    started = threading.Barrier(len(tickers), timeout=5)

    def fetch(ticker, *, table, start=None, end=None):
        started.wait()
        dates = pd.bdate_range("2026-01-05", periods=3)
        return pd.DataFrame({"date": dates, "closeadj": [1.0, 2.0, 3.0]})

    monkeypatch.setattr(
        sh_reference, "fetch_price_tables", lambda wanted: dict.fromkeys(wanted, "SFP")
    )
    monkeypatch.setattr(sh_prices, "_fetch_price_rows", fetch)
    frame = sc.fetch_price_history(tickers, "2026-01-01", "2026-01-31")
    assert list(frame.columns) == list(tickers)
