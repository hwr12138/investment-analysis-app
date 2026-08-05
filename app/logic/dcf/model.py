"""Assembly: fetch → resolve → normalise → validate → value.

The page calls :func:`load_company` once per ticker and :func:`run_model` on
every assumption change. Only the first touches the network, so moving a slider
never triggers a refetch.

Everything here is orchestration. The arithmetic lives in :mod:`valuation` and
:mod:`wacc`, which stay pure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.data import sharadar_client
from app.data.sharadar_client import StatementSet
from app.logic.dcf import financials as fin
from app.logic.dcf import reverse
from app.logic.dcf import validation as val
from app.logic.dcf import valuation as dcf
from app.logic.dcf import wacc as wacc_lib
from app.logic.dcf.schema import SchemaMap, resolve_schema
from app.provenance import Sourced

#: Years of daily prices pulled for the beta regression.
BETA_HISTORY_YEARS = 5

#: Benchmark for the beta regression. Sharadar carries no index levels, so this
#: is an ETF from the fund-prices table rather than an index level.
DEFAULT_BENCHMARK = "SPY"

#: Terminal value method driving the headline figure and the equity bridge.
#: Both methods always compute and both are always displayed; this only decides
#: which one the single headline number comes from.
DEFAULT_HEADLINE_METHOD = "exit_multiple"


@dataclass
class BasePeriod:
    """The period the forecast starts from.

    Flows come from the trailing twelve months where Sharadar has four quarters,
    because a valuation struck today should not project off figures that stop at
    the last fiscal year end. Stock items always come from the most recent
    balance sheet and are labelled with its date.
    """

    label: str
    flows: pd.Series
    stocks: pd.Series
    stocks_as_of: date | None
    is_ttm: bool
    detail: str


@dataclass
class CompanyDataset:
    """Everything fetched for one ticker. Built once, reused across reruns."""

    ticker: str
    name: str
    currency: str
    template: str
    permaticker: int
    listing: dict | None
    schema: SchemaMap

    annual: pd.DataFrame
    annual_meta: pd.DataFrame
    quarterly: pd.DataFrame
    quarterly_meta: pd.DataFrame
    #: Every quarter Sharadar returned, used by the quarters-sum check.
    quarterly_all: pd.DataFrame
    ttm: pd.Series | None
    ttm_label: str | None
    manual_ttm: pd.Series | None

    prices: pd.DataFrame
    common_shares: pd.DataFrame
    weighted_shares_annual: pd.DataFrame
    weighted_shares_ttm: pd.DataFrame
    filings: pd.DataFrame

    statements_annual: StatementSet
    asreported: bool
    fiscal_year_end_month: int | None
    capex_result: fin.CapexResult
    ratios: pd.DataFrame

    # Quote and share counts, each with its own as-of date.
    last_price: float | None = None
    price_date: date | None = None
    reported_market_cap: float | None = None
    shares_point_in_time: float | None = None
    shares_point_in_time_date: date | None = None
    #: A second, independent read on the same point-in-time concept, backed out
    #: of the daily market cap. It disagrees with the filed count whenever the
    #: share factor is not 1, which is exactly when the disagreement is worth
    #: seeing.
    shares_from_prices: float | None = None
    shares_weighted_diluted: float | None = None
    shares_weighted_basic: float | None = None
    shares_weighted_label: str = ""
    shares_eps_implied: float | None = None

    @property
    def sector_name(self) -> str | None:
        return (self.listing or {}).get("sector")

    @property
    def industry_name(self) -> str | None:
        return (self.listing or {}).get("industry")

    def filing_link(self, fiscal_year: int) -> str | None:
        """Source filing for a fiscal year, preferring the annual filing index."""
        if not self.filings.empty:
            hits = self.filings[
                (self.filings["fyear"] == fiscal_year) & (self.filings["period"] == "FY")
            ]
            if not hits.empty:
                return str(hits.iloc[-1]["sourceLink"])
        if fiscal_year in self.annual_meta.index:
            link = self.annual_meta.loc[fiscal_year, "source_link"]
            if isinstance(link, str) and link:
                return link
        return None


def _reject_financial_sector(
    ticker: str, statements: StatementSet, listing: dict | None
) -> None:
    """Banks and insurers get rejected before anything is computed.

    The SIC code is authoritative — division H, codes 6000-6999, is Finance,
    Insurance and Real Estate. Sharadar's own ``sector`` label is a vendor
    classification that can be null, so it is only a fallback. The two agree on
    JPM (SIC 6021, sector "Financial Services").
    """
    sector = (listing or {}).get("sector") or statements.sector or ""

    if statements.is_financial_template:
        raise val.UnsupportedSectorError(ticker, sector, statements.template)


def _series_value(series: pd.Series | None, key: str) -> float | None:
    if series is None or key not in series.index:
        return None
    value = series[key]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def load_company(ticker: str, *, asreported: bool = False) -> CompanyDataset:
    """Fetches and normalises everything for one ticker.

    Raises :class:`~app.logic.dcf.validation.UnsupportedSectorError` for banks
    and insurers, :class:`~app.logic.dcf.schema.SchemaResolutionError` when a
    required concept cannot be resolved, and
    :class:`~app.data.sharadar_client.TickerNotFoundError` when Sharadar has no
    data — all BLOCKING, all before a single number is displayed.

    ``asreported`` selects Sharadar's ``AR`` dimensions, which exclude
    restatements and index to the SEC filing date. The default uses ``MR``,
    which includes restatements and gives the consistent time series that ratio
    derivation needs.
    """
    ticker = ticker.strip().upper()
    listing = sharadar_client.fetch_ticker_meta(ticker)

    annual_statements = sharadar_client.fetch_statements(
        ticker, annual=True, asreported=asreported
    )
    _reject_financial_sector(ticker, annual_statements, listing)

    schema = resolve_schema(
        annual_statements.frames, ticker, annual_statements.template
    )
    annual, annual_meta = fin.build_concept_table(annual_statements, schema)

    # Sharadar publishes quarterly figures only for US domestic filers — foreign
    # issuers and ADRs have annual and trailing-twelve-month dimensions but no
    # quarterly one. That is a supported company, not a failure, so the model
    # falls back to annual-only and the base period says so.
    quarterly_statements: StatementSet | None = None
    quarterly_all = pd.DataFrame()
    quarterly_meta_all = pd.DataFrame()
    try:
        quarterly_statements = sharadar_client.fetch_statements(
            ticker, annual=False, asreported=asreported
        )
        quarterly_all, quarterly_meta_all = fin.build_concept_table(
            quarterly_statements, schema
        )
    except sharadar_client.SharadarError:
        quarterly_statements = None

    # Eight quarters is enough for a TTM plus a prior-year comparison, and is
    # what the page displays. The full series is kept for the quarters-sum
    # check, which gets far stronger coverage from it at no extra network cost.
    quarterly = quarterly_all.tail(8)
    quarterly_meta = quarterly_meta_all.tail(8)

    ttm_series: pd.Series | None = None
    ttm_label: str | None = None
    ttm_statements: StatementSet | None = None
    try:
        ttm_statements = sharadar_client.fetch_statements(
            ticker, annual=False, ttm=True, asreported=asreported
        )
        ttm_frame, _ = fin.build_concept_table(ttm_statements, schema)
        if not ttm_frame.empty:
            ttm_series = ttm_frame.iloc[-1]
            ttm_label = str(ttm_frame.index[-1])
    except sharadar_client.SharadarError:
        ttm_series, ttm_label, ttm_statements = None, None, None

    manual = fin.manual_ttm(quarterly, fin.FLOW_KEYS)

    prices = sharadar_client.fetch_prices(
        ticker,
        start=(date.today() - timedelta(days=365 * BETA_HISTORY_YEARS + 10)).isoformat(),
    )

    # Share counts and the filing index ride along on the SF1 rows rather than
    # coming from dedicated endpoints, so these are projections of frames
    # already fetched. The trailing-twelve-month weighted counts come from the
    # TTM statements where we have them, since those are the counts that pair
    # with the TTM earnings the base period uses.
    weighted_ttm = (
        sharadar_client.weighted_shares_frame(ttm_statements)
        if ttm_statements is not None
        else pd.DataFrame(columns=["endDate", "fyear", "period", "basic", "diluted"])
    )

    dataset = CompanyDataset(
        ticker=ticker,
        name=annual_statements.name or ticker,
        currency=annual_statements.currency,
        template=annual_statements.template,
        permaticker=annual_statements.permaticker,
        listing=listing,
        schema=schema,
        annual=annual,
        annual_meta=annual_meta,
        quarterly=quarterly,
        quarterly_meta=quarterly_meta,
        quarterly_all=quarterly_all,
        ttm=ttm_series,
        ttm_label=ttm_label,
        manual_ttm=manual,
        prices=prices,
        # Point-in-time shares come off the newest filing we have, which is the
        # latest quarter where the company files one.
        common_shares=sharadar_client.common_shares_frame(
            quarterly_statements if quarterly_statements is not None else annual_statements
        ),
        weighted_shares_annual=sharadar_client.weighted_shares_frame(annual_statements),
        weighted_shares_ttm=weighted_ttm,
        filings=sharadar_client.filings_frame(annual_statements),
        statements_annual=annual_statements,
        asreported=asreported,
        fiscal_year_end_month=sharadar_client.fiscal_year_end_month(annual_statements),
        capex_result=fin.capex(annual),
        ratios=fin.driver_ratios(annual),
    )

    _attach_quote_and_shares(dataset)
    return dataset


def _attach_quote_and_shares(dataset: CompanyDataset) -> None:
    """Reads the current quote and all three share-count sources onto the dataset."""
    prices = dataset.prices
    if not prices.empty:
        latest = prices.iloc[-1]
        dataset.price_date = pd.Timestamp(latest["Date"]).date()
        for column, target in (
            ("Last Closing Price", "last_price"),
            ("Market-Cap", "reported_market_cap"),
            ("Common Shares Outstanding", "shares_from_prices"),
        ):
            if column in prices.columns and not pd.isna(latest.get(column)):
                setattr(dataset, target, float(latest[column]))

    if not dataset.common_shares.empty:
        row = dataset.common_shares.iloc[-1]
        dataset.shares_point_in_time = float(row["value"])
        dataset.shares_point_in_time_date = pd.Timestamp(row["endDate"]).date()

    if not dataset.weighted_shares_ttm.empty:
        row = dataset.weighted_shares_ttm.iloc[-1]
        dataset.shares_weighted_diluted = (
            float(row["diluted"]) if not pd.isna(row.get("diluted")) else None
        )
        dataset.shares_weighted_basic = (
            float(row["basic"]) if not pd.isna(row.get("basic")) else None
        )
        dataset.shares_weighted_label = f"TTM to FY{row['fyear']} {row['period']}"
    elif not dataset.weighted_shares_annual.empty:
        row = dataset.weighted_shares_annual.iloc[-1]
        dataset.shares_weighted_diluted = (
            float(row["diluted"]) if not pd.isna(row.get("diluted")) else None
        )
        dataset.shares_weighted_basic = (
            float(row["basic"]) if not pd.isna(row.get("basic")) else None
        )
        dataset.shares_weighted_label = f"FY{row['fyear']}"

    # A third, independent read on the share count, used only as a cross-check.
    source = dataset.ttm if dataset.ttm is not None else (
        dataset.annual.iloc[-1] if not dataset.annual.empty else None
    )
    net_income = _series_value(source, "net_income")
    eps_diluted = _series_value(source, "eps_diluted")
    if net_income is not None and eps_diluted:
        dataset.shares_eps_implied = net_income / eps_diluted


def base_period(dataset: CompanyDataset) -> BasePeriod:
    """Picks the period the forecast starts from.

    Prefers the API's trailing twelve months; falls back to the latest fiscal
    year and says so.
    """
    stocks = dataset.quarterly.iloc[-1] if not dataset.quarterly.empty else (
        dataset.annual.iloc[-1] if not dataset.annual.empty else pd.Series(dtype="float64")
    )
    stocks_as_of = None
    if not dataset.quarterly_meta.empty:
        raw = dataset.quarterly_meta["report_date"].iloc[-1]
        stocks_as_of = None if pd.isna(raw) else pd.Timestamp(raw).date()
    elif not dataset.annual_meta.empty:
        raw = dataset.annual_meta["report_date"].iloc[-1]
        stocks_as_of = None if pd.isna(raw) else pd.Timestamp(raw).date()

    if dataset.ttm is not None:
        return BasePeriod(
            label=f"TTM to {dataset.ttm_label}",
            flows=dataset.ttm,
            stocks=stocks,
            stocks_as_of=stocks_as_of,
            is_ttm=True,
            detail="Trailing twelve months, as published by Sharadar under its "
                   "trailing-twelve-month dimension.",
        )

    latest = dataset.annual.iloc[-1] if not dataset.annual.empty else pd.Series(dtype="float64")
    label = str(dataset.annual.index[-1]) if not dataset.annual.empty else "n/a"
    return BasePeriod(
        label=f"FY{label}",
        flows=latest,
        stocks=stocks,
        stocks_as_of=stocks_as_of,
        is_ttm=False,
        detail="Sharadar returned no trailing-twelve-month period, so the last "
               "reported fiscal year is the base.",
    )


# ---------------------------------------------------------------------------
# Base-period FCFF, both ways
# ---------------------------------------------------------------------------


def base_period_delta_nwc(dataset: CompanyDataset, base: BasePeriod) -> tuple[float | None, str]:
    """Change in net working capital over the same span as the base period.

    A trailing-twelve-month cash flow has to be paired with a
    trailing-twelve-month working capital movement. Differencing the last two
    *annual* balance sheets against a TTM cash flow measures two different
    spans and inflates the gap between the two FCFF builds — which then reads as
    a data problem rather than as the wiring mistake it is.

    Returns the change and a description of the span it covers.
    """
    if base.is_ttm and len(dataset.quarterly) >= 5:
        nwc = fin.net_working_capital(dataset.quarterly).dropna()
        if len(nwc) >= 5:
            change = float(nwc.iloc[-1] - nwc.iloc[-5])
            return change, (
                f"Trailing twelve months: net working capital at {nwc.index[-1]} "
                f"less {nwc.index[-5]}"
            )

    annual = fin.delta_nwc(dataset.annual).dropna()
    if annual.empty:
        return None, "No two consecutive balance sheets available."
    span = (
        f"Fiscal year {annual.index[-1]}"
        + (f" less {dataset.annual.index[-2]}" if len(dataset.annual) > 1 else "")
    )
    if base.is_ttm:
        span += " — annual basis, because fewer than five quarters are available"
    return float(annual.iloc[-1]), span


@dataclass
class BaseFcff:
    method_a: float | None
    method_b: float | None
    inputs: dict[str, float | None]
    notes: list[str] = field(default_factory=list)

    @property
    def divergence(self) -> float | None:
        if self.method_a is None or self.method_b is None:
            return None
        return dcf.divergence(self.method_a, self.method_b)


def base_period_fcff(
    dataset: CompanyDataset,
    base: BasePeriod,
    *,
    tax_rate: float,
    interest_basis: str = "net",
    delta_nwc: float | None = None,
) -> BaseFcff:
    """Computes FCFF for the base period both ways, keeping every input.

    Either method returns ``None`` when one of its inputs is not reported —
    Apple, for instance, has no interest expense line at all, which makes method
    A's add-back uncomputable rather than zero.
    """
    flows = base.flows
    notes: list[str] = []

    cfo = _series_value(flows, "cfo")
    ebit = _series_value(flows, "ebit")
    dna = _series_value(flows, "dna")

    capex_series = fin.capex(pd.DataFrame([flows]))
    capex_value = (
        float(capex_series.values.iloc[0])
        if not capex_series.values.empty and not pd.isna(capex_series.values.iloc[0])
        else None
    )
    if capex_value is None:
        notes.append("CapEx is not reported for the base period.")

    interest = fin.interest_expense(pd.DataFrame([flows]), prefer=interest_basis)
    interest_value = (
        float(interest.iloc[0]) if not interest.empty and not pd.isna(interest.iloc[0]) else None
    )
    if interest_value is None:
        notes.append(
            "Sharadar reports no interest expense for this company, so method A's "
            "after-tax interest add-back cannot be computed."
        )
    elif interest_value == 0.0:
        notes.append(
            "Sharadar reports interest expense as 0 for this company. That is how "
            "it records a line item absent from the filing, so it may mean not "
            "disclosed rather than nil — in which case method A understates FCFF "
            "by the after-tax interest."
        )

    if delta_nwc is None:
        notes.append("No change in net working capital available for method B.")

    method_a = None
    if all(v is not None for v in (cfo, interest_value, capex_value)):
        method_a = dcf.fcff_from_cash_flow(cfo, interest_value, tax_rate, capex_value)

    method_b = None
    if all(v is not None for v in (ebit, dna, capex_value, delta_nwc)):
        method_b = dcf.fcff_from_income_statement(ebit, tax_rate, dna, capex_value, delta_nwc)

    return BaseFcff(
        method_a=method_a,
        method_b=method_b,
        inputs={
            "CFO": cfo,
            "EBIT": ebit,
            "D&A": dna,
            "CapEx": capex_value,
            "Interest expense": interest_value,
            "ΔNWC": delta_nwc,
            "Tax rate": tax_rate,
        },
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------


@dataclass
class BetaOutcome:
    result: wacc_lib.BetaResult | None
    source: str
    error: str | None = None
    price_source_note: str = ""


def regress_company_beta(
    dataset: CompanyDataset,
    *,
    frequency: str = "monthly",
    blume: bool = False,
    index_ticker: str = DEFAULT_BENCHMARK,
    years: int = BETA_HISTORY_YEARS,
) -> BetaOutcome:
    """Regresses the company's returns on the benchmark's.

    Both sides are Sharadar total-return closes — the company from SEP, the
    benchmark ETF from SFP — so they share one adjustment convention and one
    trading calendar, and no date-set reconciliation is needed.

    The benchmark is an ETF rather than an index level because Sharadar carries
    no index levels. SPY's adjusted close includes reinvested distributions,
    where the S&P 500 index level does not, so betas differ slightly from a
    regression against the bare index — the dividend-inclusive pairing is the
    more defensible one, since the company side includes dividends too.
    """
    if dataset.prices.empty or "Adjusted Closing Price" not in dataset.prices.columns:
        return BetaOutcome(None, "Sharadar", "No adjusted price series available.")

    start = (date.today() - timedelta(days=365 * years + 10)).isoformat()
    end = (date.today() + timedelta(days=1)).isoformat()

    company = (
        dataset.prices.set_index("Date")["Adjusted Closing Price"].dropna().sort_index()
    )
    company = company[company.index >= pd.Timestamp(start)]
    index_prices = sharadar_client.fetch_index_prices(index_ticker, start, end)

    source = f"Sharadar adjusted closes (SEP) vs {index_ticker} (SFP)"
    note = ""

    try:
        result = wacc_lib.regress_beta(
            company,
            index_prices,
            frequency=frequency,
            blume=blume,
            index_ticker=index_ticker,
        )
    except wacc_lib.InsufficientObservationsError as error:
        return BetaOutcome(None, source, str(error), note)
    except Exception as error:  # a malformed series should not take the page down
        return BetaOutcome(None, source, f"Beta regression failed: {error}", note)

    return BetaOutcome(result, source, None, note)


# ---------------------------------------------------------------------------
# Historical EV/EBITDA range, for the exit multiple default
# ---------------------------------------------------------------------------


def historical_ev_ebitda(dataset: CompanyDataset) -> Sourced:
    """The company's own historical EV/EBITDA range, from Sharadar's daily series."""
    column = "EV/EBITDA - ttm"
    source = "Sharadar DAILY (EV/EBITDA)"
    if dataset.prices.empty or column not in dataset.prices.columns:
        return Sourced.unavailable(
            source=source,
            reason="Sharadar returned no EV/EBITDA series for this company.",
        )
    series = pd.to_numeric(dataset.prices[column], errors="coerce").dropna()
    series = series[(series > 0) & (series < 200)]
    if series.empty:
        return Sourced.unavailable(
            source=source,
            reason="Every EV/EBITDA observation was missing or implausible.",
        )
    return Sourced.derived(
        value=float(series.median()),
        source=source,
        formula=(
            f"Median of {len(series):,} daily observations "
            f"(range {series.min():.1f}× – {series.max():.1f}×)"
        ),
    )


# ---------------------------------------------------------------------------
# Model run
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """One complete valuation, both terminal-value methods."""

    projection: dcf.Projection
    terminal_values: dict[str, dcf.TerminalValue]
    bridges: dict[str, dcf.EquityBridge]
    enterprise_values: dict[str, float]
    wacc: wacc_lib.WaccResult
    base: BasePeriod
    base_fcff: BaseFcff
    headline_method: str

    @property
    def headline_bridge(self) -> dcf.EquityBridge:
        return self.bridges[self.headline_method]

    @property
    def headline_terminal(self) -> dcf.TerminalValue:
        return self.terminal_values[self.headline_method]


def run_model(
    dataset: CompanyDataset,
    base: BasePeriod,
    *,
    drivers: dcf.DriverPath,
    wacc_result: wacc_lib.WaccResult,
    terminal_growth: float,
    exit_multiple: float,
    stub: float,
    mid_year: bool,
    shares: float,
    share_basis: str,
    include_short_term_investments: bool,
    operating_cash_floor: float,
    non_operating_assets: float | None,
    base_fcff: BaseFcff,
    headline_method: str = DEFAULT_HEADLINE_METHOD,
    sbc: float | None = None,
    deduct_sbc: bool = False,
) -> ModelResult:
    """Projects, terminates both ways, and bridges each to a value per share."""
    base_revenue = _series_value(base.flows, "revenue")
    if base_revenue is None:
        raise dcf.ProjectionError(
            "The base period has no revenue figure to project from."
        )

    projection = dcf.project(
        base_revenue,
        drivers,
        wacc=wacc_result.wacc,
        stub=stub,
        mid_year=mid_year,
        sbc=sbc,
        deduct_sbc=deduct_sbc,
    )
    terminal_values = dcf.build_terminal_values(
        projection, terminal_growth=terminal_growth, exit_multiple=exit_multiple
    )

    stocks = base.stocks
    total_debt = fin.total_debt(pd.DataFrame([stocks]))
    debt_value = (
        float(total_debt.iloc[0]) if not total_debt.empty and not pd.isna(total_debt.iloc[0])
        else None
    )

    enterprise_values: dict[str, float] = {}
    bridges: dict[str, dcf.EquityBridge] = {}
    for key, terminal in terminal_values.items():
        enterprise_value = projection.pv_explicit + terminal.present_value
        enterprise_values[key] = enterprise_value
        bridges[key] = dcf.build_equity_bridge(
            enterprise_value=enterprise_value,
            total_debt=debt_value,
            minority_interest=_series_value(stocks, "minority_interest"),
            preferred_equity=_series_value(stocks, "preferred_equity"),
            cash=_series_value(stocks, "cash"),
            short_term_investments=_series_value(stocks, "short_term_investments"),
            non_operating_assets=non_operating_assets,
            shares=shares,
            share_basis=share_basis,
            include_short_term_investments=include_short_term_investments,
            operating_cash_floor=operating_cash_floor,
        )

    return ModelResult(
        projection=projection,
        terminal_values=terminal_values,
        bridges=bridges,
        enterprise_values=enterprise_values,
        wacc=wacc_result,
        base=base,
        base_fcff=base_fcff,
        headline_method=headline_method,
    )


# ---------------------------------------------------------------------------
# Reverse model
# ---------------------------------------------------------------------------


def solve_implied_growth(
    dataset: CompanyDataset,
    base: BasePeriod,
    *,
    target_price: float | None,
    drivers: dcf.DriverPath,
    bounds: tuple[float, float] = reverse.GROWTH_BOUNDS,
    grid: tuple[float, ...] = reverse.GROWTH_GRID,
    **run_model_kwargs,
) -> reverse.ReverseDcfResult:
    """What flat revenue growth rate reproduces ``target_price`` per share.

    The forward model asks what the shares are worth given a forecast. This asks
    what forecast the market price already contains, by holding every other
    assumption at the value the forward run used and solving the one remaining
    unknown.

    ``run_model_kwargs`` is a deliberate pass-through: the reverse model has to
    invert *the model on screen*, and forwarding the arguments untouched is what
    stops the two drifting apart when :func:`run_model` gains one.
    """
    if target_price is None or not math.isfinite(target_price) or target_price <= 0:
        return reverse.ReverseDcfResult(
            status=reverse.Status.NO_TARGET,
            target=target_price,
            growth=None,
            bounds=bounds,
            reason=(
                f"No usable market price for {dataset.ticker} — the last close came back "
                f"as {target_price!r}, so there is nothing to reverse-engineer. Nothing "
                f"has been substituted."
            ),
        )

    def value_per_share(growth: float) -> float:
        # `replace`, never `DriverPath.flat`: only revenue growth may move. The
        # other five drivers have to arrive byte-identical to the forward run,
        # or this inverts a different model than the one being displayed.
        path = replace(drivers, revenue_growth=(growth,) * drivers.years)
        return run_model(
            dataset, base, drivers=path, **run_model_kwargs
        ).headline_bridge.value_per_share

    return reverse.solve_growth(
        value_per_share,
        float(target_price),
        bounds=bounds,
        grid=grid,
        # Probe where the forward model has just succeeded, so "not evaluable"
        # means the model is broken rather than the probe being unlucky.
        probe_at=drivers.revenue_growth[0],
    )


def _zero_filled_base_items(
    dataset: CompanyDataset, base: BasePeriod
) -> tuple[tuple[str, str], ...]:
    """Base-period line items that are exactly 0 and might mean "not disclosed".

    Sharadar writes 0, not null, for a line item absent from a filing, so the
    two are indistinguishable in the data. Rather than guess, the ones that
    could plausibly be either are collected here and shown to the user as
    ``(concept label, resolved column)`` pairs.
    """
    from app.logic.dcf.schema import CONCEPTS_BY_KEY

    found: list[tuple[str, str]] = []
    for series in (base.flows, base.stocks):
        if series is None:
            continue
        for key, value in series.items():
            column = dataset.schema.column(str(key))
            if column is None or not sharadar_client.is_zero_filled(column):
                continue
            if value is None or pd.isna(value) or float(value) != 0.0:
                continue
            concept = CONCEPTS_BY_KEY.get(str(key))
            label = concept.label if concept else str(key)
            if (label, column) not in found:
                found.append((label, column))
    return tuple(found)


def build_validation_context(
    dataset: CompanyDataset,
    *,
    base: BasePeriod,
    base_fcff: BaseFcff | None = None,
    valuation_date: date | None = None,
    effective_tax_rate: float | None = None,
    beta: BetaOutcome | None = None,
    equity_risk_premium: float | None = None,
    result: ModelResult | None = None,
    terminal_growth: float | None = None,
    risk_free_rate: float | None = None,
    sbc: float | None = None,
) -> val.ValidationContext:
    """Packs everything the checks need into one context."""
    annual = dataset.annual
    bs_delta = fin.delta_nwc(annual)

    context = val.ValidationContext(
        ticker=dataset.ticker,
        company=dataset.name,
        annual=annual,
        annual_meta=dataset.annual_meta,
        quarterly=dataset.quarterly_all,
        api_ttm=dataset.ttm,
        manual_ttm=dataset.manual_ttm,
        currencies_seen=(dataset.currency,) if dataset.currency else (),
        asreported_flags=(dataset.asreported,),
        price_date=dataset.price_date,
        last_price=dataset.last_price,
        reported_market_cap=dataset.reported_market_cap,
        valuation_date=valuation_date or date.today(),
        shares_point_in_time=dataset.shares_point_in_time,
        shares_point_in_time_date=dataset.shares_point_in_time_date,
        shares_from_prices=dataset.shares_from_prices,
        shares_weighted_diluted=dataset.shares_weighted_diluted,
        shares_eps_implied=dataset.shares_eps_implied,
        effective_tax_rate=effective_tax_rate,
        equity_risk_premium=equity_risk_premium,
        terminal_growth=terminal_growth,
        risk_free_rate=risk_free_rate,
        sbc=sbc,
        latest_revenue=_series_value(base.flows, "revenue"),
        balance_sheet_delta_nwc=(
            float(bs_delta.iloc[-1]) if len(bs_delta) and not pd.isna(bs_delta.iloc[-1]) else None
        ),
        zero_filled=_zero_filled_base_items(dataset, base),
    )

    if base_fcff is not None:
        context.fcff_method_a = base_fcff.method_a
        context.fcff_method_b = base_fcff.method_b

    if beta is not None and beta.result is not None:
        context.beta_r_squared = beta.result.r_squared
        context.beta_observations = beta.result.observations

    if result is not None:
        context.debt_weight = result.wacc.debt_weight
        context.long_term_debt = _series_value(base.stocks, "long_term_debt")
        terminal = result.headline_terminal
        enterprise_value = result.enterprise_values[result.headline_method]
        context.terminal_value_share = terminal.share_of(enterprise_value)

    return context
