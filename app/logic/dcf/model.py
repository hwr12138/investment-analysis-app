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

from app.data import market_data, sharadar_client
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
#:
#: Gordon growth — the owner's choice, reversing round 2's exit multiple: the
#: headline should rest on the company's own cash flows, not on the multiple it
#: traded at while it grew faster than the forecast lets it grow by year N. Its
#: terminal year earns the company's own return on new capital, never below the
#: WACC (``financials.historical_roic``); the exit multiple is always computed
#: beside it as the market's cross-check, and the checks flag when the two part
#: company.
DEFAULT_HEADLINE_METHOD = "gordon"

#: The DCF tab's untouched assumption set, which ``app.logic.dcf.inputs``
#: assembles for the API's panel and for the summary chart's line of past
#: values. Conventions rather than measurements: each reads as an estimate on
#: the tab until someone changes it.
DEFAULT_FORECAST_YEARS = 10
DEFAULT_GROWTH_HOLD = 3
DEFAULT_TERMINAL_GROWTH = 0.025
DEFAULT_MARGINAL_TAX = 0.25
#: The exit multiple when the company has no EV/EBITDA history of its own — a
#: placeholder, which never reaches the Gordon headline.
PLACEHOLDER_EXIT_MULTIPLE = 10.0
DEFAULT_BETA_FREQUENCY = "monthly"
DEFAULT_BLUME = True


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
    #: The last day the flows cover — what the forecast is anchored to. See
    #: :func:`~app.logic.dcf.valuation.elapsed_years`.
    flows_as_of: date | None = None
    #: The currency ``flows`` and ``stocks`` are in. Always the price's: a
    #: foreign reporter's base is converted at spot on the way in.
    currency: str = ""
    #: Spot units of ``currency`` per unit of the reporting currency; 1.0 when
    #: no conversion was needed.
    fx_rate: float = 1.0

    def converted(self, dataset: "CompanyDataset") -> "BasePeriod":
        """This base period in the price's currency, at the dataset's spot rate.

        Base revenue, SBC, every balance-sheet item the bridge and the WACC read,
        and so both base-period FCFF builds — everything that meets a dollar
        figure — pass through here. Ratios need no conversion, and the history
        tables stay in the reporting currency.
        """
        rate = dataset.fx_rate_value
        return replace(
            self,
            flows=_to_valuation_currency(self.flows, rate),
            stocks=_to_valuation_currency(self.stocks, rate),
            currency=dataset.valuation_currency,
            fx_rate=rate,
            detail=self.detail + (
                f" Converted from {dataset.currency} at a spot rate of {rate:.6g} USD per "
                f"{dataset.currency}."
                if dataset.is_converted else ""
            ),
        )


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
    #: Every trailing-twelve-month period, for the historical exit multiple.
    ttm_history: pd.DataFrame
    ttm_history_meta: pd.DataFrame

    prices: pd.DataFrame
    common_shares: pd.DataFrame
    weighted_shares_annual: pd.DataFrame
    weighted_shares_ttm: pd.DataFrame
    filings: pd.DataFrame

    statements_annual: StatementSet
    asreported: bool
    #: The ``asreported`` flag of every statement set actually fetched — annual,
    #: quarterly and TTM — so the basis check compares real flags, not one.
    statement_bases: tuple[bool, ...]
    #: ADR shares per underlying share (Sharadar's ``sharefactor``); 1 for an
    #: ordinary listing. Applied to every share count compared with the price.
    share_factor: float
    fiscal_year_end_month: int | None
    capex_result: fin.CapexResult
    ratios: pd.DataFrame

    #: Spot US dollars per unit of the reporting currency, for a company that
    #: does not report in dollars; ``None`` for one that does.
    fx_rate: Sourced | None = None

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
    def valuation_currency(self) -> str:
        """The currency the valuation is struck in — always the price's."""
        return sharadar_client.PRICE_CURRENCY

    @property
    def is_converted(self) -> bool:
        """True when the statements are in a currency other than the price's."""
        return self.fx_rate is not None

    @property
    def fx_rate_value(self) -> float:
        """The spot rate applied at the valuation boundary; 1.0 when none is needed."""
        return float(self.fx_rate.value) if self.fx_rate is not None else 1.0

    @property
    def foreign_filer(self) -> bool:
        """A non-US filer, whatever it reports in — the ``TICKERS`` category says so."""
        category = str((self.listing or {}).get("category") or "")
        return bool(category) and not category.startswith("Domestic")

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


def _spot_rate(
    ticker: str, statements: StatementSet, as_of: date | None = None
) -> Sourced | None:
    """Spot US dollars per unit of the reporting currency, or ``None`` for a USD reporter.

    The drivers come from the company's own reporting-currency history, so its
    growth and margins carry no FX noise; everything is converted once, at the
    valuation boundary, at this rate (:meth:`BasePeriod.converted`). No rate
    means no valuation: :class:`~app.logic.dcf.validation.UnsupportedCurrencyError`
    is raised with the reason rather than a rate substituted. An empty currency
    is let through as missing data, which the checks report.
    """
    currency = (statements.currency or "").strip().upper()
    if not currency or currency == sharadar_client.PRICE_CURRENCY:
        return None
    rate = (
        market_data.fetch_fx_rate(currency)
        if as_of is None
        else market_data.fetch_fx_rate_on(currency, as_of)
    )
    if not rate.ok:
        raise val.UnsupportedCurrencyError(
            ticker, currency, sharadar_client.PRICE_CURRENCY, rate.detail or ""
        )
    return rate


def _series_value(series: pd.Series | None, key: str) -> float | None:
    if series is None or key not in series.index:
        return None
    value = series[key]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def _filed_by(statements: StatementSet, as_of: date) -> StatementSet:
    """The statements as they stood on ``as_of``: every row filed by then.

    Only meaningful for the ``AR`` dimensions, whose ``datekey`` is the SEC
    filing date; under ``MR`` it is the report period, and a restatement moves
    it — see :func:`load_company`.
    """
    cutoff = pd.Timestamp(as_of)
    return replace(
        statements,
        frames={
            name: frame[frame["Publish Date"] <= cutoff].reset_index(drop=True)
            if "Publish Date" in frame.columns else frame
            for name, frame in statements.frames.items()
        },
    )


def load_company(
    ticker: str, *, asreported: bool = False, as_of: date | None = None
) -> CompanyDataset:
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

    ``as_of`` builds the company as it stood on a past date: only what had been
    filed by then, prices up to then, and that day's exchange rate. It requires
    ``asreported``, because ``AR`` is the one dimension dated by filing — under
    ``MR`` the date is the report period and a restatement moves it, so a
    cut-off would drop every period later restated. Share counts need no
    adjustment: Sharadar restates them for splits even under ``AR`` (NVDA's
    October 2023 filing reads 24.7bn shares, the post-split count), so a past
    value per share is already on the basis of today's split-adjusted prices.
    Every fetch is the same cached request whatever the date, so a date costs
    arithmetic, not a request.
    """
    if as_of is not None and not asreported:
        raise ValueError(
            "A past date needs the as-reported statements: the restated ones are "
            "dated by report period, not by filing."
        )
    ticker = ticker.strip().upper()
    listing = sharadar_client.fetch_ticker_meta(ticker)

    def fetch(**kwargs) -> StatementSet:
        statements = sharadar_client.fetch_statements(ticker, asreported=asreported, **kwargs)
        return statements if as_of is None else _filed_by(statements, as_of)

    annual_statements = fetch(annual=True)
    _reject_financial_sector(ticker, annual_statements, listing)
    fx_rate = _spot_rate(ticker, annual_statements, as_of)

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
        quarterly_statements = fetch(annual=False)
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
    ttm_frame, ttm_meta = pd.DataFrame(), pd.DataFrame()
    try:
        ttm_statements = fetch(annual=False, ttm=True)
        ttm_frame, ttm_meta = fin.build_concept_table(ttm_statements, schema)
        if not ttm_frame.empty:
            ttm_series = ttm_frame.iloc[-1]
            ttm_label = str(ttm_frame.index[-1])
    except sharadar_client.SharadarError:
        ttm_series, ttm_label, ttm_statements = None, None, None
        ttm_frame, ttm_meta = pd.DataFrame(), pd.DataFrame()

    manual = fin.manual_ttm(quarterly, fin.FLOW_KEYS)

    anchor = as_of or date.today()
    prices = sharadar_client.fetch_prices(
        ticker,
        start=(anchor - timedelta(days=365 * BETA_HISTORY_YEARS + 10)).isoformat(),
    )
    if as_of is not None and not prices.empty:
        prices = prices[pd.to_datetime(prices["Date"]) <= pd.Timestamp(as_of)].reset_index(
            drop=True
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
        ttm_history=ttm_frame,
        ttm_history_meta=ttm_meta,
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
        statement_bases=tuple(
            statements.asreported
            for statements in (annual_statements, quarterly_statements, ttm_statements)
            if statements is not None
        ),
        share_factor=sharadar_client.share_factor(annual_statements) or 1.0,
        fx_rate=fx_rate,
        fiscal_year_end_month=sharadar_client.fiscal_year_end_month(annual_statements),
        capex_result=fin.capex(annual),
        ratios=fin.driver_ratios(annual),
    )

    _attach_quote_and_shares(dataset, split_adjusted=as_of is not None)
    return dataset


def _attach_quote_and_shares(dataset: CompanyDataset, *, split_adjusted: bool = False) -> None:
    """Reads the current quote and all three share-count sources onto the dataset.

    The point-in-time count arrives with ``sharefactor`` already applied. The
    weighted counts and EPS are per *underlying* share, so they are multiplied
    by the same factor here — otherwise an ADR's default denominator would be
    in a different unit from its price.

    ``split_adjusted`` quotes the split-adjusted close instead of the traded
    one, for a past date: the share counts are restated for every split since,
    so the traded price before NVDA's ten-for-one split times them overstated
    its market cap tenfold. Today the two closes are the same number.
    """
    prices = dataset.prices
    if not prices.empty:
        latest = prices.iloc[-1]
        dataset.price_date = pd.Timestamp(latest["Date"]).date()
        quote = (
            "Split-Adjusted Close"
            if split_adjusted and "Split-Adjusted Close" in prices.columns
            else "Last Closing Price"
        )
        for column, target in (
            (quote, "last_price"),
            ("Market-Cap", "reported_market_cap"),
            ("Common Shares Outstanding", "shares_from_prices"),
        ):
            if column in prices.columns and not pd.isna(latest.get(column)):
                setattr(dataset, target, float(latest[column]))

    if not dataset.common_shares.empty:
        row = dataset.common_shares.iloc[-1]
        dataset.shares_point_in_time = float(row["value"])
        dataset.shares_point_in_time_date = pd.Timestamp(row["endDate"]).date()

    factor = dataset.share_factor
    if not dataset.weighted_shares_ttm.empty:
        row = dataset.weighted_shares_ttm.iloc[-1]
        dataset.shares_weighted_diluted = (
            float(row["diluted"]) * factor if not pd.isna(row.get("diluted")) else None
        )
        dataset.shares_weighted_basic = (
            float(row["basic"]) * factor if not pd.isna(row.get("basic")) else None
        )
        dataset.shares_weighted_label = f"TTM to FY{row['fyear']} {row['period']}"
    elif not dataset.weighted_shares_annual.empty:
        row = dataset.weighted_shares_annual.iloc[-1]
        dataset.shares_weighted_diluted = (
            float(row["diluted"]) * factor if not pd.isna(row.get("diluted")) else None
        )
        dataset.shares_weighted_basic = (
            float(row["basic"]) * factor if not pd.isna(row.get("basic")) else None
        )
        dataset.shares_weighted_label = f"FY{row['fyear']}"

    # A third, independent read on the share count, used only as a cross-check.
    source = dataset.ttm if dataset.ttm is not None else (
        dataset.annual.iloc[-1] if not dataset.annual.empty else None
    )
    net_income = _series_value(source, "net_income")
    eps_diluted = _series_value(source, "eps_diluted")
    if net_income is not None and eps_diluted:
        dataset.shares_eps_implied = net_income / eps_diluted * factor


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
            flows_as_of=_last_date(dataset.ttm_history_meta),
        ).converted(dataset)

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
        flows_as_of=_last_date(dataset.annual_meta),
    ).converted(dataset)


#: Concepts that are not money and so are never converted.
NON_MONETARY_CONCEPTS = frozenset({"fx_to_usd"})


def _to_valuation_currency(series: pd.Series, rate: float) -> pd.Series:
    """Every monetary line of ``series`` at spot; rates left untouched."""
    if rate == 1.0 or series is None or series.empty:
        return series
    monetary = [key for key in series.index if str(key) not in NON_MONETARY_CONCEPTS]
    converted = series.copy()
    converted[monetary] = pd.to_numeric(series[monetary], errors="coerce") * rate
    return converted


def _last_date(meta: pd.DataFrame) -> date | None:
    """The report period of the newest row, or ``None``."""
    if meta is None or meta.empty or "report_date" not in meta.columns:
        return None
    raw = meta["report_date"].iloc[-1]
    return None if pd.isna(raw) else pd.Timestamp(raw).date()


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
            change = float(nwc.iloc[-1] - nwc.iloc[-5]) * base.fx_rate
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
    return float(annual.iloc[-1]) * base.fx_rate, span


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
    as_of: date | None = None,
) -> BetaOutcome:
    """Regresses the company's returns on the benchmark's.

    The window ends on ``as_of`` — today when it is ``None`` — so a past
    valuation uses the beta that could have been measured then.

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

    anchor = as_of or date.today()
    start = (anchor - timedelta(days=365 * years + 10)).isoformat()
    end = (anchor + timedelta(days=1)).isoformat()

    company = (
        dataset.prices.set_index("Date")["Adjusted Closing Price"].dropna().sort_index()
    )
    company = company[
        (company.index >= pd.Timestamp(start)) & (company.index <= pd.Timestamp(anchor))
    ]
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


def _model_ebitda_history(dataset: CompanyDataset) -> pd.Series:
    """Operating income plus D&A per period, indexed by period end date.

    Trailing-twelve-month periods where Sharadar has them, fiscal years where it
    does not — the same EBITDA the exit multiple is applied to.
    """
    for values, meta in (
        (dataset.ttm_history, dataset.ttm_history_meta),
        (dataset.annual, dataset.annual_meta),
    ):
        if values.empty or meta.empty or not {"ebit", "dna"}.issubset(values.columns):
            continue
        ebitda = values["ebit"] + values["dna"]
        if dataset.is_converted:
            # DAILY enterprise value is in US dollars, so each period's EBITDA is
            # converted at the rate Sharadar published for that period — units
            # of the reporting currency per dollar, hence the division. A period
            # with no published rate drops out rather than borrowing another's.
            rate = values.get("fx_to_usd", pd.Series(np.nan, index=values.index))
            ebitda = ebitda / rate.where(rate > 0)
        ebitda = ebitda.to_numpy()
        dates = pd.to_datetime(meta["report_date"].reindex(values.index), errors="coerce")
        series = pd.Series(ebitda, index=dates.to_numpy()).dropna()
        series = series[series.index.notna()]
        if not series.empty:
            return series.sort_index()
    return pd.Series(dtype="float64")


def historical_ev_ebitda(dataset: CompanyDataset) -> Sourced:
    """The company's own historical EV/EBITDA range, on the model's EBITDA.

    Sharadar's own ``EV/EBITDA`` builds on *its* EBITDA — net income plus tax,
    interest and D&A, so non-operating items included — while the exit multiple
    is applied to operating income plus D&A. Where Sharadar's daily enterprise
    value is available, each day's EV is therefore divided by the model's
    EBITDA for the latest period reported by then, so the multiple is measured
    on the number it is later applied to — for a foreign reporter, in dollars at
    each period's own published rate, since DAILY EV is in dollars. Sharadar's
    ratio is the fallback, and
    says so.
    """
    source = "Sharadar DAILY (enterprise value) ÷ operating income + D&A"
    if not dataset.prices.empty and "Enterprise Value" in dataset.prices.columns:
        ebitda = _model_ebitda_history(dataset)
        daily = dataset.prices[["Date", "Enterprise Value"]].copy()
        daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
        daily["Enterprise Value"] = pd.to_numeric(daily["Enterprise Value"], errors="coerce")
        daily = daily.dropna().sort_values("Date")
        if not ebitda.empty and not daily.empty:
            frame = pd.merge_asof(
                daily,
                ebitda.rename("ebitda").rename_axis("period_end").reset_index(),
                left_on="Date",
                right_on="period_end",
            ).dropna(subset=["ebitda"])
            frame = frame[frame["ebitda"] > 0]
            multiple = frame["Enterprise Value"] / frame["ebitda"]
            multiple = multiple[(multiple > 0) & (multiple < 200)]
            if not multiple.empty:
                return Sourced.derived(
                    value=float(multiple.median()),
                    source=source,
                    formula=(
                        f"Median of {len(multiple):,} daily EV ÷ (operating income + D&A) "
                        f"observations (range {multiple.min():.1f}× – {multiple.max():.1f}×)"
                    ),
                )

    column = "EV/EBITDA - ttm"
    fallback = "Sharadar DAILY (EV/EBITDA)"
    if dataset.prices.empty or column not in dataset.prices.columns:
        return Sourced.unavailable(
            source=fallback,
            reason="Sharadar returned no EV/EBITDA series for this company.",
        )
    series = pd.to_numeric(dataset.prices[column], errors="coerce").dropna()
    series = series[(series > 0) & (series < 200)]
    if series.empty:
        return Sourced.unavailable(
            source=fallback,
            reason="Every EV/EBITDA observation was missing or implausible.",
        )
    return Sourced.derived(
        value=float(series.median()),
        source=fallback,
        formula=(
            f"Median of {len(series):,} daily observations "
            f"(range {series.min():.1f}× – {series.max():.1f}×). Sharadar's EBITDA "
            "includes non-operating items the model's does not."
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
    elapsed: float,
    mid_year: bool,
    shares: float,
    share_basis: str,
    include_short_term_investments: bool,
    operating_cash_floor: float,
    non_operating_assets: float | None,
    base_fcff: BaseFcff,
    headline_method: str = DEFAULT_HEADLINE_METHOD,
    ronic: float | None = None,
    terminal_tax_rate: float | None = None,
    ebita_proxy: bool = True,
) -> ModelResult:
    """Projects, terminates both ways, and bridges each to a value per share.

    ``elapsed`` is the time from the base period's end to the valuation date
    (:func:`~app.logic.dcf.valuation.elapsed_years`). ``ronic`` of ``None``
    sets the terminal year's return on new capital to the WACC,
    ``terminal_tax_rate`` of ``None`` carries the final forecast year's rate, and
    ``ebita_proxy`` adds back amortisation in the terminal year.
    """
    base_revenue = _series_value(base.flows, "revenue")
    if base_revenue is None:
        raise dcf.ProjectionError(
            "The base period has no revenue figure to project from."
        )

    projection = dcf.project(
        base_revenue,
        drivers,
        wacc=wacc_result.wacc,
        elapsed=elapsed,
        mid_year=mid_year,
    )
    terminal_values = dcf.build_terminal_values(
        projection,
        terminal_growth=terminal_growth,
        exit_multiple=exit_multiple,
        ronic=ronic,
        terminal_tax_rate=terminal_tax_rate,
        ebita_proxy=ebita_proxy,
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
    growth_fade_to: float | None = None,
    growth_hold: int = 1,
    **run_model_kwargs,
) -> reverse.ReverseDcfResult:
    """What year-1 revenue growth rate reproduces ``target_price`` per share.

    The forward model asks what the shares are worth given a forecast. This asks
    what forecast the market price already contains, by holding every other
    assumption at the value the forward run used and solving the one remaining
    unknown. With ``growth_fade_to`` set, the solved rate is held for
    ``growth_hold`` years and then fades toward it exactly as the forward
    forecast's does, so the two describe the same shape of path; without it
    the rate is flat.

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
        path = replace(
            drivers,
            revenue_growth=dcf.growth_path(
                growth, drivers.years, growth_fade_to, hold=growth_hold
            ),
        )
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


#: Concepts whose base-period zero carries no information, and so would make
#: the zero-fill warning fire on the majority of companies for no reason.
#: ``fx_effect`` is zero for any filer without translation exposure, and
#: :func:`~app.logic.dcf.validation.check_cash_flow_ties` already treats an
#: absent FX line as expected rather than suspect — flagging it here as well
#: would have the two checks contradict each other on the same figure.
#:
#: The three plausibility inputs are here for the same reason: none reaches the
#: valuation, and the checks that read them treat a zero as nil — which is what
#: it means for the great majority of companies.
ZERO_FILL_UNINFORMATIVE = frozenset(
    {"fx_effect", "net_income_nci", "preferred_dividends", "acquisitions"}
)


def _zero_filled_base_items(
    dataset: CompanyDataset, base: BasePeriod
) -> tuple[tuple[str, str], ...]:
    """Base-period line items that are exactly 0 and might mean "not disclosed".

    Sharadar writes 0, not null, for a line item absent from a filing, so the
    two are indistinguishable in the data. Rather than guess, the ones that
    could plausibly be either are collected here and shown to the user as
    ``(concept label, resolved column)`` pairs. Concepts in
    :data:`ZERO_FILL_UNINFORMATIVE` are left out: the ambiguity is real for them
    too, but it never changes a decision.
    """
    from app.logic.dcf.schema import CONCEPTS_BY_KEY

    found: list[tuple[str, str]] = []
    for series in (base.flows, base.stocks):
        if series is None:
            continue
        for key, value in series.items():
            if str(key) in ZERO_FILL_UNINFORMATIVE:
                continue
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
    periods_in_use: frozenset[str] | None = None,
    drivers: dcf.DriverPath | None = None,
    beta_in_use: float | None = None,
    acquisitions_in_capex: bool = False,
    growth_basis: str | None = None,
    implied_growth: dict[str, float | None] | None = None,
    assumed_margins: dict[str, float] | None = None,
    spread_table_as_of: date | None = None,
) -> val.ValidationContext:
    """Packs everything the checks need into one context.

    ``periods_in_use`` is the set of period labels the assumptions were actually
    derived from — the driver averaging windows, the effective-tax-rate window,
    and the base period. Passing it lets the per-period reconciliation checks
    scope their severity to what can reach the valuation; omitting it leaves
    every period at full severity.
    """
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
        statement_currency=base.currency or dataset.currency or None,
        price_currency=sharadar_client.PRICE_CURRENCY,
        reporting_currency=dataset.currency or None,
        foreign_filer=dataset.foreign_filer,
        asreported_flags=dataset.statement_bases,
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
        periods_in_use=periods_in_use,
    )

    if base_fcff is not None:
        context.fcff_method_a = base_fcff.method_a
        context.fcff_method_b = base_fcff.method_b
    context.base_sbc = _series_value(base.flows, "sbc")

    if beta is not None and beta.result is not None:
        context.beta_r_squared = beta.result.r_squared
        context.beta_observations = beta.result.observations
    context.beta = beta_in_use

    # Plausibility of what the data layer derived rather than reported.
    if "net_income_nci" in annual.columns:
        context.net_income_nci = tuple(
            float(v) for v in annual["net_income_nci"].tail(3) if not pd.isna(v)
        )
    context.preferred_dividends = _series_value(base.flows, "preferred_dividends")
    if not annual.empty and "revenue" in annual.columns:
        paired = pd.concat(
            [
                fin.acquisition_spend(annual).tail(5),
                fin.capex(annual).values.tail(5),
                annual["revenue"].tail(5),
            ],
            axis=1,
        ).dropna()
        if not paired.empty and float(paired.iloc[:, 1].sum()) > 0:
            context.acquisitions_to_capex = float(
                paired.iloc[:, 0].sum() / paired.iloc[:, 1].sum()
            )
        if not paired.empty and float(paired.iloc[:, 2].sum()) > 0:
            context.acquisitions_to_revenue = float(
                paired.iloc[:, 0].sum() / paired.iloc[:, 2].sum()
            )
        context.acquisitions_by_year = tuple(
            (f"FY{label}", float(spend)) for label, spend in paired.iloc[:, 0].items()
        )
    window = fin.ttm_window_acquisitions(dataset.ttm_history, dataset.ttm_history_meta)
    if window is not None:
        context.ttm_window_acquisitions, context.ttm_window_periods = window
    context.base_net_income = _series_value(base.flows, "net_income")
    context.spread_table_as_of = spread_table_as_of
    context.acquisitions_in_capex = acquisitions_in_capex

    margins = (
        fin.driver_ratios(annual)["operating_margin"].dropna().tolist()
        if not annual.empty else []
    )
    trailing = fin.base_period_driver(
        "operating_margin", base.flows, base.stocks, is_ttm=base.is_ttm, label=base.label
    )
    if trailing.ok:
        margins.append(trailing.value)
    if margins:
        context.historical_margin_range = (float(min(margins)), float(max(margins)))
    context.assumed_margins = dict(assumed_margins or {})
    ratios = fin.driver_ratios(annual) if not annual.empty else pd.DataFrame()
    five_year = fin.driver_default(ratios, "revenue_growth", "5-year CAGR")
    context.realised_cagr = five_year.value if five_year.ok else None
    context.growth_basis = growth_basis
    context.implied_growth = dict(implied_growth or {})

    if not annual.empty and "dna" in annual.columns:
        paired = pd.concat(
            [fin.capex(annual).values.tail(5), annual["dna"].tail(5)], axis=1
        ).dropna()
        if not paired.empty and float(paired.iloc[:, 1].sum()) > 0:
            context.historical_capex_to_dna = float(
                paired.iloc[:, 0].sum() / paired.iloc[:, 1].sum()
            )

    if dataset.is_converted:
        context.fx_spot = dataset.fx_rate_value
        context.fx_spot_as_of = dataset.fx_rate.as_of
        rates = _published_rates(dataset)
        context.fx_reported = rates[-1][1] if rates else None
        context.fx_history = rates

    if drivers is not None:
        context.forecast_tax_rates = drivers.effective_tax_rate
        context.forecast_growth = drivers.revenue_growth

    if result is not None:
        context.debt_weight = result.wacc.debt_weight
        context.equity_weight = result.wacc.equity_weight
        context.wacc = result.wacc.wacc
        context.cost_of_debt = result.wacc.cost_of_debt
        context.cost_of_equity = result.wacc.cost_of_equity
        context.long_term_debt = _series_value(base.stocks, "long_term_debt")
        context.pv_explicit = result.projection.pv_explicit

        gordon = result.terminal_values["gordon"]
        exit_multiple = result.terminal_values["exit_multiple"]
        context.ronic = gordon.implied_ronic
        context.exit_implied_growth = exit_multiple.implied_growth
        context.exit_implied_ronic = exit_multiple.implied_ronic
        context.gordon_terminal_value = gordon.value
        context.gordon_value_per_share = result.bridges["gordon"].value_per_share
        context.exit_value_per_share = result.bridges["exit_multiple"].value_per_share
        context.enterprise_values = dict(result.enterprise_values)
        context.headline_equity_value = result.headline_bridge.equity_value
        if dataset.last_price:
            context.upside = result.headline_bridge.upside_vs(dataset.last_price)

        final = result.projection.final_row
        context.final_capex = final.capex
        context.final_dna = final.dna
        context.final_growth = final.revenue_growth
        context.year_one_fcff = result.projection.rows[0].fcff

    return context


def _published_rates(dataset: CompanyDataset, years: int = 5) -> tuple[tuple[date, float], ...]:
    """Sharadar's ``fxusd`` over up to the last ``years`` fiscal years, oldest first.

    Annual periods, so the depreciation check measures a trend rather than a
    quarter's noise. Units of the reporting currency per US dollar.
    """
    annual, meta = dataset.annual, dataset.annual_meta
    if annual.empty or meta.empty or "fx_to_usd" not in annual.columns:
        return ()
    dates = pd.to_datetime(meta["report_date"].reindex(annual.index), errors="coerce")
    frame = pd.DataFrame({"date": dates.to_numpy(), "rate": annual["fx_to_usd"].to_numpy()})
    frame = frame.dropna()
    frame = frame[frame["rate"] > 0].sort_values("date")
    if frame.empty:
        return ()
    cutoff = frame["date"].iloc[-1] - pd.DateOffset(years=years)
    frame = frame[frame["date"] >= cutoff]
    return tuple(
        (pd.Timestamp(when).date(), float(rate))
        for when, rate in zip(frame["date"], frame["rate"])
    )


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


#: Exit multiples in the sensitivity grid, as offsets from the chosen multiple.
EXIT_MULTIPLE_STEPS = (-0.20, -0.10, 0.0, 0.10, 0.20)


@dataclass(frozen=True)
class SensitivityGrid:
    """Headline value per share across WACC and the headline's own terminal input.

    ``values[i][j]`` pairs ``waccs[i]`` with ``columns[j]``: terminal growth
    under a Gordon headline, the exit multiple under an exit headline — the
    input that actually moves the headline. ``None`` where the model cannot be
    built, which includes a WACC at or below terminal growth (Gordon is always
    computed, even beside an exit headline).
    """

    method: str
    waccs: tuple[float, ...]
    column_label: str
    columns: tuple[float, ...]
    values: tuple[tuple[float | None, ...], ...]

    def column_heading(self, value: float) -> str:
        if self.method == "exit_multiple":
            return f"{value:,.2f}×"
        return f"g {value:.2%}"


def sensitivity_grid(
    dataset: CompanyDataset,
    base: BasePeriod,
    *,
    wacc_result: wacc_lib.WaccResult,
    terminal_growth: float,
    exit_multiple: float,
    headline_method: str = DEFAULT_HEADLINE_METHOD,
    wacc_step: float = 0.005,
    growth_step: float = 0.005,
    steps: int = 2,
    **run_model_kwargs,
) -> SensitivityGrid:
    """Reruns the model on screen across WACC × the headline's terminal input.

    The range matters more than the point estimate for a decision, and each
    cell is only the arithmetic — no fetch — so the grid is cheap. Under an
    exit-multiple headline terminal growth barely moves the value, so the
    columns there are the multiple at ±10% and ±20%; under Gordon they are
    terminal growth. The WACC is overridden whole rather than rebuilt from its
    inputs: the grid asks what a different discount rate does, not which input
    moved it.
    """
    offsets = range(-steps, steps + 1)
    waccs = tuple(wacc_result.wacc + k * wacc_step for k in offsets)
    exit_headline = headline_method == "exit_multiple"
    if exit_headline:
        columns = tuple(exit_multiple * (1.0 + step) for step in EXIT_MULTIPLE_STEPS)
        column_label = "Exit multiple"
    else:
        columns = tuple(terminal_growth + k * growth_step for k in offsets)
        column_label = "Terminal growth"

    rows: list[tuple[float | None, ...]] = []
    for rate in waccs:
        cells: list[float | None] = []
        for column in columns:
            growth = terminal_growth if exit_headline else column
            multiple = column if exit_headline else exit_multiple
            if growth >= rate or rate <= 0:
                cells.append(None)
                continue
            try:
                outcome = run_model(
                    dataset,
                    base,
                    wacc_result=replace(wacc_result, wacc=rate),
                    terminal_growth=growth,
                    exit_multiple=multiple,
                    headline_method=headline_method,
                    **run_model_kwargs,
                )
            except (dcf.ProjectionError, dcf.TerminalValueError):
                cells.append(None)
                continue
            cells.append(outcome.headline_bridge.value_per_share)
        rows.append(tuple(cells))
    return SensitivityGrid(
        method=headline_method,
        waccs=waccs,
        column_label=column_label,
        columns=columns,
        values=tuple(rows),
    )


# ---------------------------------------------------------------------------
# Pieces of the assumption set — assembled in app.logic.dcf.inputs
# ---------------------------------------------------------------------------


def synthetic_cost_of_debt(
    dataset: CompanyDataset, base: BasePeriod, risk_free: float
) -> tuple[wacc_lib.SyntheticCostOfDebt | None, str]:
    """The synthetic-rating cost of debt — see :func:`wacc.resolve_synthetic_cost_of_debt`.

    Coverage is a ratio, so the reporting currency does not matter.
    """
    flows = base.flows
    ebit = flows.get("ebit")
    interest = fin.interest_expense(pd.DataFrame([flows]))
    interest_value = (
        float(interest.iloc[0]) if not interest.empty and not pd.isna(interest.iloc[0]) else None
    )
    debt = fin.total_debt(pd.DataFrame([base.stocks]))
    debt_value = float(debt.iloc[0]) if not debt.empty and not pd.isna(debt.iloc[0]) else None
    annual = dataset.annual
    history: list[tuple[str, float | None, float | None]] = []
    if not annual.empty and "ebit" in annual.columns:
        annual_interest = fin.interest_expense(annual)
        history = [
            (
                f"FY{label}",
                None if pd.isna(annual.loc[label, "ebit"]) else float(annual.loc[label, "ebit"]),
                None if pd.isna(annual_interest.loc[label]) else float(annual_interest.loc[label]),
            )
            for label in annual.index
        ]
    return wacc_lib.resolve_synthetic_cost_of_debt(
        float(ebit) if ebit is not None and not pd.isna(ebit) else None,
        interest_value,
        risk_free,
        debt_value,
        history,
    )


class DefaultUnavailable(Exception):
    """An input the DCF tab would leave for the reader to type is missing.

    ``transient`` marks a failed fetch — a rate that may arrive on a retry —
    as opposed to data that is simply not there, so a caller caching results
    can decline to cache the first.
    """

    def __init__(self, reason: str, *, transient: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.transient = transient
