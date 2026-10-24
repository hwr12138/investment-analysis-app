"""Which names the collector follows, and the theme each is drawn under.

**Curated, and shown on the page as a judgement** — the same kind of claim as
``research.holdings.FILER_FAMILIES``. The list was transcribed from the
screenshots the page was specified from: 90 of the 106 names there were
legible, the rest overlapped one another. The themes are reading-aids, not a
classification: Sharadar's sector labels cover stocks and say nothing useful
about a silver-miners ETF. Edit this file to change the universe; the collector
picks the change up on its next run and the manifest records the list's hash.
"""

from __future__ import annotations

import hashlib

#: ticker → theme. Order is display order within a theme.
UNIVERSE: dict[str, str] = {
    # Benchmarks and their leveraged / volatility wrappers
    "SPY": "Benchmark", "SPX": "Benchmark", "VOO": "Benchmark", "QQQ": "Benchmark", "DIA": "Benchmark",
    "IWM": "Benchmark", "MDY": "Benchmark", "RSP": "Benchmark", "SCHD": "Benchmark",
    "UPRO": "Leveraged", "TQQQ": "Leveraged", "TNA": "Leveraged",
    "UVXY": "Volatility", "VXX": "Volatility",
    # Sectors
    "XLK": "Technology", "ARKK": "Technology", "SMH": "Semiconductors", "SOXX": "Semiconductors",
    "MU": "Semiconductors", "AVGO": "Semiconductors", "NVDA": "Semiconductors",
    "XLC": "Communications", "META": "Internet", "KWEB": "Internet", "NFLX": "Entertainment",
    "XLY": "Consumer", "XRT": "Consumer", "XHB": "Consumer", "XLP": "Consumer Staples", "MSOS": "Consumer",
    "TSLA": "Auto", "JETS": "Airlines", "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities",
    "XLRE": "Real Estate", "VNQ": "Real Estate", "IYR": "Real Estate",
    "XLF": "Financials", "KRE": "Financials", "KBE": "Financials", "CME": "Financials",
    "XLV": "Healthcare", "UNH": "Healthcare", "XBI": "Biotech", "ARKG": "Biotech",
    "QTUM": "Quantum",
    # Energy and commodities
    "XLE": "Energy", "XOP": "Energy", "OIH": "Energy",
    "USO": "Commodities", "BNO": "Commodities", "UNG": "Commodities", "WEAT": "Commodities",
    "URNM": "Clean Energy", "URA": "Clean Energy", "NLR": "Clean Energy", "ICLN": "Clean Energy",
    "TAN": "Clean Energy",
    "GLD": "Metals", "IAU": "Metals", "SLV": "Metals", "GDX": "Metals", "GDXJ": "Metals", "SILJ": "Metals",
    "XME": "Metals",
    # Rates, credit and currencies
    "TLT": "Fixed Income", "IEF": "Fixed Income", "LQD": "Fixed Income", "EMB": "Fixed Income",
    "MUB": "Fixed Income",
    "FXE": "FX", "FXY": "FX",
    # Abroad
    "EFA": "Foreign", "VWO": "Foreign", "EEM": "Foreign", "FEZ": "Foreign", "EWG": "Foreign",
    "EWJ": "Foreign", "EWY": "Foreign", "EWZ": "Foreign", "EWW": "Foreign", "EWC": "Foreign",
    "INDA": "Foreign", "FXI": "Foreign", "ASHR": "Foreign",
    # Crypto
    "IBIT": "Crypto", "ETHA": "Crypto", "GBTC": "Crypto", "BITO": "Crypto",
}

#: Index options trade under more than one root. SPX monthlies are AM-settled
#: under ``SPX``; the weeklies and PM-settled monthlies under ``SPXW``. One
#: expiry date can appear under both, so the collector fetches each root and
#: keeps the PM-settled ``SPXW`` row where they overlap.
INDEX_ROOTS: dict[str, tuple[str, ...]] = {"SPX": ("SPXW", "SPX")}

#: Underlyings whose spot is an index level rather than a traded price.
INDICES = frozenset({"SPX"})

#: Names that report earnings — the only ones the earnings filter applies to.
#: Everything else in the universe is a fund or an index.
SINGLE_STOCKS = frozenset({"MU", "AVGO", "NVDA", "META", "NFLX", "TSLA", "UNH", "CME"})


def themes() -> list[str]:
    return sorted(set(UNIVERSE.values()))


def universe_hash() -> str:
    """Short, stable fingerprint of the list, recorded in the manifest."""
    text = ",".join(f"{t}:{UNIVERSE[t]}" for t in sorted(UNIVERSE))
    return hashlib.sha256(text.encode()).hexdigest()[:12]
