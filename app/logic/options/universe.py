"""Which names the collector follows, and the theme each is drawn under.

**Curated, and shown on the page as a judgement** — the same kind of claim as
``research.holdings.FILER_FAMILIES``. The themes are reading-aids, not a
classification: Sharadar's sector labels cover stocks and say nothing useful
about a silver-miners ETF, and a name sits where a reader would look for it.
Edit this file to change the universe; the collector picks the change up on its
next run and the manifest records the list's hash. The page reads each name's
theme from here, not from the stored tables, so once the API deploys a renamed
or moved theme shows and a name taken out leaves; a name added shows after the
collector's next run.

An edit carries three things with it:

* **A single stock goes in** :data:`SINGLE_STOCKS` **as well**, or it gets no
  earnings date and the earnings filter lets it through without saying so.
* **A name whose options are not listed under its own ticker** goes in
  :data:`OPTION_ROOTS` (Berkshire's are under ``BRKB``, not ``BRK.B``).
* **A new name has no history.** The daily run fetches only sessions after the
  newest one stored, so the name shows no score until it has
  ``min_history_sessions`` of them — about six months — unless it is backfilled
  (``collector/README.md``). Tickers are reused, so a backfill starts no earlier
  than the listing: SPCX was a SPAC fund near $22 until 2026-04-06 and is SpaceX
  from 2026-06-12.
"""

from __future__ import annotations

import hashlib

#: ticker → theme. Order is display order within a theme.
UNIVERSE: dict[str, str] = {
    # Benchmarks and volatility
    "SPY": "Benchmark", "QQQ": "Benchmark", "DIA": "Benchmark",
    "IWM": "Benchmark", "MDY": "Benchmark", "RSP": "Benchmark", "SCHD": "Benchmark",
    "VXX": "Volatility",
    # Sectors
    "XLK": "Technology", "ARKK": "Technology", "AAPL": "Technology", "ANT": "Technology", "AMZN": "Technology", "GOOG": "Communications",
    "META": "Communications", "MSFT": "Technology", "NFLX": "Communications", "SPCX": "Technology", "TSLA": "Technology", "PLTR": "Technology",
    "SNOW": "Technology", "CRM": "Technology",
    "PANW": "Cybersecurity", "CRWD": "Cybersecurity", "NET": "Cybersecurity", "FTNT": "Cybersecurity", "ZS": "Cybersecurity", "OKTA": "Cybersecurity",
    "SMH": "Semiconductors", "SOXX": "Semiconductors", "NVDA": "Semiconductors", "TSM": "Semiconductors", "MU": "Semiconductors",
    "AVGO": "Semiconductors", "AMD": "Semiconductors", "INTC": "Semiconductors", "SKHY": "Semiconductors", "AMAT": "Semiconductors",
    "TXN": "Semiconductors", "ADI": "Semiconductors", "ASML": "Semiconductors", "KLAC": "Semiconductors",
    "XLY": "Consumer Discretionary", "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "BKNG": "Consumer Discretionary", "MAR": "Consumer Discretionary", "HLT": "Consumer Discretionary",
    "XLP": "Consumer Staples", "WMT": "Consumer Staples", "COST": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "TGT": "Consumer Staples",
    "XLU": "Utilities", "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "AEP": "Utilities", "CEG": "Utilities", "D": "Utilities",
    "XLF": "Financials", "BRK.B": "Financials", "JPM": "Financials", "V": "Financials", "MA": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials",
    "XLV": "Healthcare", "LLY": "Healthcare", "JNJ": "Healthcare", "ABBV": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "QTUM": "Quantum", "QBTS": "Quantum", "RGTI": "Quantum", "IONQ": "Quantum",
    "XLC": "Other Sectors", "XLI": "Other Sectors", "XLB": "Other Sectors", "XHB": "Other Sectors", "XLRE": "Other Sectors",
    # Energy and metals
    "XLE": "Energy", "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "USO": "Energy", "UNG": "Energy",
    "GLD": "Metals", "SLV": "Metals", "GDX": "Metals", "COPX": "Metals", "URA": "Metals", "MP": "Metals",
    # Rates and credit
    "TLT": "Fixed Income", "IEF": "Fixed Income", "LQD": "Fixed Income", "VCIT": "Fixed Income",
    # Abroad
    "EEM": "Foreign", "EMXC": "Foreign", "EWC": "Foreign", "EWU": "Foreign", "EWG": "Foreign", "EWA": "Foreign",
    "EWJ": "Foreign", "EWY": "Foreign", "EWZ": "Foreign", "EWW": "Foreign", "INDA": "Foreign",
    # Crypto
    "IBIT": "Crypto", "ETHA": "Crypto",
}

#: Names whose options are listed under another root, or under several. The
#: OCC root drops a share class's dot. An index can have two: SPX, when it was
#: followed, was ``("SPXW", "SPX")`` — weeklies and PM-settled monthlies under
#: ``SPXW``, AM-settled monthlies under ``SPX`` — and where one expiry is listed
#: under both, the collector keeps the first root's row.
OPTION_ROOTS: dict[str, tuple[str, ...]] = {"BRK.B": ("BRKB",)}

#: Underlyings whose spot is an index level rather than a traded price, so their
#: closes come from ThetaData's index history. None is followed now; SPX was,
#: until 2026-11.
INDICES: frozenset[str] = frozenset()

#: Names that report earnings — the only ones the earnings filter applies to.
#: Everything else in the universe is a fund. In :data:`UNIVERSE` order.
SINGLE_STOCKS = frozenset({
    "AAPL", "ANT", "AMZN", "GOOG", "META", "MSFT", "NFLX", "SPCX", "TSLA", "PLTR", "SNOW", "CRM",
    "PANW", "CRWD", "NET", "FTNT", "ZS", "OKTA",
    "NVDA", "TSM", "MU", "AVGO", "AMD", "INTC", "SKHY", "AMAT", "TXN", "ADI", "ASML", "KLAC",
    "HD", "MCD", "LOW", "SBUX", "BKNG", "MAR", "HLT",
    "WMT", "COST", "PG", "KO", "PEP", "TGT",
    "NEE", "DUK", "SO", "AEP", "CEG", "D",
    "BRK.B", "JPM", "V", "MA", "BAC", "GS", "MS",
    "LLY", "JNJ", "ABBV", "UNH", "PFE",
    "QBTS", "RGTI", "IONQ",
    "XOM", "CVX", "COP",
    "MP",
})


def themes() -> list[str]:
    return sorted(set(UNIVERSE.values()))


def universe_hash() -> str:
    """Short, stable fingerprint of the list, recorded in the manifest."""
    text = ",".join(f"{t}:{UNIVERSE[t]}" for t in sorted(UNIVERSE))
    return hashlib.sha256(text.encode()).hexdigest()[:12]
