# data/universe_provider.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Dict, List, Tuple
import re
import pandas as pd


# --- IBKR contract spec you can later convert to ib_insync.Contract ---
@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    currency: str
    exchange: str = "SMART"
    primary_exchange: Optional[str] = None  # e.g., "IBIS", "SBF", "AEB"
    sec_id_type: Optional[str] = None       # e.g., "ISIN"
    sec_id: Optional[str] = None            # the ISIN string
    source: Optional[str] = None            # for logging/debugging


_SUFFIX_TO_PRIMARY_EXCHANGE: Dict[str, Tuple[str, Optional[str]]] = {
    # Yahoo suffix -> (currency, primaryExchange)
    "DE": ("EUR", "IBIS"),   # Xetra (IBIS) on IBKR
    "PA": ("EUR", "SBF"),    # Euronext Paris
    "AS": ("EUR", "AEB"),    # Euronext Amsterdam
    "BR": ("EUR", "ENEXT.BE"),  # Euronext Brussels (often used; you can also omit primary_exchange if unsure)
    "MI": ("EUR", "BVME"),   # Borsa Italiana (often BVME)
    # add more when needed...
}

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Wikipedia tables sometimes have MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in df.columns.values]
    return df

def _parse_yahoo_style_ticker(ticker: str) -> Tuple[str, Optional[str]]:
    # "SAP.DE" -> ("SAP", "DE"), "ADYEN.AS" -> ("ADYEN","AS"), "AAPL" -> ("AAPL",None)
    m = re.match(r"^([A-Za-z0-9.\-]+?)(?:\.([A-Za-z]+))?$", ticker.strip())
    if not m:
        return ticker.strip(), None
    return m.group(1), (m.group(2).upper() if m.group(2) else None)

def _pick_table_by_columns(tables: Iterable[pd.DataFrame], must_have_any: List[str]) -> pd.DataFrame:
    """
    Find the first table whose columns contain (case-insensitive) all keywords in must_have_any (any-match).
    Example: must_have_any=["Ticker"].
    """
    for t in tables:
        t = _flatten_columns(t)
        cols = [str(c).lower() for c in t.columns]
        if any(any(k.lower() in c for c in cols) for k in must_have_any):
            return t
    raise ValueError(f"Cannot find table with columns like: {must_have_any}")

def _extract_tickers(df: pd.DataFrame, col_keywords: List[str]) -> List[str]:
    df = _flatten_columns(df)
    # find best matching column
    col = None
    for c in df.columns:
        cl = str(c).lower()
        if any(k.lower() in cl for k in col_keywords):
            col = c
            break
    if col is None:
        raise ValueError(f"Ticker column not found. Existing columns: {list(df.columns)}")

    tickers = []
    for x in df[col].astype(str).tolist():
        x = x.strip()
        if x and x.lower() != "nan":
            tickers.append(x)
    return tickers

def _maybe_extract_isin(df: pd.DataFrame) -> Dict[str, str]:
    """
    If the table contains an ISIN column, return mapping {row_index_str: isin}.
    (Wikipedia tables differ; this is best-effort.)
    """
    df = _flatten_columns(df)
    isin_col = None
    for c in df.columns:
        if "isin" in str(c).lower():
            isin_col = c
            break
    if isin_col is None:
        return {}

    out = {}
    for i, v in enumerate(df[isin_col].astype(str).tolist()):
        v = v.strip()
        if v and v.lower() != "nan":
            out[str(i)] = v
    return out


class UniverseProvider:
    """
    universe_name -> List[ContractSpec]
    Uses Wikipedia tables for: DAX, EUROSTOXX50, CAC40, FTSE100, SP500 (easy to extend).
    """

    WIKI = {
        "DAX": "https://en.wikipedia.org/wiki/DAX",
        "EUROSTOXX50": "https://en.wikipedia.org/wiki/EURO_STOXX_50",
        "CAC40": "https://en.wikipedia.org/wiki/CAC_40",
        "FTSE100": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    }

    ALIASES = {
        "DAX40": "DAX",
        "DAX": "DAX",
        "STOXX50": "EUROSTOXX50",
        "EUROSTOXX50": "EUROSTOXX50",
        "CAC40": "CAC40",
        "FTSE100": "FTSE100",
        "SP500": "SP500",
        "S&P500": "SP500",
    }

    def get(self, name: str) -> List[ContractSpec]:
        key = self.ALIASES.get(name.strip().upper().replace(" ", ""), None)
        if not key:
            # fallback: try raw upper
            key = name.strip().upper()

        if key not in self.WIKI:
            raise ValueError(f"Unsupported universe: {name}. Supported: {sorted(self.WIKI.keys())}")

        url = self.WIKI[key]
        tables = pd.read_html(url)

        if key == "SP500":
            # Wikipedia SP500 table has column "Symbol"
            df = _pick_table_by_columns(tables, must_have_any=["Symbol"])
            tickers = _extract_tickers(df, col_keywords=["Symbol"])
            return [ContractSpec(symbol=t, currency="USD", source=url) for t in tickers]

        # For DAX / EUROSTOXX50 / CAC40 / FTSE100: often Yahoo-style tickers exist in a "Ticker" column
        df = _pick_table_by_columns(tables, must_have_any=["Ticker"])
        tickers = _extract_tickers(df, col_keywords=["Ticker"])

        # ISIN is optional; if present, use it (best for disambiguation)
        isins_by_row = _maybe_extract_isin(df)

        specs: List[ContractSpec] = []
        for i, t in enumerate(tickers):
            base, suffix = _parse_yahoo_style_ticker(t)
            currency = "EUR"
            primary = None

            if suffix and suffix in _SUFFIX_TO_PRIMARY_EXCHANGE:
                currency, primary = _SUFFIX_TO_PRIMARY_EXCHANGE[suffix]

            isin = isins_by_row.get(str(i))
            if isin:
                specs.append(ContractSpec(
                    symbol=base, currency=currency, exchange="SMART",
                    primary_exchange=primary,
                    sec_id_type="ISIN", sec_id=isin,
                    source=url
                ))
            else:
                specs.append(ContractSpec(
                    symbol=base, currency=currency, exchange="SMART",
                    primary_exchange=primary,
                    source=url
                ))

        return specs
