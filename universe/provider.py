# universe/provider.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional
import os
import time
import json

import pandas as pd


UniverseName = Literal["NASDAQ100", "SP500"]


NASDAQ100_URL = "https://www.nasdaq.com/solutions/global-indexes/nasdaq-100/companies"  # :contentReference[oaicite:2]{index=2}
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"           # :contentReference[oaicite:3]{index=3}


def to_ibkr_symbol(symbol: str) -> str:
    """
    IBKR 常见 class-share ticker 处理：
    - Wikipedia: BRK.B / BF.B
    - IBKR:      BRK B / BF B
    其他带点的 ticker 也用同样规则（更通用）。
    """
    s = symbol.strip().upper()
    return s.replace(".", " ")


def _read_html_tables(url: str) -> List[pd.DataFrame]:
    # read_html 依赖 lxml；pip install lxml
    return pd.read_html(url)


def _extract_column(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    cols = [str(c).strip().lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in cols:
            col = df.columns[cols.index(cand.lower())]
            return df[col]
    return None


def fetch_nasdaq100_symbols(source: Literal["nasdaq"] = "nasdaq") -> List[str]:
    """
    从 Nasdaq 官方页面抓 NASDAQ-100 tickers。:contentReference[oaicite:4]{index=4}
    """
    if source != "nasdaq":
        raise ValueError("NASDAQ100 only supports source='nasdaq' in this helper.")

    tables = _read_html_tables(NASDAQ100_URL)
    import ipdb; ipdb.set_trace()
    # 页面里通常有 “Ticker” 列
    for t in tables:
        s = _extract_column(t, ["Ticker", "Symbol"])
        if s is None:
            continue
        syms = (
            s.astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": None, "": None})
            .dropna()
            .tolist()
        )
        # 粗过滤：ticker 通常是字母/数字/点/短横线
        syms = [x for x in syms if len(x) <= 10]
        if len(syms) >= 80:  # 容错：至少接近 100
            return syms

    raise RuntimeError("Failed to parse NASDAQ-100 tickers from Nasdaq page (structure may have changed).")


def fetch_sp500_symbols(source: Literal["wikipedia"] = "wikipedia") -> List[str]:
    """
    从 Wikipedia 的 S&P 500 成分表抓 tickers。:contentReference[oaicite:5]{index=5}
    """
    if source != "wikipedia":
        raise ValueError("SP500 only supports source='wikipedia' in this helper.")

    tables = _read_html_tables(SP500_WIKI_URL)
    # 页面里有一个 “Symbol” 列的成分表
    for t in tables:
        s = _extract_column(t, ["Symbol"])
        if s is None:
            continue
        syms = (
            s.astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": None, "": None})
            .dropna()
            .tolist()
        )
        if len(syms) >= 400:
            return syms

    raise RuntimeError("Failed to parse S&P 500 tickers from Wikipedia table.")


@dataclass
class UniverseProviderConfig:
    cache_dir: str = "./universe_cache"
    cache_ttl_seconds: int = 24 * 3600  # 默认缓存 24h
    return_ibkr_symbols: bool = True    # True: 返回适配 IBKR 的 ticker（BRK.B -> BRK B）


class UniverseProvider:
    def __init__(self, cfg: UniverseProviderConfig = UniverseProviderConfig()):
        self.cfg = cfg
        os.makedirs(self.cfg.cache_dir, exist_ok=True)

    def _cache_path(self, universe: str) -> str:
        return os.path.join(self.cfg.cache_dir, f"{universe}.json")

    def _load_cache(self, universe: str) -> Optional[List[str]]:
        path = self._cache_path(universe)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            ts = float(obj.get("ts", 0))
            if time.time() - ts > self.cfg.cache_ttl_seconds:
                return None
            syms = obj.get("symbols", [])
            if isinstance(syms, list) and syms:
                return syms
        except Exception:
            return None
        return None

    def _save_cache(self, universe: str, symbols: List[str]) -> None:
        path = self._cache_path(universe)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "symbols": symbols}, f, ensure_ascii=False, indent=2)

    def get_symbols(self, universe: UniverseName, force_refresh: bool = False) -> List[str]:
        """
        universe:
          - "NASDAQ100"
          - "SP500"
        """
        uni = universe.upper()
        import ipdb; ipdb.set_trace()
        if not force_refresh:
            cached = self._load_cache(uni)
            if cached is not None:
                return [to_ibkr_symbol(x) for x in cached] if self.cfg.return_ibkr_symbols else cached
        import ipdb; ipdb.set_trace()
        if uni == "NASDAQ100":
            symbols = fetch_nasdaq100_symbols(source="nasdaq")
        elif uni == "SP500":
            symbols = fetch_sp500_symbols(source="wikipedia")
        else:
            raise ValueError(f"Unsupported universe: {universe}")
        import ipdb; ipdb.set_trace()
        self._save_cache(uni, symbols)
        return [to_ibkr_symbol(x) for x in symbols] if self.cfg.return_ibkr_symbols else symbols
