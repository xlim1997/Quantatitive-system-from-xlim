# algorithm/volume_contraction_algo.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import pandas as pd
from ib_insync import IB, util

from strategies.volume_contraction import VolumeContractionConfig, compute_signal_and_score
from universe.ibkr_scanner_universe import IBKRScannerConfig, scan_universe


@dataclass
class Candidate:
    symbol: str
    score: float
    meta: Dict[str, Any]


class VolumeContractionSelector:
    def __init__(
        self,
        ib: IB,
        scanner_cfg: IBKRScannerConfig,
        strat_cfg: VolumeContractionConfig,
    ):
        self.ib = ib
        self.scanner_cfg = scanner_cfg
        self.strat_cfg = strat_cfg

    def _fetch_daily_df(self, contract) -> Optional[pd.DataFrame]:
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="90 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            return None
        df = util.df(bars)
        # 标准化列名
        df = df.rename(columns={"date": "time", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df

    def select_top(self) -> List[Candidate]:
        contracts = scan_universe(self.ib, self.scanner_cfg)

        cands: List[Candidate] = []
        for c in contracts:
            df = self._fetch_daily_df(c)
            if df is None or df.empty:
                continue
            out = compute_signal_and_score(df, self.strat_cfg)
            if out is None:
                continue
            cands.append(Candidate(symbol=c.symbol, score=float(out["score"]), meta=out))

        cands.sort(key=lambda x: x.score, reverse=True)
        return cands[: self.strat_cfg.top_n]


def candidates_to_equal_weight_targets(cands: List[Candidate], max_weight: float = 0.15) -> Dict[str, float]:
    if not cands:
        return {}
    w = min(1.0 / len(cands), max_weight)
    return {c.symbol: w for c in cands}
