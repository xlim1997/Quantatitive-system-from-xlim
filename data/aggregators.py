# data/aggregators.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd

from core.events import MarketDataEvent, Bar, EventType

@dataclass
class AggConfig:
    rule: str = "1min"  # pandas resample rule, e.g. "1min", "5min"


class BarAggregator:
    """
    将输入 MarketDataEvent（timestamp + bar）按 rule 聚合。
    - 输入可以是 5s / 1m bar
    - 输出为聚合后的 bar（在窗口结束时吐出）
    """

    def __init__(self, cfg: AggConfig) -> None:
        self.cfg = cfg
        self._buffers: Dict[str, pd.DataFrame] = {}
        self._last_emitted_bucket: Dict[str, pd.Timestamp] = {}

    def push(self, ev: MarketDataEvent) -> Optional[MarketDataEvent]:
        sym = ev.symbol
        ts = pd.Timestamp(ev.timestamp)
        row = {
            "open": ev.bar.open,
            "high": ev.bar.high,
            "low": ev.bar.low,
            "close": ev.bar.close,
            "volume": getattr(ev.bar, "volume", 0.0),
        }
        df = self._buffers.get(sym)
        if df is None:
            df = pd.DataFrame(columns=row.keys())
            self._buffers[sym] = df

        df.loc[ts] = row
        df.sort_index(inplace=True)

        bucket = ts.floor(self.cfg.rule)
        last_bucket = self._last_emitted_bucket.get(sym)

        # 当进入新 bucket 时，吐出上一个 bucket 的聚合结果
        if last_bucket is not None and bucket > last_bucket:
            window = df[(df.index >= last_bucket) & (df.index < bucket)]
            if len(window) == 0:
                self._last_emitted_bucket[sym] = bucket
                return None

            out = Bar(
                open=float(window["open"].iloc[0]),
                high=float(window["high"].max()),
                low=float(window["low"].min()),
                close=float(window["close"].iloc[-1]),
                volume=float(window["volume"].sum()),
            )

            # 清掉已经输出的窗口数据，保留当前 bucket 之后的数据
            self._buffers[sym] = df[df.index >= bucket].copy()
            self._last_emitted_bucket[sym] = bucket

            return MarketDataEvent(
                type=EventType.MARKET,
                timestamp=bucket,  # 用 bucket 作为时间戳
                symbol=sym,
                bar=out,
                extra={"aggregated": True, "rule": self.cfg.rule},
            )

        # 初始化 bucket
        if last_bucket is None:
            self._last_emitted_bucket[sym] = bucket

        return None
