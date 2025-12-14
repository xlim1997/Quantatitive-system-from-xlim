# data/local_csv.py
"""
本地 CSV 数据源（Local CSV DataFeed）
===================================

这个 DataFeed 主要用于回测：
- 从多个 CSV 文件中加载历史 K 线数据
- 按日期合并成一个时间序列
- 每个日期返回：{symbol: MarketDataEvent, ...}

CSV 格式约定（最简版本）：
- 必须包含：
    date, open, high, low, close
- 可选：
    volume
- date 列将被解析为日期，并作为索引（pandas.DatetimeIndex）
"""

from __future__ import annotations

from typing import Dict, Iterator

import pandas as pd

from core.events import MarketDataEvent, EventType, Bar
from .base import BaseDataFeed


class LocalCSVDataFeed(BaseDataFeed):
    """
    LocalCSVDataFeed

    参数：
    - symbol_to_path: Dict[str, str]
        例如：
        {
            "AAPL": "data/aapl_daily.csv",
            "MSFT": "data/msft_daily.csv",
        }

    行为：
    - 读取所有 CSV，按日期对齐
    - 从最早的公共日期开始迭代
    - 每个时间步返回一个 dict：{symbol: MarketDataEvent, ...}
    """

    def __init__(self, symbol_to_path: Dict[str, str]) -> None:
        self.symbol_to_path = symbol_to_path

        # 存储最近一次产生的行情切片
        self._last_market_data: Dict[str, MarketDataEvent] = {}

    # -----------------------------
    # BaseDataFeed 抽象属性实现
    # -----------------------------
    @property
    def last_market_data(self) -> Dict[str, MarketDataEvent]:
        return self._last_market_data

    # -----------------------------
    # 迭代接口：按时间输出行情切片
    # -----------------------------
    def __iter__(self) -> Iterator[Dict[str, MarketDataEvent]]:
        """
        迭代逻辑：

        1. 读取 symbol_to_path 中所有 CSV，解析为 DataFrame
        2. 把每个 DataFrame 的 date 列设为索引
        3. 合并出所有日期的并集，排序
        4. 对每个日期 dt：
            - 检查每个 symbol 是否在该日期有数据
            - 有数据则构造 MarketDataEvent 放入 dict
            - 把该 dict 作为该时间步的行情切片 yield 出去
        """

        # 1. 读入所有 CSV → {symbol: DataFrame}
        dfs: Dict[str, pd.DataFrame] = {}

        for symbol, path in self.symbol_to_path.items():
            df = pd.read_csv(path)

            # 要求至少有 date, open, high, low, close
            if "date" not in df.columns:
                raise ValueError(f"CSV for {symbol} at {path} missing 'date' column.")

            # 解析日期
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            # 转为 float，避免 int/str 混杂
            for col in ["open", "high", "low", "close"]:
                if col not in df.columns:
                    raise ValueError(f"CSV for {symbol} at {path} missing '{col}' column.")
                df[col] = df[col].astype(float)

            # volume 可选
            if "volume" in df.columns:
                df["volume"] = df["volume"].astype(float)
            else:
                df["volume"] = 0.0

            dfs[symbol] = df

        if not dfs:
            return  # 没有任何数据，直接返回（不会 yield）

        # 2. 计算所有日期的并集，并排序
        all_dates = sorted(set().union(*[df.index for df in dfs.values()]))

        # 3. 逐日期迭代
        for dt in all_dates:
            events: Dict[str, MarketDataEvent] = {}

            for symbol, df in dfs.items():
                if dt not in df.index:
                    # 该 symbol 在这个日期没数据（停牌/尚未上市/退市）
                    continue

                row = df.loc[dt]
                bar = Bar(
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0)),
                )

                md_event = MarketDataEvent(
                    type=EventType.MARKET,
                    timestamp=dt,
                    symbol=symbol,
                    bar=bar,
                    extra=None,  # 你可以后来加因子/指标等
                )

                events[symbol] = md_event

            # 如果这个日期至少有一个 symbol 有数据，就 yield 一次
            if events:
                self._last_market_data = events
                yield events
