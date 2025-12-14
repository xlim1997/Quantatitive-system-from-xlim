# strategies/sample_buy_and_hold/main.py
from __future__ import annotations

from typing import Dict, List

from algorithm.base import BaseAlgorithm
from core.events import MarketDataEvent
from portfolio.models import Insight, InsightDirection


class BuyAndHoldOnceStrategy(BaseAlgorithm):
    """
    示例策略：买入并持有（只买一次）
    - 第一次有数据时：给出一个 UP insight，weight_hint=1.0（满仓）
    - 之后不再重复发 insight，由 Engine 的 active_insights 缓存保持观点
    """

    def __init__(self, symbol: str = "AAPL") -> None:
        super().__init__()
        self.symbol = symbol
        self._sent = False

    def initialize(self) -> None:
        self.add_equity(self.symbol)

    def on_data(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
        if self._sent:
            return []  # 不再重复输出，交给 Engine 缓存 active_insights

        if self.symbol not in data:
            return []

        self._sent = True
        return [
            Insight(
                symbol=self.symbol,
                direction=InsightDirection.UP,
                weight_hint=1.0,
            )
        ]
