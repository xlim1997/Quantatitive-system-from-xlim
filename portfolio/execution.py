# portfolio/execution.py
"""
执行模型（Execution Model）
=========================

职责：
- 把最终的目标权重（Adjusted Targets）转成具体订单（OrderEvent）
- 订单交给 Engine.emit_order() -> Brokerage.place_order() 执行

关键点：
- ExecutionModel 不负责决定“目标权重”，它只负责把目标变成单子
- 未来可以在这里实现：
  - TWAP/VWAP 分批执行
  - 根据 bid/ask spread 决定成交价格
  - 限价单挂单策略
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, TYPE_CHECKING

from core.events import OrderEvent, EventType, OrderSide, OrderType
from portfolio.models import PortfolioTarget
from portfolio.state import Portfolio

if TYPE_CHECKING:
    from core.engine import Engine


class BaseExecutionModel(ABC):
    def __init__(self) -> None:
        self._engine: Engine | None = None

    def set_engine(self, engine: "Engine") -> None:
        self._engine = engine

    @abstractmethod
    def execute(
        self,
        portfolio: Portfolio,
        targets: List[PortfolioTarget],
        last_prices: Dict[str, float],
    ) -> None:
        """
        把 targets 转成订单，并交给引擎发送给 Brokerage。
        """
        ...


class ImmediateExecutionModel(BaseExecutionModel):
    """
    最简单的执行模型：
    - 一次性把当前仓位调整到目标仓位（按目标权重换算成目标市值 -> 目标股数）
    - 生成市价单（MARKET）

    参数：
    - min_trade_value: 小于该金额的调仓忽略（避免频繁小单）
    """

    def __init__(self, min_trade_value: float = 0.0) -> None:
        super().__init__()
        self.min_trade_value = min_trade_value

    def execute(
        self,
        portfolio: Portfolio,
        targets: List[PortfolioTarget],
        last_prices: Dict[str, float],
    ) -> None:
        if self._engine is None:
            raise RuntimeError("ExecutionModel has no engine attached.")

        # 当前组合净值
        equity = portfolio.total_value(last_prices)

        # 将 targets 转为 dict，方便查找
        target_map = {t.symbol: t.target_percent for t in targets}

        # 处理“既有持仓但 target 没给”的情况：
        # - 这意味着目标是 0（需要清仓）
        symbols_union = set(list(portfolio.positions.keys()) + list(target_map.keys()))

        for symbol in symbols_union:
            price = last_prices.get(symbol)
            if price is None or price <= 0:
                # 没价格就无法换算目标股数
                continue

            target_percent = target_map.get(symbol, 0.0)
            target_value = equity * target_percent

            current_pos = portfolio.positions.get(symbol)
            current_qty = 0.0 if current_pos is None else current_pos.quantity
            current_value = current_qty * price

            diff_value = target_value - current_value

            # 忽略太小的调仓
            if abs(diff_value) < self.min_trade_value:
                continue

            # 目标差额换算成股数（整数股）
            qty = int(diff_value / price)
            if qty == 0:
                continue

            side = OrderSide.BUY if qty > 0 else OrderSide.SELL

            order = OrderEvent(
                type=EventType.ORDER,
                timestamp=self._engine.current_time,
                symbol=symbol,
                quantity=qty,
                side=side,
                order_type=OrderType.MARKET,
                tag="ImmediateExec",
            )

            # 交给引擎发送给 Brokerage
            self._engine.emit_order(order)
