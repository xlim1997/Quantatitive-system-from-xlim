# portfolio/execution.py
from __future__ import annotations

import math
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
        ...


class ImmediateExecutionModel(BaseExecutionModel):
    """
    修复要点：
    1) 强制清仓（target=0）时：qty = -current_qty（不走 diff/price，避免 int 截断残股）
    2) SELL 先执行，再 BUY（释放现金）
    3) BUY 受现金约束（避免 cash 变负），可留现金 buffer
    4) |target_percent| 很小视为 0（避免权重抖动造成尘埃单）
    """

    def __init__(
        self,
        min_trade_value: float = 0.0,
        cash_buffer_pct: float = 0.0,      # 例如 0.005 表示留 0.5% 现金
        allow_margin: bool = False,
        eps_weight: float = 1e-10,
        commission_per_share: float = 0.0, # 如果你没有佣金模型，就保持 0
        min_commission: float = 0.0,
    ) -> None:
        super().__init__()
        self.min_trade_value = float(min_trade_value)
        self.cash_buffer_pct = float(cash_buffer_pct)
        self.allow_margin = bool(allow_margin)
        self.eps_weight = float(eps_weight)
        self.commission_per_share = float(commission_per_share)
        self.min_commission = float(min_commission)

    def _est_commission(self, qty: int, price: float) -> float:
        # 只是“估算”，用于限制买入不超现金；真实佣金以 FillEvent 为准
        c = abs(qty) * self.commission_per_share
        return max(self.min_commission, c)

    def _max_affordable_buy_qty(self, cash: float, price: float, desired_qty: int) -> int:
        if desired_qty <= 0 or cash <= 0 or price <= 0:
            return 0
        budget = cash * (1.0 - self.cash_buffer_pct)

        lo, hi = 0, desired_qty
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cost = mid * price + self._est_commission(mid, price)
            if cost <= budget + 1e-9:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def execute(
        self,
        portfolio: Portfolio,
        targets: List[PortfolioTarget],
        last_prices: Dict[str, float],
    ) -> None:
        if self._engine is None:
            raise RuntimeError("ExecutionModel has no engine attached.")

        equity = portfolio.total_value(last_prices)
        target_map = {t.symbol: float(t.target_percent) for t in targets}

        symbols_union = set(portfolio.positions.keys()) | set(target_map.keys())

        sells: List[tuple[str, int, float]] = []
        buys: List[tuple[str, int, float]] = []

        for symbol in symbols_union:
            price = last_prices.get(symbol)
            if price is None or price <= 0:
                continue

            pos = portfolio.positions.get(symbol)
            current_qty = 0 if pos is None else int(round(pos.quantity))

            w = target_map.get(symbol, 0.0)
            if abs(w) <= self.eps_weight:
                w = 0.0

            # --- 关键：强制清仓不留残股 ---
            if w == 0.0 and current_qty != 0:
                order_qty = -current_qty
                sells.append((symbol, order_qty, price))
                continue

            # 非强制清仓：用目标权重换算目标股数
            target_value = equity * w
            target_qty = int(math.floor(target_value / price)) if target_value > 0 else 0

            order_qty = target_qty - current_qty
            if order_qty == 0:
                continue

            trade_value = abs(order_qty) * price
            if trade_value < self.min_trade_value:
                continue

            if order_qty < 0:
                sells.append((symbol, order_qty, price))
            else:
                buys.append((symbol, order_qty, price))

        # 先卖出释放现金
        cash_sim = float(portfolio.cash)

        for symbol, qty, price in sells:
            fee = self._est_commission(qty, price)
            proceeds = (-qty) * price - fee
            cash_sim += proceeds

            order = OrderEvent(
                type=EventType.ORDER,
                timestamp=self._engine.current_time,
                symbol=symbol,
                quantity=qty,            # qty 为负
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                tag="ImmediateExec",
            )
            self._engine.emit_order(order)

        # 再买入（若不允许保证金，就按现金约束）
        for symbol, desired_qty, price in buys:
            qty = desired_qty
            if not self.allow_margin:
                qty = self._max_affordable_buy_qty(cash_sim, price, desired_qty)

            if qty <= 0:
                continue

            fee = self._est_commission(qty, price)
            cost = qty * price + fee
            cash_sim -= cost

            order = OrderEvent(
                type=EventType.ORDER,
                timestamp=self._engine.current_time,
                symbol=symbol,
                quantity=qty,            # qty 为正
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                tag="ImmediateExec",
            )
            self._engine.emit_order(order)
