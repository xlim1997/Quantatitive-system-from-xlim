# brokerage/paper.py
"""
纸上撮合（Paper Brokerage）
==========================

这是回测阶段最关键的“最小可用”撮合器：

- 接收 OrderEvent
- 根据当前行情价格（通常用 close）立即成交
- 生成 FillEvent 返回给 Engine
- 支持简单的滑点 slippage 和手续费 commission

注意：
- 这是一个简化实现，不支持部分成交、限价单排队等复杂机制
- 后续如果你要更真实的撮合，可以扩展：
  - 限价单只有当价格穿过才成交
  - 按成交量限制单笔成交量
  - 模拟 bid/ask spread（买按 ask，卖按 bid）
"""

from __future__ import annotations

from typing import List

from core.events import (
    OrderEvent,
    FillEvent,
    EventType,
    OrderType,
    OrderSide,
)


from .base import BaseBrokerage


class PaperBrokerage(BaseBrokerage):
    """
    PaperBrokerage：最简单的回测撮合

    参数：
    - slippage: 滑点比例（如 0.0005 表示 5 bps）
      - 买单成交价 = close * (1 + slippage)
      - 卖单成交价 = close * (1 - slippage)
    - commission_rate: 手续费比例（按成交额计）
      - commission = abs(qty * price) * commission_rate
    - fixed_commission: 固定手续费（每笔）
      - commission += fixed_commission
    """

    def __init__(
        self,
        slippage: float = 0.0,
        commission_rate: float = 0.0,
        fixed_commission: float = 0.0,
    ) -> None:
        super().__init__()
        self.slippage = slippage
        self.commission_rate = commission_rate
        self.fixed_commission = fixed_commission

        # 存储待处理的成交回报（Engine 每步会取走）
        self._pending_fills: List[FillEvent] = []

        # 简单的自增订单 id
        self._order_counter = 0

    def place_order(self, order: OrderEvent) -> None:
        """
        接收订单并生成成交。

        当前实现：
        - 只支持 MARKET 单（其余类型先报错/忽略）
        - 立即以当前 close 成交
        """
        if self._engine is None:
            raise RuntimeError("PaperBrokerage has no engine attached.")

        # 目前最简单版本：只支持市价单
        if order.order_type != OrderType.MARKET:
            raise NotImplementedError(
                f"PaperBrokerage only supports MARKET orders for now, got {order.order_type}."
            )

        # 从 DataFeed 的 last_market_data 中获取最新价格
        md = self._engine.data_feed.last_market_data.get(order.symbol)
        if md is None:
            raise ValueError(
                f"No market data for symbol={order.symbol} at time={self._engine.current_time}."
            )

        close = md.bar.close

        # 计算滑点后的成交价
        # 买：价格更高；卖：价格更低
        if order.side == OrderSide.BUY:
            fill_price = close * (1.0 + self.slippage)
        else:
            fill_price = close * (1.0 - self.slippage)

        # 手续费 #TODO: 可能需要更改，取决于券商的收费方式
        commission = abs(order.quantity * fill_price) * self.commission_rate + self.fixed_commission

        # 生成订单 id
        self._order_counter += 1
        order_id = f"paper-{self._order_counter}"

        # 生成 FillEvent
        fill = FillEvent(
            type=EventType.FILL,
            timestamp=self._engine.current_time,
            order_id=order_id,
            symbol=order.symbol,
            quantity=order.quantity,   # >0 买入, <0 卖出
            price=fill_price,
            commission=commission,
            slippage=self.slippage,
            is_partial=False,
        )

        # 放入 pending 队列，等待 Engine 拉取
        self._pending_fills.append(fill)

    def get_fills(self) -> List[FillEvent]:
        """
        Engine 每个时间步会调用它，取走本步生成的 fills。
        """
        fills = self._pending_fills
        self._pending_fills = []
        return fills
