# brokerage/base.py
"""
券商/撮合接口（Brokerage Base）
==============================

Brokerage 的职责：
- 接收订单（OrderEvent）
- 在合适的时机产生成交回报（FillEvent）
  - 回测撮合：可以立即成交（简化）
  - 实盘券商：订单会异步成交，需要轮询或回调生成 FillEvent

设计要点：
- Engine 不关心具体券商实现，只调用统一接口：
  - place_order(order)
  - get_fills() -> List[FillEvent]

此外：
- Brokerage 通常需要访问“最新行情价格”来决定成交价（回测撮合）
  因此引擎会通过 set_engine 注入引用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from core.events import OrderEvent, FillEvent

if TYPE_CHECKING:
    from core.engine import Engine


class BaseBrokerage(ABC):
    """
    所有券商/撮合实现的基类。
    """

    def __init__(self) -> None:
        # 引擎引用：用于读取当前时间、最新行情等
        self._engine: Engine | None = None

    def set_engine(self, engine: "Engine") -> None:
        """
        Engine 初始化时注入 engine 引用。
        回测撮合通常需要 engine.data_feed.last_market_data 来拿当前价格。
        """
        self._engine = engine

    @abstractmethod
    def place_order(self, order: OrderEvent) -> None:
        """
        接收一个订单请求（OrderEvent）。

        回测撮合（PaperBrokerage）：
        - 可以直接在这里生成 FillEvent 放到 pending 队列（立即成交）

        实盘券商（IBKR/Futu）：
        - 会把订单发送给券商 API
        - 成交回报由回调/轮询得到，再转成 FillEvent
        """
        ...

    @abstractmethod
    def get_fills(self) -> List[FillEvent]:
        """
        返回“自上次调用以来”新产生的所有成交回报 FillEvent。

        Engine 每个时间步会调用一次 get_fills() 来处理成交并更新 Portfolio。
        """
        ...
