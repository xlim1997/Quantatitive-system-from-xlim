# brokerage/ibkr_brokerage.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import queue

from ib_insync import IB, Stock, MarketOrder  # type: ignore

from core.events import FillEvent, OrderEvent, EventType, OrderSide


@dataclass
class IBKRConnConfig:
    host: str = "127.0.0.1"
    port: int = 4002 # TWS paper 默认 7497；live 默认 7496 IB Gateway paper 默认 4002；live 默认 4001
    client_id: int = 21


class IBKRBrokerage:
    """
    实盘/仿真实盘 Broker（MVP）：
    - place_order: OrderEvent -> IB MarketOrder
    - fills: 通过 trade.fillEvent 回调塞队列
    """
    def __init__(self, conn: IBKRConnConfig = IBKRConnConfig(), queue_size: int = 20000) -> None:
        self.conn = conn
        self._engine = None
        self._q: "queue.Queue[FillEvent]" = queue.Queue(maxsize=queue_size)

        self.ib = IB()
        self.ib.connect(conn.host, conn.port, clientId=conn.client_id, readonly=False)

        # 可选：减少“open orders / completed orders”同步造成的卡顿 单位秒
        self.ib.RequestTimeout = 30

    def set_engine(self, engine) -> None:
        self._engine = engine

    def close(self) -> None:
        try:
            self.ib.disconnect()
        except Exception:
            pass

    def _contract(self, symbol: str):
        # 这里假设 symbol 是 "AAPL" 这种
        return Stock(symbol, "SMART", "USD")

    def place_order(self, order: OrderEvent) -> None:
        c = self._contract(order.symbol)
        qty = abs(int(order.quantity))
        if qty <= 0:
            return

        action = "BUY" if order.side == OrderSide.BUY else "SELL"
        ib_order = MarketOrder(action, qty)

        trade = self.ib.placeOrder(c, ib_order)

        # 成交回调：trade.fills 会更新
        def on_fills(trade_, fill_):
            # fill_.execution.shares, fill_.execution.price
            exec_ = fill_.execution
            shares = float(exec_.shares)
            px = float(exec_.price)
            signed_qty = shares if action == "BUY" else -shares

            evt = FillEvent(
                type=EventType.FILL,
                timestamp=self._engine.current_time if self._engine else None,
                symbol=order.symbol,
                quantity=signed_qty,
                price=px,
                commission=float(getattr(fill_, "commissionReport", None).commission) if getattr(fill_, "commissionReport", None) else 0.0,
                tag="IBKR",
            )
            try:
                self._q.put_nowait(evt)
            except queue.Full:
                pass

        trade.fillEvent += on_fills

    def get_fills(self) -> List[FillEvent]:
        fills: List[FillEvent] = []
        while True:
            try:
                fills.append(self._q.get_nowait())
            except queue.Empty:
                break
        return fills
