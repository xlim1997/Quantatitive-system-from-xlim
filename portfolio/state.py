# portfolio/state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.events import FillEvent


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0   # >0 long, <0 short
    avg_price: float = 0.0


@dataclass
class ClosedTrade:
    symbol: str
    direction: str              # "LONG" or "SHORT" (按被平掉的旧仓方向)
    entry_time: Any
    exit_time: Any
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float                  # net pnl（包含 entry/exit 分摊手续费）
    return_pct: float
    holding_days: float


@dataclass
class OpenTradeMeta:
    entry_time: Any
    direction: int              # 1 long, -1 short
    entry_commission: float = 0.0


def _days_between(t0: Any, t1: Any) -> float:
    try:
        delta = t1 - t0
        if hasattr(delta, "total_seconds"):
            return float(delta.total_seconds()) / 86400.0
    except Exception:
        pass
    return float("nan")


class Portfolio:
    def __init__(self, cash: float = 100_000.0) -> None:
        self.cash = float(cash)
        self.positions: Dict[str, Position] = {}

        # --- for risk & stats ---
        self._last_prices: Dict[str, float] = {}
        self.equity_peak: float = 0.0

        # --- trade stats ---
        self.realized_pnl: float = 0.0
        self.trade_log: List[ClosedTrade] = []
        self._open_meta: Dict[str, OpenTradeMeta] = {}

    @property
    def last_prices(self) -> Dict[str, float]:
        return self._last_prices

    def update_prices(self, last_prices: Dict[str, float]) -> None:
        self._last_prices = dict(last_prices)

    def update_equity_peak(self, equity: float) -> None:
        if equity > self.equity_peak:
            self.equity_peak = float(equity)

    def total_value(self, last_prices: Dict[str, float]) -> float:
        v = self.cash
        for sym, pos in self.positions.items():
            px = last_prices.get(sym)
            if px is None:
                continue
            v += pos.quantity * float(px)
        return float(v)

    def snapshot(self, last_prices: Dict[str, float]) -> Dict[str, Any]:
        equity = self.total_value(last_prices)
        self.update_equity_peak(equity)
        return {
            "cash": float(self.cash),
            "equity": float(equity),
            "realized_pnl": float(self.realized_pnl),
            "positions": {
                sym: {"quantity": float(p.quantity), "avg_price": float(p.avg_price)}
                for sym, p in self.positions.items()
                if p.quantity != 0
            },
        }

    def update_from_fill(self, fill: FillEvent) -> None:
        """
        净仓位记账 + 交易统计（已实现部分）

        约定：
        - fill.quantity >0 表示买入，<0 表示卖出
        - 佣金 fill.commission：会被分摊到“平仓部分”和“开仓部分”
        """
        sym = fill.symbol
        price = float(fill.price)
        trade_qty = float(fill.quantity)
        trade_qty_abs = abs(trade_qty)

        pos = self.positions.get(sym, Position(symbol=sym))
        old_qty = float(pos.quantity)
        old_avg = float(pos.avg_price)

        # 现金更新（买入扣钱，卖出加钱）
        self.cash -= price * trade_qty + float(fill.commission)

        # 取 meta（若无则构建兜底）
        meta = self._open_meta.get(sym)

        def _ensure_meta_for_existing() -> OpenTradeMeta:
            nonlocal meta
            if meta is None:
                meta = OpenTradeMeta(entry_time=fill.timestamp, direction=1 if old_qty > 0 else -1, entry_commission=0.0)
                self._open_meta[sym] = meta
            return meta

        # 1) 原来无仓：直接开仓
        if old_qty == 0:
            if trade_qty != 0:
                pos.quantity = trade_qty
                pos.avg_price = price
                self.positions[sym] = pos
                self._open_meta[sym] = OpenTradeMeta(
                    entry_time=fill.timestamp,
                    direction=1 if trade_qty > 0 else -1,
                    entry_commission=float(fill.commission),
                )
            return

        # 2) 同方向加仓
        if old_qty * trade_qty > 0:
            m = _ensure_meta_for_existing()
            new_qty = old_qty + trade_qty
            total_cost = old_avg * old_qty + price * trade_qty
            pos.quantity = new_qty
            pos.avg_price = total_cost / new_qty
            self.positions[sym] = pos
            m.entry_commission += float(fill.commission)  # 计入该笔开仓手续费（将来平仓时分摊）
            return

        # 3) 反方向成交：先平仓一部分/全部，再看是否翻方向
        m = _ensure_meta_for_existing()
        new_qty = old_qty + trade_qty

        close_qty = min(abs(old_qty), abs(trade_qty))
        if close_qty > 0 and old_avg > 0:
            # 平仓部分的已实现盈亏（gross）
            if old_qty > 0:
                pnl_gross = (price - old_avg) * close_qty
                direction = "LONG"
            else:
                pnl_gross = (old_avg - price) * close_qty
                direction = "SHORT"

            # 分摊 entry commission（按“平掉的比例”）
            entry_comm_total = float(m.entry_commission)
            entry_comm_close = entry_comm_total * (close_qty / abs(old_qty)) if abs(old_qty) > 0 else 0.0
            m.entry_commission -= entry_comm_close

            # 分摊 exit commission（按本次成交中“用于平仓的比例”）
            exit_comm_close = float(fill.commission) * (close_qty / trade_qty_abs) if trade_qty_abs > 0 else 0.0

            pnl_net = pnl_gross - entry_comm_close - exit_comm_close
            self.realized_pnl += pnl_net

            holding_days = _days_between(m.entry_time, fill.timestamp)
            denom = abs(old_avg) * close_qty
            ret_pct = (pnl_net / denom) if denom > 0 else 0.0

            self.trade_log.append(
                ClosedTrade(
                    symbol=sym,
                    direction=direction,
                    entry_time=m.entry_time,
                    exit_time=fill.timestamp,
                    quantity=close_qty,
                    entry_price=old_avg,
                    exit_price=price,
                    pnl=pnl_net,
                    return_pct=ret_pct,
                    holding_days=holding_days,
                )
            )

        # 3a) 部分平仓，方向不变（new_qty 与 old_qty 同号）
        if old_qty * new_qty > 0:
            pos.quantity = new_qty
            # avg_price 保持 old_avg（常见做法）
            pos.avg_price = old_avg
            self.positions[sym] = pos
            return

        # 3b) 完全平仓
        if new_qty == 0:
            pos.quantity = 0.0
            pos.avg_price = 0.0
            self.positions[sym] = pos
            self._open_meta.pop(sym, None)
            return

        # 3c) 翻方向：剩余部分开新仓（成本按当前成交价）
        open_comm = float(fill.commission) - (float(fill.commission) * (close_qty / trade_qty_abs) if trade_qty_abs > 0 else 0.0)

        pos.quantity = new_qty
        pos.avg_price = price
        self.positions[sym] = pos

        self._open_meta[sym] = OpenTradeMeta(
            entry_time=fill.timestamp,
            direction=1 if new_qty > 0 else -1,
            entry_commission=open_comm,
        )
