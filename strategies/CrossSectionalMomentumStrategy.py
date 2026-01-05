# strategies/momentum/main.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from strategies.base import BaseAlgorithm
from core.events import MarketDataEvent
from portfolio.models import Insight, InsightDirection


@dataclass
class MomentumConfig:
    universe: List[str]
    lookback: int = 60
    rebalance_every: int = 5
    top_k: int = 10
    bottom_k: int = 0
    weight_mode: str = "equal"          # "equal" or "score"
    min_mom_abs: float = 0.0

    # --- Exit rules (strategy-level) ---
    stop_loss_pct: float | None = 0.10
    take_profit_pct: float | None = None
    max_holding_bars: int | None = None
    cooldown_bars: int = 10             # 触发退出后，至少 N bars 不再买回


class CrossSectionalMomentumStrategy(BaseAlgorithm):
    def __init__(self, cfg: MomentumConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self._bar_count = 0
        self._closes: Dict[str, deque] = {sym: deque(maxlen=cfg.lookback + 1) for sym in cfg.universe}

        self._current_longs: Set[str] = set()
        self._current_shorts: Set[str] = set()

        # 记录进入 bar index，用于 max_holding_bars
        self._entered_at_bar: Dict[str, int] = {}

        # cooldown：sym -> until_bar（含）
        self._cooldown_until: Dict[str, int] = {}

    def initialize(self) -> None:
        for sym in self.cfg.universe:
            self.add_equity(sym)

    def _update_history(self, data: Dict[str, MarketDataEvent]) -> None:
        for sym, ev in data.items():
            if sym in self._closes:
                self._closes[sym].append(float(ev.bar.close))

    def _momentum(self, sym: str) -> Optional[float]:
        q = self._closes[sym]
        if len(q) < self.cfg.lookback + 1:
            return None
        past = q[0]
        now = q[-1]
        if past <= 0:
            return None
        return now / past - 1.0

    def _in_cooldown(self, sym: str) -> bool:
        until = self._cooldown_until.get(sym)
        return until is not None and self._bar_count <= until

    def _strategy_exits(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
        """
        策略层退出规则：
        - 基于当前持仓的 avg_price 和当前价格
        - 触发则发 FLAT，并写入 cooldown
        """
        if self._engine is None:
            return []

        insights: List[Insight] = []
        portfolio = self._engine.portfolio

        for sym, pos in portfolio.positions.items():
            if pos.quantity == 0:
                continue

            # 当前价：优先用当期 data，没有则用 portfolio.last_prices
            if sym in data:
                px = float(data[sym].bar.close)
            else:
                px = portfolio.last_prices.get(sym)
            if px is None or px <= 0 or pos.avg_price <= 0:
                continue

            # long / short 的收益率定义
            if pos.quantity > 0:
                ret = px / pos.avg_price - 1.0
            else:
                ret = pos.avg_price / px - 1.0

            stop = (self.cfg.stop_loss_pct is not None and ret <= -float(self.cfg.stop_loss_pct))
            take = (self.cfg.take_profit_pct is not None and ret >= float(self.cfg.take_profit_pct))

            # 最大持仓 bars（用进入 bar index 近似）
            max_hold = False
            if self.cfg.max_holding_bars is not None:
                entered = self._entered_at_bar.get(sym)
                if entered is not None and (self._bar_count - entered) >= int(self.cfg.max_holding_bars):
                    max_hold = True

            if stop or take or max_hold:
                insights.append(Insight(sym, InsightDirection.FLAT, 0.0))
                self._cooldown_until[sym] = self._bar_count + int(self.cfg.cooldown_bars)
                # 从当前集合移除
                self._current_longs.discard(sym)
                self._current_shorts.discard(sym)
                self._entered_at_bar.pop(sym, None)

        return insights

    def on_data(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
        self._bar_count += 1
        self._update_history(data)

        # 先做退出（即使不是 rebalance bar，也允许退出）
        exit_ins = self._strategy_exits(data)

        # 非 rebalance 时刻：只发退出信号即可
        if self._bar_count % self.cfg.rebalance_every != 0:
            return exit_ins

        # 1) 计算动量
        moms = []
        for sym in self.cfg.universe:
            if self._in_cooldown(sym):
                continue
            m = self._momentum(sym)
            if m is None:
                continue
            if abs(m) < self.cfg.min_mom_abs:
                continue
            moms.append((sym, m))

        if len(moms) == 0:
            # 没信号：把当前集合清掉
            ins: List[Insight] = exit_ins[:]
            for sym in (self._current_longs | self._current_shorts):
                ins.append(Insight(sym, InsightDirection.FLAT, 0.0))
                self._entered_at_bar.pop(sym, None)
            self._current_longs.clear()
            self._current_shorts.clear()
            return ins

        moms.sort(key=lambda x: x[1], reverse=True)
        mom_map = dict(moms)

        top_k = min(self.cfg.top_k, len(moms))
        new_longs = set(sym for sym, _ in moms[:top_k])

        new_shorts: Set[str] = set()
        if self.cfg.bottom_k > 0:
            bottom_k = min(self.cfg.bottom_k, len(moms) - top_k)
            if bottom_k > 0:
                new_shorts = set(sym for sym, _ in moms[-bottom_k:])
        new_shorts -= new_longs

        # 2) 生成 insights（包含 FLAT 清理旧缓存）
        insights: List[Insight] = exit_ins[:]

        to_flat = (self._current_longs - new_longs) | (self._current_shorts - new_shorts)
        for sym in to_flat:
            insights.append(Insight(sym, InsightDirection.FLAT, 0.0))
            self._entered_at_bar.pop(sym, None)

        for sym in new_longs:
            w = abs(mom_map[sym]) if self.cfg.weight_mode == "score" else 1.0
            insights.append(Insight(sym, InsightDirection.UP, float(w)))
            self._entered_at_bar.setdefault(sym, self._bar_count)

        for sym in new_shorts:
            w = abs(mom_map[sym]) if self.cfg.weight_mode == "score" else 1.0
            insights.append(Insight(sym, InsightDirection.DOWN, float(w)))
            self._entered_at_bar.setdefault(sym, self._bar_count)

        self._current_longs = new_longs
        self._current_shorts = new_shorts
        return insights
