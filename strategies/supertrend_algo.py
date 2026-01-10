from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from strategies.base import BaseAlgorithm
from core.events import MarketDataEvent
from portfolio.models import Insight, InsightDirection


# =========================
# 1) Supertrend Indicator
# =========================

@dataclass
class SupertrendValue:
    ready: bool
    supertrend: Optional[float] = None
    direction: Optional[int] = None   # +1 uptrend, -1 downtrend
    atr: Optional[float] = None
    final_upper: Optional[float] = None
    final_lower: Optional[float] = None
    flip: bool = False


class SupertrendIndicator:
    """
    Streaming Supertrend (TradingView-like):
      - ATR uses Wilder's smoothing (RMA)
      - Bands: HL2 ± factor * ATR
      - Final bands + switch logic (standard)
    """

    def __init__(self, atr_length: int = 10, factor: float = 3.0):
        if atr_length <= 0:
            raise ValueError("atr_length must be positive")
        if factor <= 0:
            raise ValueError("factor must be positive")

        self.atr_length = atr_length
        self.factor = factor

        self._tr_window = deque(maxlen=atr_length)
        self._prev_close: Optional[float] = None

        self._atr: Optional[float] = None
        self._final_upper: Optional[float] = None
        self._final_lower: Optional[float] = None
        self._supertrend: Optional[float] = None
        self._direction: Optional[int] = None  # +1 / -1

    @staticmethod
    def _true_range(high: float, low: float, prev_close: Optional[float]) -> float:
        if prev_close is None:
            return float(high - low)
        return float(max(high - low, abs(high - prev_close), abs(low - prev_close)))

    def update(self, high: float, low: float, close: float) -> SupertrendValue:
        tr = self._true_range(high, low, self._prev_close)
        self._tr_window.append(tr)

        # ATR warmup + Wilder smoothing
        if self._atr is None:
            if len(self._tr_window) < self.atr_length:
                self._prev_close = close
                return SupertrendValue(ready=False)
            self._atr = sum(self._tr_window) / self.atr_length
        else:
            n = self.atr_length
            self._atr = (self._atr * (n - 1) + tr) / n

        hl2 = (high + low) / 2.0
        basic_upper = hl2 + self.factor * self._atr
        basic_lower = hl2 - self.factor * self._atr

        # final bands
        if self._final_upper is None or self._final_lower is None:
            final_upper = basic_upper
            final_lower = basic_lower
        else:
            prev_final_upper = self._final_upper
            prev_final_lower = self._final_lower
            prev_close = self._prev_close if self._prev_close is not None else close

            final_upper = (
                basic_upper
                if (basic_upper < prev_final_upper) or (prev_close > prev_final_upper)
                else prev_final_upper
            )
            final_lower = (
                basic_lower
                if (basic_lower > prev_final_lower) or (prev_close < prev_final_lower)
                else prev_final_lower
            )

        # switch logic
        prev_supertrend = self._supertrend
        prev_direction = self._direction

        if prev_supertrend is None:
            if close >= final_lower:
                supertrend = final_lower
                direction = +1
            else:
                supertrend = final_upper
                direction = -1
        else:
            # downtrend regime if prev supertrend == prev final upper
            if prev_supertrend == self._final_upper:
                if close <= final_upper:
                    supertrend = final_upper
                    direction = -1
                else:
                    supertrend = final_lower
                    direction = +1
            else:
                # uptrend regime
                if close >= final_lower:
                    supertrend = final_lower
                    direction = +1
                else:
                    supertrend = final_upper
                    direction = -1

        flip = (prev_direction is not None) and (direction != prev_direction)

        # commit
        self._final_upper = final_upper
        self._final_lower = final_lower
        self._supertrend = supertrend
        self._direction = direction
        self._prev_close = close

        return SupertrendValue(
            ready=True,
            supertrend=supertrend,
            direction=direction,
            atr=self._atr,
            final_upper=final_upper,
            final_lower=final_lower,
            flip=flip,
        )


# =========================
# 2) Strategy (Momentum-style)
# =========================

@dataclass
class SupertrendConfig:
    universe: List[str]
    atr_length: int = 10
    factor: float = 3.0

    # ✅ long-only 开关就在这里
    long_only: bool = True

    # 避免 look-ahead：bar t 收盘确认 flip，bar t+1 再发信号
    execute_on_next_bar: bool = True

    # --- 可选：策略层退出规则（和你的 momentum 一样）---
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    max_holding_bars: int | None = None
    cooldown_bars: int = 0


class SupertrendStrategy(BaseAlgorithm):
    """
    入场/出场（按 Supertrend flip）：
      - flip 到 uptrend：UP
      - flip 到 downtrend：long_only -> FLAT；否则 DOWN

    额外（可选）：
      - stop_loss / take_profit / max_holding / cooldown
      - 风格/结构对齐你的 CrossSectionalMomentumStrategy
    """

    def __init__(self, cfg: SupertrendConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self._bar_count = 0

        self._ind: Dict[str, SupertrendIndicator] = {
            sym: SupertrendIndicator(cfg.atr_length, cfg.factor) for sym in cfg.universe
        }
        self._bar_count = 0
        # 当前“策略认为”的持仓状态（仅用于 max_holding/cooldown bookkeeping）
        self._current_longs: Set[str] = set()
        self._current_shorts: Set[str] = set()

        self._entered_at_bar: Dict[str, int] = {}
        self._cooldown_until: Dict[str, int] = {}

        # 下一根 bar 执行的 pending 信号
        self._pending: Dict[str, Optional[InsightDirection]] = {sym: None for sym in cfg.universe}

    def initialize(self) -> None:
        for sym in self.cfg.universe:
            self.add_equity(sym)

    def _in_cooldown(self, sym: str) -> bool:
        until = self._cooldown_until.get(sym)
        return until is not None and self._bar_count <= until

    def _strategy_exits(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
        """
        和你的 momentum 一样：基于 portfolio.positions 的 avg_price + 当前价触发 FLAT
        """
        if self._engine is None:
            return []

        if (
            self.cfg.stop_loss_pct is None
            and self.cfg.take_profit_pct is None
            and self.cfg.max_holding_bars is None
        ):
            return []

        portfolio = self._engine.portfolio
        insights: List[Insight] = []

        for sym, pos in portfolio.positions.items():
            if pos.quantity == 0:
                continue

            # 当前价：优先用 data，否则用 last_prices
            if sym in data:
                px = float(data[sym].bar.close)
            else:
                px = portfolio.last_prices.get(sym)

            if px is None or px <= 0 or pos.avg_price <= 0:
                continue

            # long / short 收益率
            if pos.quantity > 0:
                ret = px / pos.avg_price - 1.0
            else:
                ret = pos.avg_price / px - 1.0

            stop = (self.cfg.stop_loss_pct is not None and ret <= -float(self.cfg.stop_loss_pct))
            take = (self.cfg.take_profit_pct is not None and ret >= float(self.cfg.take_profit_pct))

            max_hold = False
            if self.cfg.max_holding_bars is not None:
                entered = self._entered_at_bar.get(sym)
                if entered is not None and (self._bar_count - entered) >= int(self.cfg.max_holding_bars):
                    max_hold = True

            if stop or take or max_hold:
                insights.append(Insight(sym, InsightDirection.FLAT, 0.0))
                if self.cfg.cooldown_bars > 0:
                    self._cooldown_until[sym] = self._bar_count + int(self.cfg.cooldown_bars)

                self._current_longs.discard(sym)
                self._current_shorts.discard(sym)
                self._entered_at_bar.pop(sym, None)

        return insights

    def on_data(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
        self._bar_count += 1

        # 0) 先做“策略层退出”（无论是否 flip，都允许止损/止盈等）
        insights: List[Insight] = self._strategy_exits(data)

        for sym in self.cfg.universe:
            ev = data.get(sym)
            if ev is None:
                continue

            bar = ev.bar
            high = float(bar.high)
            low = float(bar.low)
            close = float(bar.close)

            # 1) 如果配置为 next-bar 执行：先把上次 pending 发出去
            if self.cfg.execute_on_next_bar and self._pending[sym] is not None:
                dir_ = self._pending[sym]
                insights.append(Insight(sym, dir_, 1.0))
                self._pending[sym] = None

                # bookkeeping
                if dir_ == InsightDirection.UP:
                    self._current_longs.add(sym)
                    self._current_shorts.discard(sym)
                    self._entered_at_bar.setdefault(sym, self._bar_count)
                elif dir_ == InsightDirection.DOWN:
                    self._current_shorts.add(sym)
                    self._current_longs.discard(sym)
                    self._entered_at_bar.setdefault(sym, self._bar_count)
                elif dir_ == InsightDirection.FLAT:
                    self._current_longs.discard(sym)
                    self._current_shorts.discard(sym)
                    self._entered_at_bar.pop(sym, None)

            # 2) 更新 supertrend
            st = self._ind[sym].update(high=high, low=low, close=close)
            if not st.ready:
                continue

            # 3) 仅在 flip 时发信号
            if not st.flip:
                continue

            # cooldown：如果刚退出过，禁止立刻买回（仅对 UP 入场有意义）
            if st.direction == +1 and self._in_cooldown(sym):
                continue

            if st.direction == +1:
                new_dir = InsightDirection.UP
            else:
                new_dir = InsightDirection.FLAT if self.cfg.long_only else InsightDirection.DOWN

            if self.cfg.execute_on_next_bar:
                self._pending[sym] = new_dir
            else:
                insights.append(Insight(sym, new_dir, 1.0))

                # bookkeeping（即时执行的情况下也更新）
                if new_dir == InsightDirection.UP:
                    self._current_longs.add(sym)
                    self._current_shorts.discard(sym)
                    self._entered_at_bar.setdefault(sym, self._bar_count)
                elif new_dir == InsightDirection.DOWN:
                    self._current_shorts.add(sym)
                    self._current_longs.discard(sym)
                    self._entered_at_bar.setdefault(sym, self._bar_count)
                elif new_dir == InsightDirection.FLAT:
                    self._current_longs.discard(sym)
                    self._current_shorts.discard(sym)
                    self._entered_at_bar.pop(sym, None)
                    if self.cfg.cooldown_bars > 0:
                        self._cooldown_until[sym] = self._bar_count + int(self.cfg.cooldown_bars)

        return insights