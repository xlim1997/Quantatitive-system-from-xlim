from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import math

from strategies.base import BaseAlgorithm
from core.events import MarketDataEvent
from portfolio.models import Insight, InsightDirection


# =========================
# 1) Fisher Transform (streaming)
# =========================

@dataclass
class FisherValue:
    ready: bool
    fisher: Optional[float] = None
    slope: Optional[float] = None
    slope_up: bool = False


class FisherTransformIndicator:
    """
    Streaming Fisher Transform (TradingView-like):
      - Normalize HL2 within rolling [LL, HH] over f_len
      - Smooth x with smooth_k
      - Fisher = 0.5 * ln((1+x)/(1-x)) with clamp
      - Slope = fisher[t] - fisher[t-slope_len]
    """

    def __init__(
        self,
        f_len: int = 9,
        smooth_k: float = 0.33,
        clamp: float = 0.999,
        slope_len: int = 1,
        slope_eps: float = 0.02,
    ):
        if f_len < 2:
            raise ValueError("f_len must be >= 2")
        if not (0.0 < smooth_k < 1.0):
            raise ValueError("smooth_k must be in (0, 1)")
        if not (0.0 < clamp < 1.0):
            raise ValueError("clamp must be in (0, 1)")
        if slope_len < 1:
            raise ValueError("slope_len must be >= 1")

        self.f_len = f_len
        self.smooth_k = smooth_k
        self.clamp = clamp
        self.slope_len = slope_len
        self.slope_eps = slope_eps

        self._hl2_win = deque(maxlen=f_len)
        self._x_prev: float = 0.0
        self._f_hist = deque(maxlen=max(16, slope_len + 4))

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def update(self, high: float, low: float) -> FisherValue:
        hl2 = 0.5 * (high + low)
        self._hl2_win.append(hl2)

        if len(self._hl2_win) < self.f_len:
            return FisherValue(ready=False)

        ll = min(self._hl2_win)
        hh = max(self._hl2_win)
        rng = max(hh - ll, 1e-12)

        v = 2.0 * (hl2 - ll) / rng - 1.0
        x = self.smooth_k * v + (1.0 - self.smooth_k) * self._x_prev
        self._x_prev = x

        x = self._clamp(x, -self.clamp, self.clamp)
        fisher = 0.5 * math.log((1.0 + x) / (1.0 - x))
        self._f_hist.append(fisher)

        if len(self._f_hist) <= self.slope_len:
            return FisherValue(ready=True, fisher=fisher)

        slope = fisher - self._f_hist[-1 - self.slope_len]
        slope_up = slope > self.slope_eps
        return FisherValue(ready=True, fisher=fisher, slope=slope, slope_up=slope_up)


# =========================
# 2) Strategy Config
# =========================

@dataclass
class FisherSlopeEmaEntryConfig:
    universe: List[str]

    # Fisher params
    f_len: int = 9
    smooth_k: float = 0.33
    fisher_clamp: float = 0.999
    fisher_slope_len: int = 1
    fisher_slope_eps: float = 0.02

    # EMA params
    ema_fast_len: int = 5
    ema_mid_len: int = 10
    ema_slow_len: int = 20

    # Keep YOUR original trend condition:
    # emaUpTrend = (EMA5 > EMA10) OR (EMA5 > EMA20)

    # EMA5 slope filter
    use_ema5_slope_filter: bool = True
    ema5_slope_len: int = 1
    ema5_slope_eps: float = 0.02

    # Execution
    execute_on_next_bar: bool = True

    # After risk-control exits to FLAT, wait N bars before allowing re-entry
    cooldown_bars: int = 0


def _ema_update(prev: Optional[float], price: float, length: int) -> float:
    alpha = 2.0 / (length + 1.0)
    return price if prev is None else (prev + alpha * (price - prev))


# =========================
# 3) Strategy (Entry only, exits handled by risk control)
# =========================

class FisherSlopeEmaLongEntryStrategy(BaseAlgorithm):
    """
    Entry only (UP), exits are handled by your risk-control module.

    TradingView-equivalent entry setup:
      emaUpTrend = (EMA5 > EMA10) OR (EMA5 > EMA20)
      fisherSlopeUp: fisher[t] - fisher[t-slopeLen] > eps
      ema5SlopeUp: ema5[t] - ema5[t-ema5SlopeLen] > eps (optional)
      longSetup = emaUpTrend AND fisherSlopeUp AND ema5SlopeUp (if enabled)

    Signal policy:
      - Only enter when currently FLAT (based on portfolio real position)
      - Fire only on setup OFF->ON edge to reduce spam
      - If execute_on_next_bar=True: confirm at bar t close, send UP on bar t+1
      - Optional cooldown after risk exit (position -> flat transition)
    """

    def __init__(self, cfg: FisherSlopeEmaEntryConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._bar_count = 0

        # indicators
        self._fisher: Dict[str, FisherTransformIndicator] = {
            sym: FisherTransformIndicator(
                f_len=cfg.f_len,
                smooth_k=cfg.smooth_k,
                clamp=cfg.fisher_clamp,
                slope_len=cfg.fisher_slope_len,
                slope_eps=cfg.fisher_slope_eps,
            )
            for sym in cfg.universe
        }

        # EMA state
        self._ema_fast: Dict[str, Optional[float]] = {sym: None for sym in cfg.universe}
        self._ema_mid: Dict[str, Optional[float]] = {sym: None for sym in cfg.universe}
        self._ema_slow: Dict[str, Optional[float]] = {sym: None for sym in cfg.universe}
        self._ema_fast_hist: Dict[str, deque] = {
            sym: deque(maxlen=max(16, cfg.ema5_slope_len + 4)) for sym in cfg.universe
        }

        # edge detection + cooldown
        self._long_setup_prev: Dict[str, bool] = {sym: False for sym in cfg.universe}
        self._prev_in_pos: Dict[str, bool] = {sym: False for sym in cfg.universe}
        self._cooldown_until: Dict[str, int] = {}

        # next-bar pending
        self._pending: Dict[str, Optional[InsightDirection]] = {sym: None for sym in cfg.universe}
        self._pending_meta: Dict[str, dict] = {sym: {} for sym in cfg.universe}

    def initialize(self) -> None:
        for sym in self.cfg.universe:
            self.add_equity(sym)

    def _in_cooldown(self, sym: str) -> bool:
        until = self._cooldown_until.get(sym)
        return until is not None and self._bar_count <= until

    def _is_in_position(self, sym: str) -> bool:
        """
        IMPORTANT: use portfolio real position because exits are done by risk-control,
        not by this strategy.
        """
        if self._engine is None:
            return False
        pos = self._engine.portfolio.positions.get(sym)
        print(f"_is_in_position: sym={sym}, pos={pos}")
        if pos is not None:
            print(f"pos.quantity={pos.quantity}")
        return (pos is not None) and (pos.quantity != 0)

    def on_data(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
        self._bar_count += 1
        insights: List[Insight] = []

        for sym in self.cfg.universe:
            ev = data.get(sym)
            if ev is None:
                continue

            bar = ev.bar
            high = float(bar.high)
            low = float(bar.low)
            close = float(bar.close)

            # (A) detect risk exit: in_pos -> flat
            in_pos_now = self._is_in_position(sym)
            in_pos_prev = self._prev_in_pos.get(sym, False)
            # if in_pos_prev and (not in_pos_now) and self.cfg.cooldown_bars > 0:
            #     self._cooldown_until[sym] = self._bar_count + int(self.cfg.cooldown_bars)
            self._prev_in_pos[sym] = in_pos_now

            # (B) next-bar execution: flush pending first (only if still flat)
            if self.cfg.execute_on_next_bar and self._pending[sym] is not None:
                # only send pending entry if still flat
                if not in_pos_now:
                    dir_ = self._pending[sym]
                    meta = self._pending_meta.get(sym, {}) or {}
                    insights.append(Insight(sym, dir_, 1.0, source="FisherSlopeEMA", meta=meta))
                self._pending[sym] = None
                self._pending_meta[sym] = {}

            # (C) if already in position, we do NOTHING (exit is risk-control)
            if in_pos_now:
                # still update prev setup for stable edge behavior
                # (optional: you can skip updating indicators to save compute,
                # but keeping update makes debugging easier)
                pass

            # (D) update EMAs
            self._ema_fast[sym] = _ema_update(self._ema_fast[sym], close, self.cfg.ema_fast_len)
            self._ema_mid[sym]  = _ema_update(self._ema_mid[sym],  close, self.cfg.ema_mid_len)
            self._ema_slow[sym] = _ema_update(self._ema_slow[sym], close, self.cfg.ema_slow_len)

            ema_fast = self._ema_fast[sym]
            ema_mid = self._ema_mid[sym]
            ema_slow = self._ema_slow[sym]
            if ema_fast is None or ema_mid is None or ema_slow is None:
                self._long_setup_prev[sym] = False
                continue

            self._ema_fast_hist[sym].append(ema_fast)

            # (E) your trend condition (DO NOT change)
            ema_up_trend = (ema_fast > ema_mid) or (ema_fast > ema_slow)

            # (F) fisher update
            fv = self._fisher[sym].update(high=high, low=low)
            if (not fv.ready) or (fv.slope is None):
                self._long_setup_prev[sym] = False
                continue
            fisher_slope_up = fv.slope_up

            # (G) EMA5 slope filter (optional)
            ema5_slope_up = True
            if self.cfg.use_ema5_slope_filter:
                hist = self._ema_fast_hist[sym]
                if len(hist) <= self.cfg.ema5_slope_len:
                    self._long_setup_prev[sym] = False
                    continue
                ema5_slope = ema_fast - hist[-1 - self.cfg.ema5_slope_len]
                ema5_slope_up = ema5_slope > self.cfg.ema5_slope_eps

            long_setup = ema_up_trend and fisher_slope_up and (ema5_slope_up if self.cfg.use_ema5_slope_filter else True)

            # (H) Entry signal: only when flat + setup OFF->ON + not in cooldown
            # long_signal = (not in_pos_now) and long_setup and long_setup and (not self._long_setup_prev[sym])
            long_signal = (not in_pos_now) and long_setup
            if long_signal and self._in_cooldown(sym):
                long_signal = False

            if long_signal:
                new_dir = InsightDirection.UP
                # ---- attach trigger conditions for dashboard ----
                hist = self._ema_fast_hist[sym]
                ema5_slope_val = None
                if self.cfg.use_ema5_slope_filter and len(hist) > self.cfg.ema5_slope_len:
                    ema5_slope_val = float(ema_fast - hist[-1 - self.cfg.ema5_slope_len])

                meta = {
                    "rule": "emaUpTrend & fisherSlopeUp" + (" & ema5SlopeUp" if self.cfg.use_ema5_slope_filter else ""),
                    "conds": {
                        "emaUpTrend": bool(ema_up_trend),
                        "fisherSlopeUp": bool(fisher_slope_up),
                        "ema5SlopeUp": bool(ema5_slope_up),
                    },
                    "values": {
                        "close": float(close),
                        "ema_fast": float(ema_fast),
                        "ema_mid": float(ema_mid),
                        "ema_slow": float(ema_slow),
                        "fisher": float(fv.fisher) if fv.fisher is not None else None,
                        "fisher_slope": float(fv.slope) if fv.slope is not None else None,
                        "fisher_slope_eps": float(self.cfg.fisher_slope_eps),
                        "ema5_slope": ema5_slope_val,
                        "ema5_slope_eps": float(self.cfg.ema5_slope_eps),
                        "f_len": int(self.cfg.f_len),
                        "smooth_k": float(self.cfg.smooth_k),
                        "ema_fast_len": int(self.cfg.ema_fast_len),
                        "ema_mid_len": int(self.cfg.ema_mid_len),
                        "ema_slow_len": int(self.cfg.ema_slow_len),
                    },
                }
                if self.cfg.execute_on_next_bar:
                    self._pending[sym] = new_dir
                    self._pending_meta[sym] = meta
                else:
                    insights.append(Insight(sym, new_dir, 1.0, source="FisherSlopeEMA", meta=meta))

            self._long_setup_prev[sym] = long_setup
            print(f"sym={sym}, close={close:.2f}, in_pos={in_pos_now}, long_setup={long_setup}, long_signal={long_signal}")
        return insights
