from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from strategies.base import BaseAlgorithm
from core.events import MarketDataEvent
from portfolio.models import Insight, InsightDirection


@dataclass
class VolumeContractionConfig:
    # 交易股票池（不含SPY也可以，但建议你把SPY也喂给data_feed用作regime）
    universe: List[str]

    # === 组合控制 ===
    top_k: int = 5
    rebalance_every: int = 1
    weight_mode: str = "score"   # "equal" or "score"
    signal_lag_bars: int = 1

    # === 市场环境过滤（建议启用）===
    regime_symbol: str = "SPY"   # 需要 data_feed 里有这个symbol的历史
    regime_sma: int = 200
    enable_regime_filter: bool = True

    # === 流动性过滤 ===
    min_price: float = 5.0
    min_avg_dollar_vol20: float = 20_000_000.0

    # === 趋势过滤 ===
    sma_fast: int = 50
    sma_slow: int = 200

    # === 布林带 ===
    bb_len: int = 20
    bb_std: float = 2.0

    # === MACD ===
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # === Rolling VWAP（成交量加权滚动均价）===
    rvwap_len: int = 20

    # === 量能确认 ===
    vol_ma: int = 20
    vol_spike_mult: float = 1.5

    # === K线确认 ===
    require_candle_confirm: bool = True
    strong_bull_body_ratio: float = 0.6     # (C-O)/(H-L)
    strong_bull_close_near_high: float = 0.25  # (H-C)/(H-L)

    # === 退出 ===
    exit_on_midbb_break: bool = True
    exit_on_rvwap_break: bool = True
    exit_on_macd_hist_neg: bool = True

    # ATR trailing stop
    atr_len: int = 14
    atr_mult: float = 2.0

    # 防止刚买就卖
    min_hold_bars: int = 1

    # 退出后冷却
    cooldown_bars: int = 3

    # pending entry（防止发UP但没成交卡名额）
    pending_entry_timeout_bars: int = 2

    # 数据长度
    min_history: int = 260  # 日线建议至少一年

    # debug
    debug: bool = False
    debug_symbols: Optional[List[str]] = None


class VolumeContractionStrategy(BaseAlgorithm):
    def __init__(self, cfg: VolumeContractionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._bar_count = 0

        maxlen = max(cfg.min_history, 600)

        # OHLCV history
        self._open: Dict[str, deque] = {}
        self._high: Dict[str, deque] = {}
        self._low: Dict[str, deque] = {}
        self._close: Dict[str, deque] = {}
        self._vol: Dict[str, deque] = {}

        syms = set(cfg.universe)
        # regime_symbol 也要维护历史（即使不在交易池里）
        if cfg.enable_regime_filter and cfg.regime_symbol:
            syms.add(cfg.regime_symbol)

        for s in syms:
            self._open[s] = deque(maxlen=maxlen)
            self._high[s] = deque(maxlen=maxlen)
            self._low[s] = deque(maxlen=maxlen)
            self._close[s] = deque(maxlen=maxlen)
            self._vol[s] = deque(maxlen=maxlen)

        # 状态
        self._current_longs: Set[str] = set()     # 策略认为当前在持有的symbol
        self._entered_at_bar: Dict[str, int] = {}
        self._cooldown_until: Dict[str, int] = {}

        # 同步持仓
        self._last_qty: Dict[str, float] = {s: 0.0 for s in syms}

        # pending entries
        self._pending_since: Dict[str, int] = {}

        # trailing stop（每只票一条）
        self._trail_stop: Dict[str, float] = {}

        # entry price（用于 debug 或扩展 no-progress）
        self._entry_price: Dict[str, float] = {}

    def initialize(self) -> None:
        # 交易池都要 add_equity
        for sym in self.cfg.universe:
            self.add_equity(sym)
        # regime_symbol 也 add（需要 data_feed 支持该symbol，不交易也无所谓）
        if self.cfg.enable_regime_filter and self.cfg.regime_symbol:
            self.add_equity(self.cfg.regime_symbol)

    def _in_cooldown(self, sym: str) -> bool:
        until = self._cooldown_until.get(sym)
        return until is not None and self._bar_count <= until

    def _update_history(self, data: Dict[str, MarketDataEvent]) -> None:
        for sym, ev in data.items():
            if sym not in self._close:
                continue
            b = ev.bar
            self._open[sym].append(float(b.open))
            self._high[sym].append(float(b.high))
            self._low[sym].append(float(b.low))
            self._close[sym].append(float(b.close))
            self._vol[sym].append(float(b.volume))

    # ---------- indicators (numpy on deques) ----------
    def _np(self, d: deque) -> np.ndarray:
        return np.asarray(d, dtype=np.float64)

    def _sma_end(self, x: np.ndarray, end_i: int, w: int) -> float:
        s = end_i - w + 1
        if s < 0:
            return float("nan")
        return float(np.mean(x[s:end_i + 1]))

    def _std_end(self, x: np.ndarray, end_i: int, w: int) -> float:
        s = end_i - w + 1
        if s < 0:
            return float("nan")
        return float(np.std(x[s:end_i + 1], ddof=0))

    def _ema_series(self, x: np.ndarray, span: int) -> np.ndarray:
        # 简单 EMA 实现（O(n)），日线数据量不大够用
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(x)
        out[:] = np.nan
        if len(x) == 0:
            return out
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
        return out

    def _macd_hist_at(self, close: np.ndarray, i: int) -> Tuple[float, float]:
        # 返回 hist(i), hist(i-1) 用于判断上升
        if i <= 0:
            return float("nan"), float("nan")
        x = close[: i + 1]
        ema_fast = self._ema_series(x, self.cfg.macd_fast)
        ema_slow = self._ema_series(x, self.cfg.macd_slow)
        macd_line = ema_fast - ema_slow
        sig = self._ema_series(macd_line, self.cfg.macd_signal)
        hist = macd_line - sig
        return float(hist[-1]), float(hist[-2]) if len(hist) >= 2 else float("nan")

    def _atr_end(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, end_i: int, length: int) -> float:
        # ATR 用 TR 的 SMA
        if end_i <= 0:
            return float("nan")
        s = end_i - length + 1
        if s < 1:
            return float("nan")
        trs = []
        for t in range(s, end_i + 1):
            prev_close = close[t - 1]
            tr = max(high[t] - low[t], abs(high[t] - prev_close), abs(low[t] - prev_close))
            trs.append(tr)
        return float(np.mean(trs)) if trs else float("nan")

    def _rolling_vwap_end(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, vol: np.ndarray, end_i: int, length: int) -> float:
        s = end_i - length + 1
        if s < 0:
            return float("nan")
        tp = (high[s:end_i + 1] + low[s:end_i + 1] + close[s:end_i + 1]) / 3.0
        v = vol[s:end_i + 1]
        denom = float(np.sum(v))
        if denom <= 0:
            return float("nan")
        return float(np.sum(tp * v) / denom)

    # ---------- candle confirm ----------
    def _is_strong_bull(self, o: float, h: float, l: float, c: float) -> bool:
        rng = max(h - l, 1e-12)
        body = c - o
        if body <= 0:
            return False
        body_ratio = body / rng
        close_near_high = (h - c) / rng
        return (body_ratio >= self.cfg.strong_bull_body_ratio) and (close_near_high <= self.cfg.strong_bull_close_near_high)

    def _is_bullish_engulfing(self, o0, c0, o1, c1) -> bool:
        # prev bearish, curr bullish, curr body engulfs prev body
        if not (c0 < o0 and c1 > o1):
            return False
        return (o1 <= c0) and (c1 >= o0)

    # ---------- regime filter ----------
    def _regime_ok(self) -> bool:
        cfg = self.cfg
        if not cfg.enable_regime_filter or not cfg.regime_symbol:
            return True
        sym = cfg.regime_symbol
        if sym not in self._close:
            return True

        close = self._np(self._close[sym])
        lag = max(int(cfg.signal_lag_bars), 0)
        i = len(close) - 1 - lag
        if i <= 0:
            return False
        if len(close) < cfg.regime_sma + lag + 5:
            return False

        sma = self._sma_end(close, i, cfg.regime_sma)
        px = float(close[i])
        return (not np.isnan(sma)) and (px > sma)

    # ---------- portfolio sync ----------
    def _sync_with_portfolio(self) -> bool:
        if self._engine is None:
            return False

        closed_any = False
        port = self._engine.portfolio

        # 只同步交易池（regime_symbol 不交易）
        for sym in self.cfg.universe:
            pos = port.positions.get(sym)
            qty = float(pos.quantity) if pos is not None else 0.0
            prev_qty = self._last_qty.get(sym, 0.0)

            # 有仓：确保状态一致
            if qty != 0.0:
                self._current_longs.add(sym)
                self._entered_at_bar.setdefault(sym, self._bar_count)
                self._pending_since.pop(sym, None)
                if pos is not None and float(pos.avg_price) > 0:
                    self._entry_price[sym] = float(pos.avg_price)

            # 从有->无：可能被风险模型强平/止损
            if prev_qty != 0.0 and qty == 0.0:
                closed_any = True
                self._current_longs.discard(sym)
                self._entered_at_bar.pop(sym, None)
                self._pending_since.pop(sym, None)
                self._trail_stop.pop(sym, None)
                self._entry_price.pop(sym, None)
                self._cooldown_until[sym] = self._bar_count + int(self.cfg.cooldown_bars)

            self._last_qty[sym] = qty

        # pending timeout：发UP但没成交，释放名额
        timeout = int(self.cfg.pending_entry_timeout_bars)
        if timeout > 0:
            for sym, t0 in list(self._pending_since.items()):
                pos = port.positions.get(sym)
                qty = float(pos.quantity) if pos is not None else 0.0
                if qty == 0.0 and (self._bar_count - t0) > timeout:
                    closed_any = True
                    self._current_longs.discard(sym)
                    self._entered_at_bar.pop(sym, None)
                    self._pending_since.pop(sym, None)
                    self._trail_stop.pop(sym, None)
                    self._entry_price.pop(sym, None)
                    self._cooldown_until[sym] = self._bar_count + 1

        return closed_any

    # ---------- entry scoring ----------
    def _entry_ok_and_score(self, sym: str) -> Tuple[bool, float, Dict[str, float]]:
        cfg = self.cfg

        o = self._np(self._open[sym])
        h = self._np(self._high[sym])
        l = self._np(self._low[sym])
        c = self._np(self._close[sym])
        v = self._np(self._vol[sym])

        lag = max(int(cfg.signal_lag_bars), 0)
        i = len(c) - 1 - lag
        if i <= 1:
            return False, -1e9, {}

        need = max(cfg.min_history, cfg.sma_slow + lag + 5, cfg.bb_len + lag + 5, cfg.rvwap_len + lag + 5,
                   cfg.vol_ma + lag + 5, cfg.atr_len + lag + 5, cfg.macd_slow + cfg.macd_signal + lag + 10)
        if len(c) < need:
            return False, -1e9, {}

        # --- liquidity filter (use i) ---
        px = float(c[i])
        if px < cfg.min_price:
            return False, -1e9, {}

        # AvgDollarVol20 uses last 20 ending at i (inclusive)
        s_dv = i - 20 + 1
        if s_dv < 0:
            return False, -1e9, {}
        adv20 = float(np.mean((c[s_dv:i + 1] * v[s_dv:i + 1])))
        if adv20 < cfg.min_avg_dollar_vol20:
            return False, -1e9, {}

        # --- trend filter ---
        sma_fast = self._sma_end(c, i, cfg.sma_fast)
        sma_slow = self._sma_end(c, i, cfg.sma_slow)
        trend_ok = (not np.isnan(sma_fast)) and (not np.isnan(sma_slow)) and (sma_fast > sma_slow) and (px > sma_fast)
        if not trend_ok:
            return False, -1e9, {}

        # --- Bollinger breakout (upper) ---
        mid = self._sma_end(c, i, cfg.bb_len)
        sd = self._std_end(c, i, cfg.bb_len)
        if np.isnan(mid) or np.isnan(sd):
            return False, -1e9, {}
        upper = mid + cfg.bb_std * sd
        bb_break = px > float(upper)
        if not bb_break:
            return False, -1e9, {}

        # --- MACD hist confirm ---
        hist_i, hist_prev = self._macd_hist_at(c, i)
        macd_ok = (not np.isnan(hist_i)) and (not np.isnan(hist_prev)) and (hist_i > 0.0) and (hist_i > hist_prev)
        if not macd_ok:
            return False, -1e9, {}

        # --- volume spike ---
        s_vm = i - int(cfg.vol_ma)
        if s_vm < 0:
            return False, -1e9, {}
        vol_ma = float(np.mean(v[s_vm:i]))  # exclude i for "spike vs prev avg"
        vol_sig = float(v[i])
        vol_ok = (vol_ma > 0) and (vol_sig > cfg.vol_spike_mult * vol_ma)
        if not vol_ok:
            return False, -1e9, {}

        # --- rolling vwap filter ---
        rvwap = self._rolling_vwap_end(h, l, c, v, i, cfg.rvwap_len)
        rvwap_ok = (not np.isnan(rvwap)) and (px > rvwap)
        if not rvwap_ok:
            return False, -1e9, {}

        # --- candle confirm ---
        candle_ok = True
        if cfg.require_candle_confirm:
            strong_bull = self._is_strong_bull(float(o[i]), float(h[i]), float(l[i]), float(c[i]))
            engulf = self._is_bullish_engulfing(float(o[i - 1]), float(c[i - 1]), float(o[i]), float(c[i]))
            candle_ok = strong_bull or engulf
        if not candle_ok:
            return False, -1e9, {}

        # --- score (normalize by ATR) ---
        atr = self._atr_end(h, l, c, i, cfg.atr_len)
        if np.isnan(atr) or atr <= 1e-12:
            return False, -1e9, {}

        breakout_strength = (px - upper) / atr
        vwap_strength = (px - rvwap) / atr
        momentum_strength = hist_i / atr
        volume_strength = (vol_sig / (vol_ma + 1e-12)) - 1.0

        score = 1.0 * breakout_strength + 0.8 * momentum_strength + 0.6 * volume_strength + 0.3 * vwap_strength

        info = {
            "px": px,
            "sma_fast": float(sma_fast),
            "sma_slow": float(sma_slow),
            "upper": float(upper),
            "mid": float(mid),
            "hist": float(hist_i),
            "hist_prev": float(hist_prev),
            "vol_sig": float(vol_sig),
            "vol_ma": float(vol_ma),
            "rvwap": float(rvwap),
            "atr": float(atr),
            "score": float(score),
        }
        return True, float(score), info

    # ---------- exit signal ----------
    def _exit_signal(self, sym: str) -> Tuple[bool, Dict[str, float]]:
        cfg = self.cfg

        o = self._np(self._open[sym])
        h = self._np(self._high[sym])
        l = self._np(self._low[sym])
        c = self._np(self._close[sym])
        v = self._np(self._vol[sym])

        lag = max(int(cfg.signal_lag_bars), 0)
        i = len(c) - 1 - lag
        if i <= 1:
            return False, {}

        # midBB / upper / rvwap / macd hist
        mid = self._sma_end(c, i, cfg.bb_len)
        sd = self._std_end(c, i, cfg.bb_len)
        if np.isnan(mid) or np.isnan(sd):
            return False, {}

        rvwap = self._rolling_vwap_end(h, l, c, v, i, cfg.rvwap_len)
        hist_i, _ = self._macd_hist_at(c, i)
        px = float(c[i])

        mid_break = cfg.exit_on_midbb_break and (px < float(mid))
        rvwap_break = cfg.exit_on_rvwap_break and (not np.isnan(rvwap)) and (px < float(rvwap))
        macd_neg = cfg.exit_on_macd_hist_neg and (not np.isnan(hist_i)) and (hist_i < 0.0)

        # ATR trailing stop update (only if currently held)
        atr = self._atr_end(h, l, c, i, cfg.atr_len)
        trail_hit = False
        if (not np.isnan(atr)) and atr > 1e-12:
            new_stop = px - cfg.atr_mult * atr
            old = self._trail_stop.get(sym)
            if old is None:
                self._trail_stop[sym] = float(new_stop)
            else:
                self._trail_stop[sym] = float(max(old, new_stop))
            trail_hit = px < float(self._trail_stop[sym])

        exit_now = mid_break or rvwap_break or macd_neg or trail_hit
        info = {
            "px": px,
            "mid": float(mid),
            "rvwap": float(rvwap) if not np.isnan(rvwap) else float("nan"),
            "hist": float(hist_i) if not np.isnan(hist_i) else float("nan"),
            "trail": float(self._trail_stop.get(sym, np.nan)),
            "mid_break": float(mid_break),
            "rvwap_break": float(rvwap_break),
            "macd_neg": float(macd_neg),
            "trail_hit": float(trail_hit),
        }
        return exit_now, info

    def _strategy_exits(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
        if self._engine is None:
            return []

        cfg = self.cfg
        port = self._engine.portfolio
        outs: List[Insight] = []

        for sym, pos in port.positions.items():
            if sym not in self.cfg.universe:
                continue
            if pos.quantity == 0:
                continue

            entered = self._entered_at_bar.get(sym)
            if entered is not None and (self._bar_count - int(entered)) < int(cfg.min_hold_bars):
                continue

            exit_now, info = self._exit_signal(sym)
            if exit_now:
                outs.append(Insight(sym, InsightDirection.FLAT, 0.0))

                self._current_longs.discard(sym)
                self._entered_at_bar.pop(sym, None)
                self._pending_since.pop(sym, None)
                self._trail_stop.pop(sym, None)
                self._entry_price.pop(sym, None)
                self._cooldown_until[sym] = self._bar_count + int(cfg.cooldown_bars)

                if cfg.debug and (cfg.debug_symbols is None or sym in cfg.debug_symbols):
                    print(f"[DailyTrend] bar={self._bar_count} EXIT {sym} info={info}")

        return outs

    def _select_entries(self, need: int) -> List[Tuple[str, float, Dict[str, float]]]:
        if need <= 0:
            return []
        scored: List[Tuple[str, float, Dict[str, float]]] = []
        for sym in self.cfg.universe:
            if sym in self._current_longs:
                continue
            if self._in_cooldown(sym):
                continue
            ok, score, info = self._entry_ok_and_score(sym)
            if ok:
                scored.append((sym, score, info))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: min(need, len(scored))]

    def on_data(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
        self._bar_count += 1
        self._update_history(data)

        # 0) 同步真实持仓变化（风险模型强平、pending超时等）
        closed_by_other = self._sync_with_portfolio()

        # 1) 先做退出（任何日都允许退出）
        exit_ins = self._strategy_exits(data)

        # 2) 是否允许开新仓（regime filter）
        allow_new = self._regime_ok()

        # 3) rebalance 或者今天有退出导致空位 -> 立即补仓
        do_rebalance = (self._bar_count % int(self.cfg.rebalance_every) == 0)
        exited_today = (len(exit_ins) > 0) or closed_by_other
        need_slots = int(self.cfg.top_k) - len(self._current_longs)
        need_refill = exited_today and (need_slots > 0)

        if (not allow_new) or ((not do_rebalance) and (not need_refill)):
            return exit_ins

        entries = self._select_entries(need=need_slots)

        out = exit_ins[:]
        for sym, score, info in entries:
            w = score if self.cfg.weight_mode == "score" else 1.0
            out.append(Insight(sym, InsightDirection.UP, float(w)))

            self._current_longs.add(sym)
            self._entered_at_bar.setdefault(sym, self._bar_count)
            self._pending_since[sym] = self._bar_count

            if self.cfg.debug and (self.cfg.debug_symbols is None or sym in self.cfg.debug_symbols):
                print(f"[DailyTrend] bar={self._bar_count} BUY {sym} w={w:.4f} info={info}")

        return out
