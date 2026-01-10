# from __future__ import annotations

# from collections import deque
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Set, Tuple

# import numpy as np

# from strategies.base import BaseAlgorithm
# from core.events import MarketDataEvent
# from portfolio.models import Insight, InsightDirection


# @dataclass
# class VolumeContractionConfig:
#     universe: List[str]

#     # === 你的规则 ===
#     spike_mult: float = 2.0          # 暴量：vol > 2 * MA20(vol)
#     shrink_mult: float = 0.7         # 缩量：vol < 0.7 * MA5(vol)
#     ma_fast: int = 10                # 趋势：MA10 > MA20
#     ma_slow: int = 20

#     # 暴量检测：过去 N 天内出现过一次暴量（不含“缩量日”本身）
#     spike_vol_ma: int = 20
#     spike_lookback: int = 20

#     # 缩量均量窗口
#     shrink_vol_ma: int = 5

#     # 选股/调仓
#     rebalance_every: int = 1
#     top_k: int = 5
#     weight_mode: str = "equal"       # "equal" or "score"

#     # === 关键：避免 look-ahead ===
#     # signal_lag_bars=1：用 t-1 的日线信号，在 t 日产生 UP/FLAT（推荐）
#     # signal_lag_bars=0：用当日 close/volume 算信号并当日下单（更激进，类似你 momentum）
#     signal_lag_bars: int = 1

#     # === 退出规则（策略层）===
#     stop_loss_pct: float | None = None
#     take_profit_pct: float | None = None
#     max_holding_bars: int | None = None         # 纯日线先用 1 天持有验证逻辑
#     cooldown_bars: int = 3                   # 退出后冷却 N bars

#     # 数据要求
#     min_history: int = 80

#     # debug
#     debug: bool = False
#     debug_symbols: Optional[List[str]] = None
    
#     # === 指标退出（strategy-level signal exit）===
#     exit_on_trend_break: bool = True              # MA10 <= MA20
#     exit_on_close_below_slow_ma: bool = True      # Close < MA_slow(=20)
#     exit_on_volume_reexpand: bool = False         # 可选：放量下跌退出
#     reexpand_mult: float = 1.5                    # 放量阈值：vol > 1.5 * prev5
#     reexpand_vol_ma: int = 5

#     min_hold_bars: int = 1                        # 防止刚买就立刻被 exit 条件打掉


# class VolumeContractionStrategy(BaseAlgorithm):
#     def __init__(self, cfg: VolumeContractionConfig) -> None:
#         super().__init__()
#         self.cfg = cfg

#         self._bar_count = 0

#         # 历史：为了做 lag，我们存多一点
#         maxlen = max(cfg.min_history, 300)
#         self._closes: Dict[str, deque] = {sym: deque(maxlen=maxlen) for sym in cfg.universe}
#         self._vols: Dict[str, deque] = {sym: deque(maxlen=maxlen) for sym in cfg.universe}

#         self._current_longs: Set[str] = set()
#         self._entered_at_bar: Dict[str, int] = {}
#         self._cooldown_until: Dict[str, int] = {}

#     def initialize(self) -> None:
#         for sym in self.cfg.universe:
#             self.add_equity(sym)

#     def _in_cooldown(self, sym: str) -> bool:
#         until = self._cooldown_until.get(sym)
#         return until is not None and self._bar_count <= until

#     def _update_history(self, data: Dict[str, MarketDataEvent]) -> None:
#         for sym, ev in data.items():
#             if sym not in self._closes:
#                 continue
#             self._closes[sym].append(float(ev.bar.close))
#             # 注意：你数据里 volume 字段名如果不是 ev.bar.volume，改这里
#             self._vols[sym].append(float(ev.bar.volume))

#     def _strategy_exits(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
#         """
#         策略层退出（可选）：
#         - stop loss / take profit / max holding
#         - 触发则发 FLAT，并进入 cooldown
#         """
#         if self._engine is None:
#             return []

#         insights: List[Insight] = []
#         portfolio = self._engine.portfolio

#         for sym, pos in portfolio.positions.items():
#             if pos.quantity == 0:
#                 continue

#             # 当前价（用当期 data 的 close）
#             if sym in data:
#                 px = float(data[sym].bar.close)
#             else:
#                 px = portfolio.last_prices.get(sym)

#             if px is None or px <= 0 or pos.avg_price <= 0:
#                 continue

#             if pos.quantity > 0:
#                 ret = px / pos.avg_price - 1.0
#             else:
#                 ret = pos.avg_price / px - 1.0

#             stop = (self.cfg.stop_loss_pct is not None and ret <= -float(self.cfg.stop_loss_pct))
#             take = (self.cfg.take_profit_pct is not None and ret >= float(self.cfg.take_profit_pct))

#             max_hold = False
#             if self.cfg.max_holding_bars is not None:
#                 entered = self._entered_at_bar.get(sym)
#                 if entered is not None and (self._bar_count - entered) >= int(self.cfg.max_holding_bars):
#                     max_hold = True
#             cfg = self.cfg
#             close = np.asarray(self._closes[sym], dtype=np.float64)
#             vol = np.asarray(self._vols[sym], dtype=np.float64)

#             lag = max(int(cfg.signal_lag_bars), 0)
#             sig_i = len(close) - 1 - lag
#             if sig_i <= 0:
#                 continue

#             def sma_end(x: np.ndarray, end_i: int, w: int) -> float:
#                 s = end_i - w + 1
#                 if s < 0:
#                     return float("nan")
#                 return float(np.mean(x[s:end_i + 1]))

#             ma_fast = sma_end(close, sig_i, cfg.ma_fast)
#             ma_slow = sma_end(close, sig_i, cfg.ma_slow)
#             px = float(close[sig_i])
#             px_prev = float(close[sig_i - 1])

#             trend_break = (not np.isnan(ma_fast)) and (not np.isnan(ma_slow)) and (ma_fast <= ma_slow)
#             price_break = (not np.isnan(ma_slow)) and (px < ma_slow)

#             vol_reexpand = False
#             if cfg.exit_on_volume_reexpand:
#                 # 用 sig_i 之前的 reexpand_vol_ma 日均量（不含 sig_i）
#                 s0 = sig_i - int(cfg.reexpand_vol_ma)
#                 if s0 >= 0:
#                     v_prev = float(np.mean(vol[s0:sig_i]))
#                     if v_prev > 0:
#                         vol_reexpand = (float(vol[sig_i]) > cfg.reexpand_mult * v_prev) and (px < px_prev)

#             exit_now = (
#                 (cfg.exit_on_trend_break and trend_break) or
#                 (cfg.exit_on_close_below_slow_ma and price_break) or
#                 (cfg.exit_on_volume_reexpand and vol_reexpand)
#             )
#             if stop or take or max_hold or exit_now:
#                 insights.append(Insight(sym, InsightDirection.FLAT, 0.0))
#                 self._cooldown_until[sym] = self._bar_count + int(self.cfg.cooldown_bars)
#                 self._current_longs.discard(sym)
#                 self._entered_at_bar.pop(sym, None)

#         return insights
#     # def _exit_signal(self, sym: str) -> Tuple[bool, Dict[str, float]]:
#     #     """
#     #     指标退出：只用历史（可控 lag），不依赖 portfolio.avg_price。
#     #     返回：exit?, info
#     #     """
#     #     cfg = self.cfg
#     #     close = np.asarray(self._closes[sym], dtype=np.float64)
#     #     vol = np.asarray(self._vols[sym], dtype=np.float64)

#     #     lag = max(int(cfg.signal_lag_bars), 0)
#     #     sig_i = len(close) - 1 - lag
#     #     if sig_i <= 0:
#     #         return False, {}

#     #     def sma_end(x: np.ndarray, end_i: int, w: int) -> float:
#     #         s = end_i - w + 1
#     #         if s < 0:
#     #             return float("nan")
#     #         return float(np.mean(x[s:end_i + 1]))

#     #     ma_fast = sma_end(close, sig_i, cfg.ma_fast)
#     #     ma_slow = sma_end(close, sig_i, cfg.ma_slow)
#     #     px = float(close[sig_i])
#     #     px_prev = float(close[sig_i - 1])

#     #     trend_break = (not np.isnan(ma_fast)) and (not np.isnan(ma_slow)) and (ma_fast <= ma_slow)
#     #     price_break = (not np.isnan(ma_slow)) and (px < ma_slow)

#     #     vol_reexpand = False
#     #     if cfg.exit_on_volume_reexpand:
#     #         # 用 sig_i 之前的 reexpand_vol_ma 日均量（不含 sig_i）
#     #         s0 = sig_i - int(cfg.reexpand_vol_ma)
#     #         if s0 >= 0:
#     #             v_prev = float(np.mean(vol[s0:sig_i]))
#     #             if v_prev > 0:
#     #                 vol_reexpand = (float(vol[sig_i]) > cfg.reexpand_mult * v_prev) and (px < px_prev)

#     #     exit_now = (
#     #         (cfg.exit_on_trend_break and trend_break) or
#     #         (cfg.exit_on_close_below_slow_ma and price_break) or
#     #         (cfg.exit_on_volume_reexpand and vol_reexpand)
#     #     )

#     #     info = {
#     #         "ma_fast": float(ma_fast) if not np.isnan(ma_fast) else float("nan"),
#     #         "ma_slow": float(ma_slow) if not np.isnan(ma_slow) else float("nan"),
#     #         "px": px,
#     #         "trend_break": float(trend_break),
#     #         "price_break": float(price_break),
#     #         "vol_reexpand": float(vol_reexpand),
#     #     }
#     #     return exit_now, info

#     def _signal_score(self, sym: str) -> Tuple[bool, float, Dict[str, float]]:
#         """
#         计算“暴量→缩量→趋势”信号（严格可控 lag）。
#         返回：
#           ok, score, debug_info
#         """
#         cfg = self.cfg
#         close = np.asarray(self._closes[sym], dtype=np.float64)
#         vol = np.asarray(self._vols[sym], dtype=np.float64)

#         # 用 lag 控制：lag=1 表示最后一根用于信号的是“昨天”
#         lag = int(cfg.signal_lag_bars)
#         if lag < 0:
#             lag = 0

#         # 至少要能取到 “信号日 = -1-lag”
#         # 以及 MA、rolling 等足够
#         need = max(cfg.min_history, cfg.ma_slow + lag + 2, cfg.spike_vol_ma + cfg.spike_lookback + lag + 5)
#         if len(close) < need or len(vol) < need:
#             return False, -1e9, {}

#         # 信号日索引（相对数组末尾）
#         sig_i = len(vol) - 1 - lag
#         # “缩量日”就是 sig_i
#         v_sig = vol[sig_i]

#         # === 趋势：MA10 > MA20（以 sig_i 收盘为最后一天）===
#         def sma_end(x: np.ndarray, end_i: int, w: int) -> float:
#             s = end_i - w + 1
#             if s < 0:
#                 return float("nan")
#             return float(np.mean(x[s:end_i + 1]))

#         ma_fast = sma_end(close, sig_i, cfg.ma_fast)
#         ma_slow = sma_end(close, sig_i, cfg.ma_slow)
#         trend_ok = (not np.isnan(ma_fast)) and (not np.isnan(ma_slow)) and (ma_fast > ma_slow)

#         # === 缩量：v_sig < 0.7 * MA5(volume)（MA5 不包含 v_sig）===
#         # 用 sig_i 之前的 5 天平均
#         s0 = sig_i - cfg.shrink_vol_ma
#         s1 = sig_i
#         if s0 < 0:
#             return False, -1e9, {}
#         vol_prev5 = float(np.mean(vol[s0:s1]))
#         shrink_ok = (vol_prev5 > 0) and (v_sig < cfg.shrink_mult * vol_prev5)

#         # === 暴量：在 sig_i 之前的 spike_lookback 天里出现过一次暴量 ===
#         # 暴量判定：vol[d] > 2 * MA20(vol)（MA20 用 d 之前的20天，不含 d）
#         spike_strength = 0.0
#         spike_ok = False
#         start = sig_i - cfg.spike_lookback
#         end = sig_i  # 不包含 sig_i（缩量日不当成暴量日）
#         start = max(start, cfg.spike_vol_ma)  # 保证有足够的历史算 MA20
#         for d in range(start, end):
#             ma20_prev = float(np.mean(vol[d - cfg.spike_vol_ma:d]))
#             if ma20_prev <= 0:
#                 continue
#             r = float(vol[d] / ma20_prev)
#             if r > spike_strength:
#                 spike_strength = r
#         spike_ok = spike_strength > cfg.spike_mult

#         ok = trend_ok and shrink_ok and spike_ok

#         # score：暴量越强、缩量越明显（v_sig越小）越好
#         shrink_ratio = float(v_sig / (vol_prev5 + 1e-12))
#         score = float(spike_strength / max(shrink_ratio, 1e-6))

#         info = {
#             "ma_fast": float(ma_fast),
#             "ma_slow": float(ma_slow),
#             "vol_sig": float(v_sig),
#             "vol_prev5": float(vol_prev5),
#             "spike_strength": float(spike_strength),
#             "shrink_ratio": float(shrink_ratio),
#             "score": float(score),
#         }
#         # print(f"info[{sym}]: {info}")
#         return ok, score, info

#     def on_data(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
#         self._bar_count += 1
#         self._update_history(data)
#         # 先退出（允许非 rebalance 日也退出）
#         exit_ins = self._strategy_exits(data)

#         # rebalance 决策前：如果 signal_lag_bars=1，历史里最后一根应当是“昨天”，
#         # 所以我们要在生成信号之前先不把今天 append 进去。
#         # 若 signal_lag_bars=0，则需要先 append 今天，这样信号用“今天”。
#         if self._bar_count % self.cfg.rebalance_every != 0:
#             return exit_ins

#         # if self._bar_count % self.cfg.rebalance_every != 0:
#         #     # 非 rebalance 日：只做退出
#         #     if int(self.cfg.signal_lag_bars) != 0:
#         #         self._update_history(data)
#         #     return exit_ins

#         # 计算候选
#         scored: List[Tuple[str, float, Dict[str, float]]] = []
#         for sym in self.cfg.universe:
#             if self._in_cooldown(sym):
#                 continue
#             if sym in self._current_longs:
#                 continue  # 已持有就不重复开
#             ok, score, info = self._signal_score(sym)
#             if ok:
#                 scored.append((sym, score, info))

#         scored.sort(key=lambda x: x[1], reverse=True)
#         picks = scored[: min(self.cfg.top_k, len(scored))]

#         insights: List[Insight] = exit_ins[:]

#         if picks:
#             for sym, score, info in picks:
#                 w = score if self.cfg.weight_mode == "score" else 1.0
#                 insights.append(Insight(sym, InsightDirection.UP, float(w)))
#                 self._current_longs.add(sym)
#                 self._entered_at_bar.setdefault(sym, self._bar_count)

#                 if self.cfg.debug and (self.cfg.debug_symbols is None or sym in self.cfg.debug_symbols):
#                     print(f"[VC] bar={self._bar_count} BUY {sym} w={w:.4f} info={info}")

#         # rebalance 完后，若 lag=1，此时才把今天 append 进历史（给明天用）
#         # if int(self.cfg.signal_lag_bars) != 0:
#         #     self._update_history(data)
#         # import ipdb; ipdb.set_trace()
#         return insights


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
    universe: List[str]

    # === Entry: 暴量→缩量→趋势 ===
    spike_mult: float = 2.0          # 暴量：vol > 2 * MA20(vol)
    shrink_mult: float = 0.7         # 缩量：vol < 0.7 * MA5(vol)
    ma_fast: int = 10                # 趋势：MA10 > MA20
    ma_slow: int = 20

    spike_vol_ma: int = 20
    spike_lookback: int = 20
    shrink_vol_ma: int = 5

    # === 组合持仓控制 ===
    rebalance_every: int = 5
    top_k: int = 5
    weight_mode: str = "equal"       # "equal" or "score"

    # === 避免 look-ahead ===
    signal_lag_bars: int = 1         # 用 t-1 的信号在 t 下单（推荐）

    # === 策略层退出（价格/趋势/量能）===
    cooldown_bars: int = 3
    min_hold_bars: int = 1

    # 趋势反转退出
    exit_on_trend_break: bool = True              # MA_fast <= MA_slow
    exit_on_close_below_slow_ma: bool = True      # close < MA_slow

    # “量能没了”退出：成交量持续低于自身均量
    exit_on_volume_dry: bool = True
    volume_dry_ma: int = 20                       # 基准均量窗口
    volume_dry_mult: float = 0.55                 # vol < 0.55 * MA20(vol) 认为“量能干”
    volume_dry_bars: int = 5                     # 连续N天量能干 -> 退出

    # “买入后没走强/没突破”退出：避免一直躺着
    exit_on_no_progress: bool = True
    no_progress_bars: int = 10                    # 持有>=N天仍不走强 -> 退出
    min_progress_pct: float = 0.02                # 至少涨 2%（相对 entry_price）

    # 可选：放量下跌退出（你原先的）
    exit_on_volume_reexpand: bool = False
    reexpand_mult: float = 1.5
    reexpand_vol_ma: int = 5

    # 也可以在策略层做止损止盈（你现在主要用 RiskModel 的 StopLossRiskModel）
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    max_holding_bars: int | None = None

    # 数据要求
    min_history: int = 80

    # pending entry（防止因 risk/资金/最小成交额导致“发了UP但没成交”卡住名额）
    pending_entry_timeout_bars: int = 2

    # debug
    debug: bool = False
    debug_symbols: Optional[List[str]] = None


class VolumeContractionStrategy(BaseAlgorithm):
    def __init__(self, cfg: VolumeContractionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self._bar_count = 0

        maxlen = max(cfg.min_history, 400)
        self._closes: Dict[str, deque] = {sym: deque(maxlen=maxlen) for sym in cfg.universe}
        self._vols: Dict[str, deque] = {sym: deque(maxlen=maxlen) for sym in cfg.universe}

        # “策略认为当前应该持有/处于UP信号”的集合（与 portfolio 实际持仓允许有1-2 bar 的偏差）
        self._current_longs: Set[str] = set()

        # 进入/退出状态
        self._entered_at_bar: Dict[str, int] = {}
        self._cooldown_until: Dict[str, int] = {}

        # 追踪真实成交后的 entry price（用于 no_progress 等）
        self._entry_price: Dict[str, float] = {}

        # 量能干枯计数器
        self._vol_dry_count: Dict[str, int] = {}

        # 识别“风控强平/外部卖出”导致的持仓变化
        self._last_qty: Dict[str, float] = {sym: 0.0 for sym in cfg.universe}

        # pending entries：发了UP但可能没成交
        self._pending_since: Dict[str, int] = {}

    def initialize(self) -> None:
        for sym in self.cfg.universe:
            self.add_equity(sym)

    def _in_cooldown(self, sym: str) -> bool:
        until = self._cooldown_until.get(sym)
        return until is not None and self._bar_count <= until

    def _update_history(self, data: Dict[str, MarketDataEvent]) -> None:
        for sym, ev in data.items():
            if sym not in self._closes:
                continue
            self._closes[sym].append(float(ev.bar.close))
            self._vols[sym].append(float(ev.bar.volume))

    def _sma_end(self, x: np.ndarray, end_i: int, w: int) -> float:
        s = end_i - w + 1
        if s < 0:
            return float("nan")
        return float(np.mean(x[s:end_i + 1]))

    def _sync_with_portfolio(self) -> bool:
        """
        同步真实持仓变化：如果被风控/强平卖出了，释放名额并允许补仓。
        返回：是否检测到“仓位被平掉”（用于触发立即补仓）。
        """
        if self._engine is None:
            return False

        closed_any = False
        port = self._engine.portfolio

        for sym in self.cfg.universe:
            pos = port.positions.get(sym)
            qty = float(pos.quantity) if pos is not None else 0.0

            # 如果成交了（有真实持仓），且我们没有 entry_price，就记录一下
            if qty != 0.0:
                if sym not in self._current_longs:
                    self._current_longs.add(sym)
                if sym not in self._entered_at_bar:
                    self._entered_at_bar[sym] = self._bar_count
                if sym not in self._entry_price and pos is not None and float(pos.avg_price) > 0:
                    self._entry_price[sym] = float(pos.avg_price)
                # 成交了就不是 pending
                self._pending_since.pop(sym, None)

            prev_qty = self._last_qty.get(sym, 0.0)

            # 从有仓->无仓：说明已经被卖掉（可能是风险模型/止损/最大回撤等）
            if prev_qty != 0.0 and qty == 0.0:
                closed_any = True
                self._current_longs.discard(sym)
                self._entered_at_bar.pop(sym, None)
                self._entry_price.pop(sym, None)
                self._vol_dry_count.pop(sym, None)
                self._pending_since.pop(sym, None)
                self._cooldown_until[sym] = self._bar_count + int(self.cfg.cooldown_bars)

            self._last_qty[sym] = qty

        # pending timeout：发了UP但一直没成交 -> 释放名额
        timeout = int(self.cfg.pending_entry_timeout_bars)
        if timeout > 0 and self._engine is not None:
            port = self._engine.portfolio
            for sym, t0 in list(self._pending_since.items()):
                pos = port.positions.get(sym)
                qty = float(pos.quantity) if pos is not None else 0.0
                if qty == 0.0 and (self._bar_count - t0) > timeout:
                    # 释放名额，避免卡死
                    self._current_longs.discard(sym)
                    self._entered_at_bar.pop(sym, None)
                    self._entry_price.pop(sym, None)
                    self._vol_dry_count.pop(sym, None)
                    self._pending_since.pop(sym, None)
                    self._cooldown_until[sym] = self._bar_count + 1
                    closed_any = True

        return closed_any

    def _exit_signal(self, sym: str) -> Tuple[bool, Dict[str, float]]:
        """
        只用历史（含 lag）做“趋势/量能/无进展”退出判断，避免前视。
        """
        cfg = self.cfg
        close = np.asarray(self._closes[sym], dtype=np.float64)
        vol = np.asarray(self._vols[sym], dtype=np.float64)

        lag = max(int(cfg.signal_lag_bars), 0)
        sig_i = len(close) - 1 - lag
        if sig_i <= 1:
            return False, {}

        ma_fast = self._sma_end(close, sig_i, cfg.ma_fast)
        ma_slow = self._sma_end(close, sig_i, cfg.ma_slow)
        px = float(close[sig_i])
        px_prev = float(close[sig_i - 1])

        trend_break = (not np.isnan(ma_fast)) and (not np.isnan(ma_slow)) and (ma_fast <= ma_slow)
        price_break = (not np.isnan(ma_slow)) and (px < ma_slow)

        # 量能干枯：vol[sig_i] 低于自身 MA(volume_dry_ma)
        vol_dry = False
        dry_cnt = self._vol_dry_count.get(sym, 0)
        if cfg.exit_on_volume_dry:
            s0 = sig_i - int(cfg.volume_dry_ma)
            if s0 >= 0:
                v_base = float(np.mean(vol[s0:sig_i]))  # 不含 sig_i
                if v_base > 0:
                    vol_dry = float(vol[sig_i]) < float(cfg.volume_dry_mult) * v_base
            # 更新计数
            if vol_dry:
                dry_cnt += 1
            else:
                dry_cnt = 0
            self._vol_dry_count[sym] = dry_cnt

        # 无进展：持有 >= no_progress_bars，但价格仍未达到 entry*(1+min_progress)
        no_progress = False
        if cfg.exit_on_no_progress:
            entered = self._entered_at_bar.get(sym)
            entry_px = self._entry_price.get(sym)
            if entered is not None and entry_px is not None and entry_px > 0:
                hold_bars = self._bar_count - int(entered)
                if hold_bars >= int(cfg.no_progress_bars):
                    no_progress = px < entry_px * (1.0 + float(cfg.min_progress_pct))

        # 放量下跌退出（可选）
        vol_reexpand = False
        if cfg.exit_on_volume_reexpand:
            s0 = sig_i - int(cfg.reexpand_vol_ma)
            if s0 >= 0:
                v_prev = float(np.mean(vol[s0:sig_i]))
                if v_prev > 0:
                    vol_reexpand = (float(vol[sig_i]) > float(cfg.reexpand_mult) * v_prev) and (px < px_prev)

        exit_now = (
            (cfg.exit_on_trend_break and trend_break) or
            (cfg.exit_on_close_below_slow_ma and price_break) or
            (cfg.exit_on_volume_dry and (dry_cnt >= int(cfg.volume_dry_bars))) or
            (cfg.exit_on_no_progress and no_progress) or
            (cfg.exit_on_volume_reexpand and vol_reexpand)
        )

        info = {
            "px": px,
            "ma_fast": float(ma_fast) if not np.isnan(ma_fast) else float("nan"),
            "ma_slow": float(ma_slow) if not np.isnan(ma_slow) else float("nan"),
            "trend_break": float(trend_break),
            "price_break": float(price_break),
            "vol_dry_cnt": float(dry_cnt),
            "no_progress": float(no_progress),
            "vol_reexpand": float(vol_reexpand),
        }
        return exit_now, info

    def _strategy_exits(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
        """
        策略层退出：
        - (可选) stop loss / take profit / max holding bars
        - 指标退出：趋势反转 / 量能干枯 / 无进展 / (可选) 放量下跌
        """
        if self._engine is None:
            return []

        insights: List[Insight] = []
        portfolio = self._engine.portfolio
        cfg = self.cfg

        for sym, pos in portfolio.positions.items():
            if pos.quantity == 0:
                continue

            # 保护：至少持有 min_hold_bars 才允许触发策略退出（避免刚买就卖）
            entered = self._entered_at_bar.get(sym)
            if entered is not None:
                if (self._bar_count - int(entered)) < int(cfg.min_hold_bars):
                    continue

            # 当前价（用于止损止盈）
            px_now = None
            if sym in data:
                px_now = float(data[sym].bar.close)
            else:
                px_now = portfolio.last_prices.get(sym)

            if px_now is None or px_now <= 0 or float(pos.avg_price) <= 0:
                continue

            # 记录 entry price（如果还没记）
            if sym not in self._entry_price and float(pos.avg_price) > 0:
                self._entry_price[sym] = float(pos.avg_price)

            # PnL-based exits（可选）
            ret = (px_now / float(pos.avg_price) - 1.0) if pos.quantity > 0 else (float(pos.avg_price) / px_now - 1.0)

            stop = (cfg.stop_loss_pct is not None and ret <= -float(cfg.stop_loss_pct))
            take = (cfg.take_profit_pct is not None and ret >= float(cfg.take_profit_pct))
            import ipdb; ipdb.set_trace()
            max_hold = False
            if cfg.max_holding_bars is not None and entered is not None:
                if (self._bar_count - int(entered)) >= int(cfg.max_holding_bars):
                    max_hold = True

            sig_exit, sig_info = self._exit_signal(sym)

            if stop or take or max_hold or sig_exit:
                insights.append(Insight(sym, InsightDirection.FLAT, 0.0))

                # 状态清理 + cooldown
                self._cooldown_until[sym] = self._bar_count + int(cfg.cooldown_bars)
                self._current_longs.discard(sym)
                self._entered_at_bar.pop(sym, None)
                self._entry_price.pop(sym, None)
                self._vol_dry_count.pop(sym, None)
                self._pending_since.pop(sym, None)

                if cfg.debug and (cfg.debug_symbols is None or sym in cfg.debug_symbols):
                    reason = {
                        "stop": stop, "take": take, "max_hold": max_hold, "signal_exit": sig_exit
                    }
                    print(f"[VC] bar={self._bar_count} EXIT {sym} reason={reason} info={sig_info}")

        return insights

    def _signal_score(self, sym: str) -> Tuple[bool, float, Dict[str, float]]:
        """
        Entry信号：暴量→缩量→趋势（严格可控 lag）。
        返回：ok, score, info
        """
        cfg = self.cfg
        close = np.asarray(self._closes[sym], dtype=np.float64)
        vol = np.asarray(self._vols[sym], dtype=np.float64)

        lag = max(int(cfg.signal_lag_bars), 0)

        need = max(
            cfg.min_history,
            cfg.ma_slow + lag + 2,
            cfg.spike_vol_ma + cfg.spike_lookback + lag + 5,
            cfg.volume_dry_ma + lag + 5,
        )
        if len(close) < need or len(vol) < need:
            return False, -1e9, {}

        sig_i = len(vol) - 1 - lag
        v_sig = float(vol[sig_i])

        ma_fast = self._sma_end(close, sig_i, cfg.ma_fast)
        ma_slow = self._sma_end(close, sig_i, cfg.ma_slow)
        trend_ok = (not np.isnan(ma_fast)) and (not np.isnan(ma_slow)) and (ma_fast > ma_slow)

        # 缩量：v_sig < shrink_mult * MA(shrink_vol_ma)（不含 sig_i）
        s0 = sig_i - int(cfg.shrink_vol_ma)
        if s0 < 0:
            return False, -1e9, {}
        vol_prev = float(np.mean(vol[s0:sig_i]))
        shrink_ok = (vol_prev > 0) and (v_sig < float(cfg.shrink_mult) * vol_prev)

        # 暴量：在 sig_i 前 lookback 出现过 vol[d] > spike_mult * MA20_prev
        spike_strength = 0.0
        start = max(sig_i - int(cfg.spike_lookback), int(cfg.spike_vol_ma))
        for d in range(start, sig_i):
            ma_prev = float(np.mean(vol[d - int(cfg.spike_vol_ma):d]))
            if ma_prev <= 0:
                continue
            r = float(vol[d] / ma_prev)
            if r > spike_strength:
                spike_strength = r
        spike_ok = spike_strength > float(cfg.spike_mult)

        ok = trend_ok and shrink_ok and spike_ok

        # score：暴量越强、缩量越明显越好
        shrink_ratio = float(v_sig / (vol_prev + 1e-12))
        score = float(spike_strength / max(shrink_ratio, 1e-6))

        info = {
            "ma_fast": float(ma_fast),
            "ma_slow": float(ma_slow),
            "vol_sig": float(v_sig),
            "vol_prev": float(vol_prev),
            "spike_strength": float(spike_strength),
            "shrink_ratio": float(shrink_ratio),
            "score": float(score),
        }
        return ok, score, info

    def _select_entries(self, need: int) -> List[Tuple[str, float, Dict[str, float]]]:
        if need <= 0:
            return []

        scored: List[Tuple[str, float, Dict[str, float]]] = []
        for sym in self.cfg.universe:
            if self._in_cooldown(sym):
                continue
            if sym in self._current_longs:
                continue
            ok, score, info = self._signal_score(sym)
            if ok:
                scored.append((sym, score, info))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: min(need, len(scored))]

    def on_data(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
        self._bar_count += 1
        self._update_history(data)

        # 1) 同步真实持仓（处理 risk 强平 / 外部卖出 / pending 卡死）
        closed_by_other = self._sync_with_portfolio()

        # 2) 先做策略退出（允许非 rebalance 日也退出）
        exit_ins = self._strategy_exits(data)

        # 3) 是否需要补仓：
        #    - rebalance 日（定期评估新增）
        #    - 或者今日发生退出/强平导致名额空出来 -> 立刻补
        do_rebalance = (self._bar_count % int(self.cfg.rebalance_every) == 0)
        exited_today = (len(exit_ins) > 0) or closed_by_other
        need_slots = int(self.cfg.top_k) - len(self._current_longs)
        need_refill = exited_today and (need_slots > 0)

        if (not do_rebalance) and (not need_refill):
            return exit_ins

        # 4) 只补足到 top_k，不会超仓
        entries = self._select_entries(need=need_slots)

        insights: List[Insight] = exit_ins[:]

        for sym, score, info in entries:
            w = float(score) if self.cfg.weight_mode == "score" else 1.0
            insights.append(Insight(sym, InsightDirection.UP, w))

            self._current_longs.add(sym)
            self._entered_at_bar.setdefault(sym, self._bar_count)
            self._pending_since[sym] = self._bar_count  # 等待成交确认（sync会清掉）

            if self.cfg.debug and (self.cfg.debug_symbols is None or sym in self.cfg.debug_symbols):
                print(f"[VC] bar={self._bar_count} BUY {sym} w={w:.4f} info={info}")

        return insights
