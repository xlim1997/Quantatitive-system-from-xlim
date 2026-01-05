# strategies/volume_contraction.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math

import pandas as pd


@dataclass
class VolumeContractionConfig:
    # --- thresholds (你要的可调参数) ---
    vol_spike_mult: float = 2.0       # 暴量倍数: V > 2 * VMA20
    vol_spike_ma: int = 20

    vol_shrink_mult: float = 0.7      # 缩量倍数: V < 0.7 * VMA5
    vol_shrink_ma: int = 5

    ma_fast: int = 10                 # 趋势: MA10 > MA20
    ma_slow: int = 20

    setup_lookback: int = 10          # 最近N天内出现过暴量日
    pullback_window: int = 8          # 暴量日后最多等待几天出现缩量整理
    max_pullback_pct: float = 0.08    # 回撤不超过8%

    # ranking / portfolio
    top_n: int = 10
    w_vol_spike: float = 0.5
    w_trend: float = 0.2
    w_pullback: float = 0.3


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def compute_signal_and_score(df: pd.DataFrame, cfg: VolumeContractionConfig) -> Optional[Dict[str, Any]]:
    """
    df: 必须包含列: ['open','high','low','close','volume']，按日期升序
    返回: None 表示没信号；否则返回 {signal, score, meta...}
    规则：两阶段（暴量 setup + 缩量回踩 pullback），信号发生在 df 最后一行（当日收盘信号），下单建议次日开盘。
    """
    if len(df) < max(cfg.ma_slow, cfg.vol_spike_ma, cfg.vol_shrink_ma) + cfg.setup_lookback + 2:
        return None

    d = df.copy()
    d["ma_fast"] = _sma(d["close"], cfg.ma_fast)
    d["ma_slow"] = _sma(d["close"], cfg.ma_slow)
    d["vma_spike"] = _sma(d["volume"], cfg.vol_spike_ma)
    d["vma_shrink"] = _sma(d["volume"], cfg.vol_shrink_ma)

    last = d.iloc[-1]
    if pd.isna(last["ma_fast"]) or pd.isna(last["ma_slow"]) or pd.isna(last["vma_spike"]) or pd.isna(last["vma_shrink"]):
        return None

    # 趋势过滤（当下趋势仍然成立）
    trend_ok = (last["ma_fast"] > last["ma_slow"]) and (last["close"] >= last["ma_slow"])
    if not trend_ok:
        return None

    # --- Phase-1: Setup day within lookback ---
    tail = d.iloc[-(cfg.setup_lookback + cfg.pullback_window + 1):]  # 留出 pullback window
    # 在 tail 里找最近的暴量日 t0
    tail = tail.copy()
    tail["vol_ratio_spike"] = tail["volume"] / tail["vma_spike"]
    tail["is_spike"] = (tail["vol_ratio_spike"] > cfg.vol_spike_mult) & (tail["close"] > tail["open"])

    spike_idx = tail.index[tail["is_spike"]].tolist()
    if not spike_idx:
        return None

    # 取最近一次暴量日作为 t0
    t0 = spike_idx[-1]
    pos_t0 = tail.index.get_loc(t0)
    pivot_high = float(tail.loc[t0, "high"])
    spike_ratio = float(tail.loc[t0, "vol_ratio_spike"])

    # --- Phase-2: Pullback/Contraction must happen AFTER t0, and TODAY is contraction day ---
    # 今天必须在 t0 之后且不超过 pullback_window
    pos_last = tail.index.get_loc(tail.index[-1])
    days_after = pos_last - pos_t0
    if not (1 <= days_after <= cfg.pullback_window):
        return None

    vol_ratio_shrink = float(last["volume"] / last["vma_shrink"])
    shrink_ok = (vol_ratio_shrink < cfg.vol_shrink_mult)

    pullback_pct = (pivot_high - float(last["close"])) / pivot_high if pivot_high > 0 else math.inf
    pullback_ok = (0.0 <= pullback_pct <= cfg.max_pullback_pct)

    if not (shrink_ok and pullback_ok):
        return None

    # --- scoring (MultiSort-like blending) ---
    # 越大越好：spike_ratio, trend_strength, (1 - pullback_pct)
    trend_strength = float((last["ma_fast"] - last["ma_slow"]) / last["ma_slow"])
    score = (
        cfg.w_vol_spike * spike_ratio +
        cfg.w_trend * trend_strength +
        cfg.w_pullback * (1.0 - pullback_pct)
    )

    return {
        "signal": "LONG_NEXT_OPEN",
        "score": score,
        "pivot_high": pivot_high,
        "spike_ratio": spike_ratio,
        "vol_ratio_shrink": vol_ratio_shrink,
        "pullback_pct": pullback_pct,
    }
