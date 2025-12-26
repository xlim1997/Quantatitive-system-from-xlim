# backtesting/performance.py
from __future__ import annotations

import math
import pandas as pd


def _max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """
    返回：
    - max_dd: 最大回撤（负数，例如 -0.25 表示 -25%）
    - dd_duration: 最大回撤持续期（用 bar 数表示，简化）
    """
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())

    # duration：从峰值到修复回峰的最长区间（近似）
    underwater = dd < 0
    longest = 0
    cur = 0
    for x in underwater:
        if x:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return max_dd, float(longest)


def compute_performance(
    df: pd.DataFrame,
    *,
    equity_col: str = "equity",
    periods_per_year: int = 252,
    rf: float = 0.0,  # 年化无风险利率（如 0.02）
) -> dict:
    """
    df 至少需要包含 equity 列（净值）。
    periods_per_year：日频 252；分钟频可用 252*390/1min 等。
    """
    equity = df[equity_col].astype(float)
    rets = equity.pct_change().dropna()

    if len(rets) == 0:
        return {"error": "Not enough data to compute performance."}

    # CAGR
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(rets) / periods_per_year
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else float("nan")

    # Sharpe
    rf_per_period = rf / periods_per_year
    excess = rets - rf_per_period
    vol = float(rets.std()) * math.sqrt(periods_per_year)
    sharpe = float(excess.mean()) / float(rets.std()) * math.sqrt(periods_per_year) if rets.std() != 0 else float("nan")

    # Max Drawdown
    max_dd, dd_dur = _max_drawdown(equity)

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("inf")

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "max_dd_duration_bars": float(dd_dur),
        "calmar": float(calmar),
        "periods": int(len(rets)),
        "periods_per_year": int(periods_per_year),
    }
