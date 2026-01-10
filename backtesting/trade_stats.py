# backtesting/trade_stats.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import pandas as pd

from core.events import FillEvent
from portfolio.state import ClosedTrade


def compute_turnover(fills: List[FillEvent], equity: pd.Series) -> Dict[str, float]:
    """
    Turnover（换手/成交额）：
    - dollar_traded = sum(|qty * price|)
    - turnover = dollar_traded / avg_equity
    """
    dollar_traded = 0.0
    for f in fills:
        dollar_traded += abs(float(f.quantity) * float(f.price))

    avg_equity = float(equity.mean()) if len(equity) else float("nan")
    turnover = (dollar_traded / avg_equity) if avg_equity and avg_equity > 0 else float("nan")
    return {
        "dollar_traded": float(dollar_traded),
        "avg_equity": float(avg_equity),
        "turnover": float(turnover),
    }


def summarize_trades(trades: List[ClosedTrade]) -> Dict[str, float]:
    """
    基于 trade_log 的统计：
    - win_rate
    - avg_holding_days
    - profit_factor
    - avg_pnl
    """
    if not trades:
        return {
            "num_trades": 0.0,
            "win_rate": float("nan"),
            "avg_holding_days": float("nan"),
            "profit_factor": float("nan"),
            "avg_pnl": float("nan"),
            "total_trade_pnl": 0.0,
        }

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate = len(wins) / len(pnls) if pnls else float("nan")
    avg_hold = float(sum(t.holding_days for t in trades) / len(trades))
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    avg_pnl = float(sum(pnls) / len(pnls))

    return {
        "num_trades": float(len(trades)),
        "win_rate": float(win_rate),
        "avg_holding_days": float(avg_hold),
        "profit_factor": float(profit_factor),
        "avg_pnl": float(avg_pnl),
        "total_trade_pnl": float(sum(pnls)),
    }


# --- append below your existing functions in backtesting/trade_stats.py ---
#TODO: Still need to debug


def _align_series(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    对齐两个序列的 index（取交集），并丢弃 NaN。
    """
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return a, b
    a2, b2 = a.align(b, join="inner")
    df = pd.concat([a2.rename("a"), b2.rename("b")], axis=1).dropna()
    return df["a"], df["b"]


def _equity_to_returns(equity: pd.Series) -> pd.Series:
    """
    equity -> 简单收益率序列（pct_change），第一项为 NaN 会被上层 dropna。
    """
    if equity is None or len(equity) < 2:
        return pd.Series(dtype=float)
    return equity.pct_change()


def _cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """
    CAGR = (end/start)^(1/years) - 1
    """
    if equity is None or len(equity) < 2:
        return float("nan")
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0:
        return float("nan")

    n_periods = len(equity) - 1
    years = float(n_periods) / float(periods_per_year) if periods_per_year > 0 else float("nan")
    if not years or years <= 0:
        return float("nan")

    return float((end / start) ** (1.0 / years) - 1.0)


def make_buy_and_hold_equity(price: pd.Series, initial_equity: float) -> pd.Series:
    """
    给单资产构造 Buy&Hold 基准净值曲线：
    bench_equity[t] = initial_equity * price[t] / price[0]
    """
    if price is None or len(price) == 0:
        return pd.Series(dtype=float)
    p0 = float(price.iloc[0])
    if p0 <= 0:
        return pd.Series(dtype=float)

    return (float(initial_equity) * (price / p0)).astype(float)


def compute_active_return_and_ir(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    相对基准（Buy&Hold）的核心指标：
    - active_total_return = (1+R_strat)/(1+R_bench) - 1
    - active_return_ann = CAGR_strat - CAGR_bench
    - tracking_error = std(r_strat - r_bench) * sqrt(periods_per_year)
    - information_ratio = annualized_mean_active / tracking_error
    """
    s, b = _align_series(strategy_equity, benchmark_equity)
    if s is None or b is None or len(s) < 2 or len(b) < 2:
        return {
            "active_total_return": float("nan"),
            "active_return_ann": float("nan"),
            "tracking_error": float("nan"),
            "information_ratio": float("nan"),
        }

    # cumulative / total active return
    r_s_total = float(s.iloc[-1] / s.iloc[0] - 1.0) if float(s.iloc[0]) != 0 else float("nan")
    r_b_total = float(b.iloc[-1] / b.iloc[0] - 1.0) if float(b.iloc[0]) != 0 else float("nan")
    active_total_return = float(((1.0 + r_s_total) / (1.0 + r_b_total)) - 1.0) if (
        math.isfinite(r_s_total) and math.isfinite(r_b_total) and (1.0 + r_b_total) != 0
    ) else float("nan")

    # annualized active return (difference of CAGR)
    cagr_s = _cagr(s, periods_per_year)
    cagr_b = _cagr(b, periods_per_year)
    active_return_ann = float(cagr_s - cagr_b) if (math.isfinite(cagr_s) and math.isfinite(cagr_b)) else float("nan")

    # IR
    rs = _equity_to_returns(s)
    rb = _equity_to_returns(b)
    a = (rs - rb).dropna()
    if len(a) < 2:
        tracking_error = float("nan")
        information_ratio = float("nan")
    else:
        std_a = float(a.std(ddof=1))
        tracking_error = float(std_a * math.sqrt(float(periods_per_year))) if periods_per_year > 0 else float("nan")
        mean_a = float(a.mean())
        annual_mean_active = float(mean_a * float(periods_per_year)) if periods_per_year > 0 else float("nan")
        information_ratio = float(annual_mean_active / tracking_error) if tracking_error and tracking_error > 0 else float("nan")

    return {
        "active_total_return": float(active_total_return),
        "active_return_ann": float(active_return_ann),
        "tracking_error": float(tracking_error),
        "information_ratio": float(information_ratio),
    }


def compute_alpha_beta(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    回归意义的 Alpha/Beta（Jensen）：
      r_s = alpha + beta * r_b + eps

    这里用等价的闭式解（含截距）：
      beta = cov(r_b, r_s) / var(r_b)
      alpha = mean(r_s) - beta * mean(r_b)

    输出：
    - alpha_daily, alpha_ann
    - beta
    - r2 (可选但很有用：你到底有多少是“跟着基准走”)
    """
    s, b = _align_series(strategy_equity, benchmark_equity)
    if s is None or b is None or len(s) < 2 or len(b) < 2:
        return {
            "alpha_daily": float("nan"),
            "alpha_ann": float("nan"),
            "beta": float("nan"),
            "r2": float("nan"),
        }

    rs = _equity_to_returns(s).dropna()
    rb = _equity_to_returns(b).dropna()

    rs, rb = _align_series(rs, rb)
    if rs is None or rb is None or len(rs) < 3:
        return {
            "alpha_daily": float("nan"),
            "alpha_ann": float("nan"),
            "beta": float("nan"),
            "r2": float("nan"),
        }

    var_b = float(rb.var(ddof=1))
    if not math.isfinite(var_b) or var_b <= 0:
        beta = float("nan")
        alpha_daily = float("nan")
        r2 = float("nan")
    else:
        beta = float(rb.cov(rs) / var_b)
        alpha_daily = float(rs.mean() - beta * rb.mean())

        # R^2
        pred = alpha_daily + beta * rb
        resid = rs - pred
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((rs - rs.mean()) ** 2).sum())
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    alpha_ann = float(alpha_daily * float(periods_per_year)) if (math.isfinite(alpha_daily) and periods_per_year > 0) else float("nan")

    return {
        "alpha_daily": float(alpha_daily),
        "alpha_ann": float(alpha_ann),
        "beta": float(beta),
        "r2": float(r2),
    }
