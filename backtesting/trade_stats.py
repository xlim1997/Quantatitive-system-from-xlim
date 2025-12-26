# backtesting/trade_stats.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
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
