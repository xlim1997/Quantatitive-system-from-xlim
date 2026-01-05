# portfolio/kill_switch.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from portfolio.models import PortfolioTarget
from portfolio.state import Portfolio
from portfolio.risk import BaseRiskManagementModel


@dataclass
class KillSwitchConfig:
    max_daily_loss: float = 2000.0     # 当日亏损超过阈值（货币单位），停止交易并清仓
    max_total_loss: float = 5000.0     # 总亏损超过阈值（相对起始资金）
    hard_halt: bool = True


class KillSwitchRiskModel(BaseRiskManagementModel):
    """
    MVP：基于 equity 和 portfolio.equity_peak / 起始资金做保护。
    建议你后续扩展：按“日”滚动重置、按交易日历、发通知等。
    """
    def __init__(self, cfg: KillSwitchConfig, initial_cash: float) -> None:
        self.cfg = cfg
        self.initial_cash = float(initial_cash)

    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        prices = portfolio.last_prices
        if not prices:
            return targets

        equity = portfolio.total_value(prices)
        total_pnl = equity - self.initial_cash

        # 总亏损保护
        if total_pnl <= -self.cfg.max_total_loss:
            portfolio.halt_trading = True
            return [PortfolioTarget(symbol=s, target_percent=0.0) for s in portfolio.positions.keys()]

        return targets
