# portfolio/risk.py
"""
风险管理模型（Risk Management Model）
===================================

升级内容：
1) ChainRiskManagementModel：支持多个 RiskModel 串联
2) StopLossRiskModel：基于当前价格/成本的止损止盈（风控层）
3) PortfolioMaxDrawdownRiskModel：组合层最大回撤保护（触发后清仓）

说明：
- 为了不改 BaseRiskManagementModel 的签名，我们让 RiskModel 从
  portfolio.last_prices 里读取“最新价格”。Engine 每个时间步会调用
  portfolio.update_prices(last_prices) 来更新。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict

from portfolio.models import PortfolioTarget
from portfolio.state import Portfolio


class BaseRiskManagementModel(ABC):
    @abstractmethod
    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        ...


class NoRiskModel(BaseRiskManagementModel):
    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        return targets


class ChainRiskManagementModel(BaseRiskManagementModel):
    """
    链式风控：按顺序依次应用多个 risk model。
    """
    def __init__(self, models: List[BaseRiskManagementModel]) -> None:
        self.models = models

    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        out = targets
        for m in self.models:
            out = m.manage_risk(portfolio, out)
        return out


class MaxGrossExposureRiskModel(BaseRiskManagementModel):
    def __init__(self, max_gross_exposure: float = 1.0) -> None:
        self.max_gross_exposure = max_gross_exposure

    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        gross = sum(abs(t.target_percent) for t in targets)
        if gross == 0 or gross <= self.max_gross_exposure:
            return targets

        scale = self.max_gross_exposure / gross
        return [
            PortfolioTarget(symbol=t.symbol, target_percent=t.target_percent * scale)
            for t in targets
        ]


class MaxPositionWeightRiskModel(BaseRiskManagementModel):
    def __init__(self, max_weight: float = 0.2) -> None:
        self.max_weight = max_weight

    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        adjusted: List[PortfolioTarget] = []
        for t in targets:
            w = max(-self.max_weight, min(self.max_weight, t.target_percent))
            adjusted.append(PortfolioTarget(symbol=t.symbol, target_percent=w))
        return adjusted


class StopLossRiskModel(BaseRiskManagementModel):
    """
    退出规则（风控层）：
    - 对当前持仓，根据 avg_price 和 last_prices 计算浮动收益
    - 触发 stop_loss / take_profit 则强制该 symbol target=0（清仓）

    注意：
    - 这是“风险层强制退出”，不依赖策略是否还想持有
    """
    def __init__(self, stop_loss_pct: float = 0.10, take_profit_pct: float | None = None) -> None:
        self.stop_loss_pct = float(stop_loss_pct)
        self.take_profit_pct = float(take_profit_pct) if take_profit_pct is not None else None

    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        prices = portfolio.last_prices
        if not prices:
            return targets

        target_map: Dict[str, float] = {t.symbol: t.target_percent for t in targets}

        for sym, pos in portfolio.positions.items():
            if pos.quantity == 0:
                continue
            if sym not in prices:
                continue

            px = prices[sym]
            if pos.avg_price <= 0 or px <= 0:
                continue

            # long: ret = px/avg - 1 ; short: ret = avg/px - 1
            if pos.quantity > 0:
                ret = px / pos.avg_price - 1.0
            else:
                ret = pos.avg_price / px - 1.0

            stop = ret <= -self.stop_loss_pct
            take = (self.take_profit_pct is not None and ret >= self.take_profit_pct)

            if stop or take:
                target_map[sym] = 0.0  # 强制清仓

        return [PortfolioTarget(symbol=s, target_percent=w) for s, w in target_map.items()]


class PortfolioMaxDrawdownRiskModel(BaseRiskManagementModel):
    """
    组合最大回撤保护：
    - 用 portfolio.total_value(prices) 得到当前 equity
    - 与峰值 equity_peak 比较，回撤超过阈值则强制全清仓（targets 全部置 0）
    """
    def __init__(self, max_drawdown: float = 0.20) -> None:
        self.max_drawdown = float(max_drawdown)

    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        prices = portfolio.last_prices
        if not prices:
            return targets

        equity = portfolio.total_value(prices)
        portfolio.update_equity_peak(equity)

        peak = portfolio.equity_peak
        if peak <= 0:
            return targets

        dd = equity / peak - 1.0
        if dd <= -self.max_drawdown:
            # 强制清仓：把当前持仓全部 target=0
            forced = {sym: 0.0 for sym in portfolio.positions.keys()}
            # 同时也把外部 targets 清掉（避免立刻又买回）
            return [PortfolioTarget(symbol=s, target_percent=0.0) for s in forced.keys()]

        return targets
