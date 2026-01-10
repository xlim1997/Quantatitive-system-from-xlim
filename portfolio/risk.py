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

    额外：
    - last_actions: 收集子 risk model 在本 bar 触发的动作（用于可视化复盘）。
      这些动作不会改变 manage_risk 的返回签名，只是 side-channel。
    """
    def __init__(self, models: List[BaseRiskManagementModel]) -> None:
        self.models = models
        self.last_actions: List[Dict] = []

    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        out = targets
        self.last_actions = []
        for m in self.models:
            out = m.manage_risk(portfolio, out)

            # side-channel: collect child actions if any
            acts = getattr(m, "last_actions", None)
            if acts:
                try:
                    self.last_actions.extend(list(acts))
                except Exception:
                    pass
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
        self.last_actions: List[Dict] = []

    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        prices = portfolio.last_prices
        # reset side-channel actions each bar
        self.last_actions = []
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
                self.last_actions.append({
                    "symbol": sym,
                    "action": "FORCE_FLAT",
                    "reason": "STOP_LOSS" if stop else "TAKE_PROFIT",
                    "ret": float(ret),
                    "avg_price": float(pos.avg_price),
                    "price": float(px),
                    "qty": float(pos.quantity),
                    "stop_loss_pct": float(self.stop_loss_pct),
                    "take_profit_pct": float(self.take_profit_pct) if self.take_profit_pct is not None else None,
                })
        # 3) 返回时用稳定顺序：先按原 targets 的顺序，再补上当前持仓但不在 targets 里的
        ordered_syms = [t.symbol for t in targets]
        for sym in portfolio.positions.keys():
            if sym not in target_map:
                # 如果策略没给该 symbol target，就不动它（不返回也行，取决于你执行层如何解释“缺失”）
                # 如果你执行层把“缺失”当“不调整”，这里可以不加
                pass
            if sym not in ordered_syms and sym in target_map:
                ordered_syms.append(sym)

        out: List[PortfolioTarget] = []
        for sym in ordered_syms:
            if sym in target_map:
                out.append(PortfolioTarget(symbol=sym, target_percent=target_map[sym]))
        return out
        # return [PortfolioTarget(symbol=s, target_percent=w) for s, w in target_map.items()]


class PortfolioMaxDrawdownRiskModel(BaseRiskManagementModel):
    """
    组合最大回撤保护（可选：是否允许恢复交易）

    原版逻辑的问题：
      - equity_peak 一直不重置
      - 一旦触发 dd 阈值并清仓，组合净值通常无法“自然恢复”
        => dd 始终处于阈值之下 => targets 永远被强制 0 => 之后再也买不回

    这里提供三种模式（用参数控制）：

    1) halt_trading_on_trigger=True:
         - 触发后强制清仓，并设置 portfolio.halt_trading=True
         - 之后引擎会跳过执行（真正的 kill-switch，永不恢复，需人工解除）

    2) reset_peak_on_trigger=True (推荐用于回测/自动恢复):
         - 触发当步强制清仓
         - 同时把 equity_peak 重置为当前 equity
         - 下一根 bar 起允许重新交易（dd=0 从新开始）

    3) 两者都 False:
         - 只在 dd 超阈值时强制清仓，但不重置 peak
         - 等组合净值恢复到阈值之上才允许重新交易（对“清仓后无法恢复”的场景不适用）
    """

    def __init__(
        self,
        max_drawdown: float = 0.20,
        *,
        halt_trading_on_trigger: bool = False,
        reset_peak_on_trigger: bool = True,
    ) -> None:
        self.max_drawdown = float(max_drawdown)
        self.halt_trading_on_trigger = bool(halt_trading_on_trigger)
        self.reset_peak_on_trigger = bool(reset_peak_on_trigger)
        self.last_actions: List[Dict] = []

    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        prices = portfolio.last_prices
        # reset side-channel actions each bar
        self.last_actions = []
        if not prices:
            return targets

        equity = portfolio.total_value(prices)
        portfolio.update_equity_peak(equity)

        peak = portfolio.equity_peak
        if peak <= 0:
            return targets

        dd = equity / peak - 1.0

        if dd <= -self.max_drawdown:
            # side-channel: for dashboard timeline
            self.last_actions.append({
                "symbol": "",
                "action": "FORCE_FLAT_ALL",
                "reason": "PORTFOLIO_MAX_DRAWDOWN",
                "dd": float(dd),
                "max_drawdown": float(self.max_drawdown),
                "equity": float(equity),
                "peak": float(peak),
                "halt_trading_on_trigger": bool(self.halt_trading_on_trigger),
                "reset_peak_on_trigger": bool(self.reset_peak_on_trigger),
            })
            # 1) 强制全清仓（包括外部 targets 和当前持仓）
            all_syms = set(portfolio.positions.keys()) | {t.symbol for t in targets}
            forced = [PortfolioTarget(symbol=s, target_percent=0.0) for s in sorted(all_syms)]

            # 2) 可选：触发后停机（之后永远不执行，除非你手动把 portfolio.halt_trading=False）
            if self.halt_trading_on_trigger:
                setattr(portfolio, "halt_trading", True)

            # 3) 可选：重置 peak，使得清仓后还能重新开仓
            if self.reset_peak_on_trigger:
                portfolio.equity_peak = float(equity)

            return forced

        return targets

