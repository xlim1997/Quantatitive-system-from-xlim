# portfolio/risk.py
"""
风险管理模型（Risk Management Model）
===================================

职责：
- 在组合构建给出的 Targets 基础上，应用风险约束：
  - 限制总杠杆/总曝险
  - 限制单票最大权重
  - 限制行业/风格暴露（以后扩展）
  - 最大回撤保护（以后扩展）

输入：
- Portfolio（组合当前状态）
- Targets（目标权重）

输出：
- Adjusted Targets（风险调整后的目标权重）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from portfolio.models import PortfolioTarget
from portfolio.state import Portfolio


class BaseRiskManagementModel(ABC):
    @abstractmethod
    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        """
        输入：组合状态 + 初始目标
        输出：风险过滤/调整后的目标
        """
        ...


class NoRiskModel(BaseRiskManagementModel):
    """不做任何风险调整：直接返回 targets。"""
    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        return targets


class MaxGrossExposureRiskModel(BaseRiskManagementModel):
    """
    限制总绝对权重（gross exposure）不超过 max_gross_exposure。

    例子：
    - max_gross_exposure = 1.0  -> 多空绝对权重之和 <= 100%
    - max_gross_exposure = 1.5  -> 允许 150% 杠杆
    """

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
    """
    限制单票最大权重，超出就截断。

    示例：
    - max_weight = 0.2 -> 单票权重不超过 20%
    """

    def __init__(self, max_weight: float = 0.2) -> None:
        self.max_weight = max_weight

    def manage_risk(self, portfolio: Portfolio, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        adjusted: List[PortfolioTarget] = []
        for t in targets:
            w = max(-self.max_weight, min(self.max_weight, t.target_percent))
            adjusted.append(PortfolioTarget(symbol=t.symbol, target_percent=w))
        return adjusted
