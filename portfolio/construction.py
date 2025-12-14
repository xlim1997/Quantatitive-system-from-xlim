# portfolio/construction.py
"""
组合构建模型（Portfolio Construction Model）
===========================================

职责：
- 把策略产生的 Insights（观点）转换成可执行的组合目标 PortfolioTargets（目标权重）。

输入：
- 当前 Portfolio（组合状态）
- Insights 列表：每个 symbol 的方向 + weight_hint

输出：
- Targets 列表：每个 symbol 的 target_percent（目标权重）

说明：
- Lean 里对应 PortfolioConstructionModel。
- 组合构建不负责风控、不负责下单，只负责“目标权重如何分配”。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from portfolio.models import Insight, InsightDirection, PortfolioTarget
from portfolio.state import Portfolio


class BasePortfolioConstructionModel(ABC):
    """
    抽象基类：所有组合构建模型都应继承它。
    """

    @abstractmethod
    def create_targets(self, portfolio: Portfolio, insights: List[Insight]) -> List[PortfolioTarget]:
        """
        把 Insights -> Targets。

        注意：
        - 这里不关心价格，不关心下单细节，仅生成目标权重。
        """
        ...


class EqualWeightLongOnlyPC(BasePortfolioConstructionModel):
    """
    最简单的组合构建：
    - 只做多（只接受 UP）
    - 把所有 UP 的标的等权分配（总权重 = 1.0）
    - FLAT/DOWN -> 不产生目标（意味着目标为 0，后续 Execution 会趋近清仓）
    """

    def create_targets(self, portfolio: Portfolio, insights: List[Insight]) -> List[PortfolioTarget]:
        ups = [ins for ins in insights if ins.direction == InsightDirection.UP]
        if not ups:
            return []

        w = 1.0 / len(ups)
        return [PortfolioTarget(symbol=ins.symbol, target_percent=w) for ins in ups]


class WeightedByHintPC(BasePortfolioConstructionModel):
    """
    更通用一点的组合构建：
    - 支持 UP / DOWN / FLAT
    - 使用 weight_hint 作为权重来源
    - 最后会做一个归一化（可选），使总绝对权重 <= 1.0

    适合：
    - 你有信号强度/因子分数，想把它映射为目标权重
    """

    def __init__(self, normalize_gross: bool = True, gross_cap: float = 1.0) -> None:
        self.normalize_gross = normalize_gross
        self.gross_cap = gross_cap

    def create_targets(self, portfolio: Portfolio, insights: List[Insight]) -> List[PortfolioTarget]:
        targets: List[PortfolioTarget] = []

        # 1) 先把 Insight 转成原始 target_percent
        for ins in insights:
            if ins.direction == InsightDirection.FLAT or ins.weight_hint == 0:
                continue

            # direction 作为符号修正（也允许 weight_hint 已经带符号）
            sign = 1.0 if ins.direction == InsightDirection.UP else -1.0
            raw = sign * abs(ins.weight_hint)

            targets.append(PortfolioTarget(symbol=ins.symbol, target_percent=raw))

        if not targets:
            return []

        # 2) 归一化总曝险（gross exposure）
        if self.normalize_gross:
            gross = sum(abs(t.target_percent) for t in targets)
            if gross > 0:
                scale = min(1.0, self.gross_cap / gross)
                targets = [
                    PortfolioTarget(symbol=t.symbol, target_percent=t.target_percent * scale)
                    for t in targets
                ]

        return targets
