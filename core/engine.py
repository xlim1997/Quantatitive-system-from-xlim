# core/engine.py
"""
引擎（Engine）
=============

Engine 是整个框架的“中枢调度器”，负责把各个模块按顺序串起来：

每个时间步（一个 bar）：
1) DataFeed 提供 MarketDataEvent 切片：{symbol: MarketDataEvent}
2) Algorithm 根据行情生成 Insights（观点）
3) PortfolioConstructionModel 把 Insights 转成 Targets（目标权重）
4) RiskManagementModel 对 Targets 做风险调整
5) ExecutionModel 把 Targets 转成 OrderEvent，并交给 Brokerage
6) Brokerage 返回 FillEvent（成交），Portfolio 根据 FillEvent 更新持仓和现金
7) Engine 记录组合净值等结果（用于回测分析）

额外：我们实现一个“Active Insights”机制（轻量版 Lean InsightManager）
- Algorithm 不必每个 bar 都重复输出相同的 Insight
- Engine 会缓存 active_insights，直到 Insight 被 FLAT 覆盖或过期
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from algorithm.base import BaseAlgorithm
from core.events import MarketDataEvent, FillEvent
from data.base import BaseDataFeed
from brokerage.base import BaseBrokerage
from portfolio.state import Portfolio
from portfolio.models import Insight, InsightDirection, PortfolioTarget
from portfolio.construction import BasePortfolioConstructionModel
from portfolio.risk import BaseRiskManagementModel
from portfolio.execution import BaseExecutionModel


@dataclass
class EngineRecord:
    """
    用于记录回测/实盘运行过程中的关键数据（最小版本）。
    你可以后续扩展：记录订单、成交、指标、风险暴露等。
    """
    timestamp: Any
    cash: float
    equity: float
    positions: dict


class Engine:
    """
    主引擎：把 Algorithm / PC / Risk / Execution / Brokerage / Portfolio 串起来。
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        data_feed: BaseDataFeed,
        brokerage: BaseBrokerage,
        portfolio: Portfolio,
        pc_model: BasePortfolioConstructionModel,
        risk_model: BaseRiskManagementModel,
        exec_model: BaseExecutionModel,
        *,
        keep_insights_active: bool = True,
    ) -> None:
        self.algorithm = algorithm
        self.data_feed = data_feed
        self.brokerage = brokerage
        self.portfolio = portfolio
        self.pc_model = pc_model
        self.risk_model = risk_model
        self.exec_model = exec_model

        # 是否缓存 Insight（推荐开启）
        self.keep_insights_active = keep_insights_active

        # 运行时状态
        self.current_time: Any = None
        self._active_insights: Dict[str, Insight] = {}  # symbol -> Insight
        self.records: List[EngineRecord] = []

        # 反向注入：让策略/券商/执行知道 engine
        self.algorithm.set_engine(self)
        self.brokerage.set_engine(self)
        self.exec_model.set_engine(self)

    # ---------------------------------------------------------------------
    # 给 ExecutionModel 调用：发单给 Brokerage
    # ---------------------------------------------------------------------
    def emit_order(self, order_event) -> None:
        """
        ExecutionModel 调用该方法来发出订单。
        Engine 再把订单交给 Brokerage（回测撮合 or 实盘券商）处理。
        """
        self.brokerage.place_order(order_event)

    # ---------------------------------------------------------------------
    # 处理成交：更新 Portfolio
    # ---------------------------------------------------------------------
    def handle_fill(self, fill: FillEvent) -> None:
        """
        Brokerage 返回 FillEvent，Engine 负责把它交给 Portfolio 更新状态。
        """
        self.portfolio.update_from_fill(fill)

    # ---------------------------------------------------------------------
    # Insight 缓存（轻量 InsightManager）
    # ---------------------------------------------------------------------
    def _update_active_insights(self, new_insights: List[Insight]) -> None:
        """
        更新 active insights：
        - 如果 keep_insights_active=False，则不缓存（直接用新 insights）
        - 如果 keep_insights_active=True：
          - new insight 覆盖同 symbol 的旧 insight
          - FLAT 会删除该 symbol 的 insight（表示不再持有观点）
          - expiry（可选）：过期后自动移除
        """
        if not self.keep_insights_active:
            self._active_insights = {ins.symbol: ins for ins in new_insights}
            return

        # 先移除过期（如果 Insight.expiry 可比较）
        to_remove = []
        for sym, ins in self._active_insights.items():
            if ins.expiry is not None and self.current_time is not None:
                try:
                    if self.current_time >= ins.expiry:
                        to_remove.append(sym)
                except Exception:
                    # 如果无法比较，就忽略 expiry
                    pass
        for sym in to_remove:
            self._active_insights.pop(sym, None)

        # 再合并新 insights
        for ins in new_insights:
            if ins.direction == InsightDirection.FLAT or ins.weight_hint == 0:
                # FLAT：视为撤销观点
                self._active_insights.pop(ins.symbol, None)
            else:
                self._active_insights[ins.symbol] = ins

    def _get_effective_insights(self, new_insights: List[Insight]) -> List[Insight]:
        """
        返回本时间步用于组合构建的 insights（可能是 active insights）。
        """
        self._update_active_insights(new_insights)
        return list(self._active_insights.values())

    # ---------------------------------------------------------------------
    # 主循环：跑完整个 DataFeed
    # ---------------------------------------------------------------------
    def run(self) -> List[EngineRecord]:
        """
        运行整个引擎，遍历 DataFeed，返回记录列表（可用于回测分析）。
        """
        # 1) 初始化策略
        self.algorithm.initialize()

        # 2) 遍历历史行情（或实盘事件流）
        for market_slice in self.data_feed:
            if not market_slice:
                continue

            # 当前时间：取 slice 里任意一个 symbol 的 timestamp
            self.current_time = next(iter(market_slice.values())).timestamp

            # 最新价格：用于估值 & 执行模型换算目标股数
            last_prices = {sym: ev.bar.close for sym, ev in market_slice.items()}

            # 2.1 策略生成 Insights
            new_insights = self.algorithm.on_data(market_slice) or []

            # 2.2 形成“有效 insights”（含缓存）
            insights = self._get_effective_insights(new_insights)

            # 2.3 组合构建：Insights -> Targets
            targets: List[PortfolioTarget] = self.pc_model.create_targets(self.portfolio, insights)

            # 2.4 风险管理：Targets -> Adjusted Targets
            adj_targets: List[PortfolioTarget] = self.risk_model.manage_risk(self.portfolio, targets)

            # 2.5 执行：Adjusted Targets -> Orders（由 ExecutionModel 发单）
            self.exec_model.execute(self.portfolio, adj_targets, last_prices)

            # 2.6 拉取成交并更新组合
            fills = self.brokerage.get_fills()
            for fill in fills:
                self.handle_fill(fill)

            # 2.7 记录结果（现金、净值、仓位）
            snapshot = self.portfolio.snapshot(last_prices)
            self.records.append(
                EngineRecord(
                    timestamp=self.current_time,
                    cash=snapshot["cash"],
                    equity=snapshot["equity"],
                    positions=snapshot["positions"],
                )
            )

        return self.records
