# # backtesting/engine.py
"""
回测引擎包装器（Backtest Engine Wrapper）
======================================

职责：
- 帮你把各模块组装成一个 Engine
- 提供一个 run() 入口，返回净值记录

为什么要有它？
- 让 main 脚本更干净
- 以后你要切换不同配置（手续费、滑点、初始资金、模型组合）更方便
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from algorithm.base import BaseAlgorithm
from core.engine import Engine, EngineRecord
from data.local_csv import LocalCSVDataFeed
from brokerage.paper import PaperBrokerage
from portfolio.state import Portfolio
from portfolio.construction import BasePortfolioConstructionModel
from portfolio.risk import BaseRiskManagementModel
from portfolio.execution import BaseExecutionModel
from data.base import BaseDataFeed

class BacktestEngine:
    def __init__(
        self,
        *,
        algorithm: BaseAlgorithm,
        data_feed: BaseDataFeed | None = None,
        symbol_to_path: Dict[str, str],
        pc_model: BasePortfolioConstructionModel,
        risk_model: BaseRiskManagementModel,
        exec_model: BaseExecutionModel,
        initial_cash: float = 100_000.0,
        slippage: float = 0.0,
        commission_rate: float = 0.0,
        fixed_commission: float = 0.0,
        keep_insights_active: bool = True,
    ) -> None:
        self.algorithm = algorithm
        if data_feed is not None:
            self.data_feed = data_feed
        else:
            assert symbol_to_path is not None
            self.data_feed = LocalCSVDataFeed(symbol_to_path)
        self.brokerage = PaperBrokerage(
            slippage=slippage,
            commission_rate=commission_rate,
            fixed_commission=fixed_commission,
        )
        self.portfolio = Portfolio(cash=initial_cash)

        self.engine = Engine(
            algorithm=self.algorithm,
            data_feed=self.data_feed,
            brokerage=self.brokerage,
            portfolio=self.portfolio,
            pc_model=pc_model,
            risk_model=risk_model,
            exec_model=exec_model,
            keep_insights_active=keep_insights_active,
        )

    def run(self) -> List[EngineRecord]:
        return self.engine.run()

    def run_to_dataframe(self) -> pd.DataFrame:
        """
        把回测记录转成 DataFrame，便于画图分析。
        """
        records = self.run()
        df = pd.DataFrame([{
            "timestamp": r.timestamp,
            "cash": r.cash,
            "equity": r.equity,
            "positions": r.positions,
        } for r in records])
        return df



# core/engine.py（替换版）
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

from algorithm.base import BaseAlgorithm
from core.events import MarketDataEvent, FillEvent, OrderEvent
from data.base import BaseDataFeed
from brokerage.base import BaseBrokerage
from portfolio.state import Portfolio
from portfolio.models import Insight, InsightDirection, PortfolioTarget
from portfolio.construction import BasePortfolioConstructionModel
from portfolio.risk import BaseRiskManagementModel
from portfolio.execution import BaseExecutionModel


@dataclass
class EngineRecord:
    timestamp: Any
    cash: float
    equity: float
    realized_pnl: float
    positions: dict


class Engine:
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

        self.keep_insights_active = keep_insights_active
        self.current_time: Any = None
        self._active_insights: Dict[str, Insight] = {}

        self.records: List[EngineRecord] = []
        self.order_log: List[OrderEvent] = []
        self.fill_log: List[FillEvent] = []

        self.algorithm.set_engine(self)
        self.brokerage.set_engine(self)
        self.exec_model.set_engine(self)

    def emit_order(self, order_event: OrderEvent) -> None:
        self.order_log.append(order_event)
        self.brokerage.place_order(order_event)

    def handle_fill(self, fill: FillEvent) -> None:
        self.fill_log.append(fill)
        self.portfolio.update_from_fill(fill)

    def _update_active_insights(self, new_insights: List[Insight]) -> None:
        if not self.keep_insights_active:
            self._active_insights = {ins.symbol: ins for ins in new_insights}
            return

        # expiry（可选）
        to_remove = []
        for sym, ins in self._active_insights.items():
            if ins.expiry is not None and self.current_time is not None:
                try:
                    if self.current_time >= ins.expiry:
                        to_remove.append(sym)
                except Exception:
                    pass
        for sym in to_remove:
            self._active_insights.pop(sym, None)

        for ins in new_insights:
            if ins.direction == InsightDirection.FLAT or ins.weight_hint == 0:
                self._active_insights.pop(ins.symbol, None)
            else:
                self._active_insights[ins.symbol] = ins

    def _get_effective_insights(self, new_insights: List[Insight]) -> List[Insight]:
        self._update_active_insights(new_insights)
        return list(self._active_insights.values())

    def run(self) -> List[EngineRecord]:
        self.algorithm.initialize()

        for market_slice in self.data_feed:
            if not market_slice:
                continue

            self.current_time = next(iter(market_slice.values())).timestamp
            last_prices = {sym: float(ev.bar.close) for sym, ev in market_slice.items()}

            # ✅ 关键：更新 portfolio.last_prices，RiskModel 可读取
            self.portfolio.update_prices(last_prices)

            # 1) insights
            new_insights = self.algorithm.on_data(market_slice) or []
            insights = self._get_effective_insights(new_insights)

            # 2) PC -> targets
            targets: List[PortfolioTarget] = self.pc_model.create_targets(self.portfolio, insights)

            # 3) Risk -> adjusted targets
            adj_targets: List[PortfolioTarget] = self.risk_model.manage_risk(self.portfolio, targets)

            # 4) Execution -> orders
            self.exec_model.execute(self.portfolio, adj_targets, last_prices)

            # 5) fills -> portfolio
            fills = self.brokerage.get_fills()
            for fill in fills:
                self.handle_fill(fill)

            snap = self.portfolio.snapshot(last_prices)
            self.records.append(
                EngineRecord(
                    timestamp=self.current_time,
                    cash=snap["cash"],
                    equity=snap["equity"],
                    realized_pnl=snap["realized_pnl"],
                    positions=snap["positions"],
                )
            )

        return self.records
