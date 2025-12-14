# backtesting/engine.py
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
