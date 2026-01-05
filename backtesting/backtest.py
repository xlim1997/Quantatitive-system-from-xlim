# backtesting/backtest.py
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional

import pandas as pd

from core.engine import Engine  # 这里用你“真正的 Engine”（如果你的 Engine 就在 backtesting/engine.py，就改成对应 import）
from data.base import BaseDataFeed
from data.local_csv import LocalCSVDataFeed
from brokerage.base import BaseBrokerage
from brokerage.paper import PaperBrokerage
from portfolio.state import Portfolio
from strategies.base import BaseAlgorithm
from portfolio.construction import BasePortfolioConstructionModel
from portfolio.risk import BaseRiskManagementModel
from portfolio.execution import BaseExecutionModel


class Backtest:
    """
    只负责“把回测参数 -> 组装成你的 Engine”，然后跑起来。
    Engine 本身完全不需要改。
    """

    def __init__(
        self,
        *,
        algorithm: BaseAlgorithm,
        pc_model: BasePortfolioConstructionModel,
        risk_model: BaseRiskManagementModel,
        exec_model: BaseExecutionModel,
        symbol_to_path: Optional[Dict[str, str]] = None,
        data_feed: Optional[BaseDataFeed] = None,
        brokerage: Optional[BaseBrokerage] = None,
        portfolio: Optional[Portfolio] = None,
        initial_cash: float = 100_000.0,
        slippage: float = 0.0,
        commission_rate: float = 0.0,
        fixed_commission: float = 0.0,
        keep_insights_active: bool = True,
    ):
        # DataFeed
        if data_feed is None:
            assert symbol_to_path is not None, "Either data_feed or symbol_to_path must be provided."
            data_feed = LocalCSVDataFeed(symbol_to_path)

        # Portfolio
        if portfolio is None:
            portfolio = Portfolio(cash=initial_cash)

        # Brokerage
        if brokerage is None:
            brokerage = PaperBrokerage(
                slippage=slippage,
                commission_rate=commission_rate,
                fixed_commission=fixed_commission,
            )

        # 组装你的 Engine（注意：这里完全匹配你 Engine 的签名）
        self.engine = Engine(
            algorithm=algorithm,
            data_feed=data_feed,
            brokerage=brokerage,
            portfolio=portfolio,
            pc_model=pc_model,
            risk_model=risk_model,
            exec_model=exec_model,
            keep_insights_active=keep_insights_active,
        )

        # 透出 logs 给外部统计
        self.portfolio = portfolio
        self.brokerage = brokerage

    def run(self) -> pd.DataFrame:
        records = self.engine.run()  # 你 Engine 返回 List[EngineRecord]
        df = pd.DataFrame([asdict(r) if hasattr(r, "__dict__") is False else r.__dict__ for r in records])
        if not df.empty and "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        # 把 engine 的日志透出来（如果你 engine 里有这些字段）
        self.fill_log = getattr(self.engine, "fill_log", [])
        self.order_log = getattr(self.engine, "order_log", [])
        return df
