# algorithm/base.py
"""
策略基类（Algorithm Base）
=========================

本文件定义了所有交易策略的“基类接口”：
- 你写的每一个具体策略，都应该继承 BaseAlgorithm。
- 策略只做一件事：在 on_data() 里，根据行情产生若干 Insight。

注意：
- 策略不直接下单（不推荐直接操作订单），
  而是通过 Insight -> PortfolioConstruction -> Risk -> Execution
  这一整套流程最终转为订单。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, TYPE_CHECKING

from core.events import MarketDataEvent
from portfolio.models import Insight

if TYPE_CHECKING:
    # 仅用于类型提示，避免循环引用
    from core.engine import Engine


class BaseAlgorithm(ABC):
    """
    所有策略的基类。

    生命周期基本流程：
    1. Engine 初始化时创建策略实例 algo = MyStrategy()
    2. Engine 调用 algo.set_engine(engine) 注入引擎引用
    3. Engine 调用 algo.initialize() 进行策略初始化（选标的、设参数等）
    4. 每个时间步，Engine 把当期行情传给 algo.on_data(...)
       -> 由策略返回 List[Insight]
    """

    def __init__(self) -> None:
        # 当前策略关注的标的列表（你可以在 initialize() 中通过 add_equity 填充）
        self.symbols: list[str] = []

        # 引擎引用：策略可以用它访问当前时间、组合等（必要时）
        self._engine: Engine | None = None

    # -----------------------------
    # 引擎注入：由 Engine 调用
    # -----------------------------
    def set_engine(self, engine: Engine) -> None:
        """
        在 Engine 构造完成后调用，为策略注入一个引擎引用。

        作用：
        - 允许策略访问 engine.current_time 等信息
        - 不建议策略直接调用 engine.emit_order() 下单，
          更推荐通过 Insight -> ExecutionModel 这条路径。
        """
        self._engine = engine

    # -----------------------------
    # 辅助方法：添加标的
    # -----------------------------
    def add_equity(self, symbol: str) -> None:
        """
        声明策略会关注某个股票/ETF/资产。

        在 initialize() 中调用：
            self.add_equity("AAPL")
            self.add_equity("MSFT")
        """
        if symbol not in self.symbols:
            self.symbols.append(symbol)

    # -----------------------------
    # 抽象方法：策略初始化
    # -----------------------------
    @abstractmethod
    def initialize(self) -> None:
        """
        初始化函数：

        一般在这里做：
        - 设定初始参数（如窗口长度、风险参数等）
        - 选择标的（调用 self.add_equity(...)）
        - 初始化内部状态（如价格队列、指标缓存）

        注意：不在这里下单，因为此时还没有行情信息。
        """
        ...

    # -----------------------------
    # 抽象方法：接收行情 -> 产生 Insights
    # -----------------------------
    @abstractmethod
    def on_data(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
        """
        每个时间步，Engine 会把当前时间的行情切片传给该方法。

        参数：
        - data: {symbol: MarketDataEvent}
          例如：
            {
              "AAPL": MarketDataEvent(...),
              "MSFT": MarketDataEvent(...),
            }

        返回：
        - 一个 Insight 列表（可以为空 list）：
          [
            Insight(symbol="AAPL", direction=UP, weight_hint=0.2),
            Insight(symbol="MSFT", direction=DOWN, weight_hint=-0.1),
          ]

        说明：
        - 不建议在这里直接“下单”（即使你可以通过 self._engine 拿到 engine），
          而是让 PortfolioConstruction / Risk / Execution 负责后续动作。
        """
        ...
