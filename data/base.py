# data/base.py
"""
数据源接口（DataFeed Base）
=========================

本文件定义了 DataFeed 的抽象基类：
- DataFeed 的职责是“按时间顺序产出行情事件”。
- 无论数据来自 CSV、本地数据库、还是实时 API，都应该实现这个接口。

核心约定：
- DataFeed 是一个可迭代对象（__iter__），每次迭代产出：
    Dict[str, MarketDataEvent]
  即：某个时间点，多个 symbol 的行情切片。
- DataFeed 需要维护一个 last_market_data 属性，供撮合等模块获取最新价格。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator

from core.events import MarketDataEvent


class BaseDataFeed(ABC):
    """
    抽象基类：所有数据源都要继承它。

    例子：
    - LocalCSVDataFeed       : 用于回测，从本地 CSV 文件加载历史数据
    - LiveAPIDataFeed (以后) : 用于实盘，从券商 / 行情 API 拉实时数据
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, MarketDataEvent]]:
        """
        使 DataFeed 可迭代。

        每次迭代返回：
            {symbol: MarketDataEvent, ...}
        对应同一个时间点的多标的行情。

        用法示例：
            for slice in data_feed:
                # slice 是一个 dict，里面是多个 symbol 的 MarketDataEvent
                ...
        """
        ...

    @property
    @abstractmethod
    def last_market_data(self) -> Dict[str, MarketDataEvent]:
        """
        返回最近一次迭代产生的行情数据切片。

        用途：
        - 撮合（Brokerage）需要获取当前价格来决定成交价
        - 组合估值、日志记录也会用到

        例如在 PaperBrokerage 中：
            last_close = engine.data_feed.last_market_data[symbol].bar.close
        """
        ...
