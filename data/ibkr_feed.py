# data/ibkr_feed.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Any
import queue

from ib_insync import IB, Stock, Contract, util  # type: ignore

from core.events import MarketDataEvent, EventType, Bar
from data.base import BaseDataFeed


@dataclass
class IBKRConnConfig:
    host: str = "127.0.0.1"
    port: int = 4002          # TWS paper 默认 7497；live 默认 7496 IB Gateway paper 默认 4002；live 默认 4001
    client_id: int = 1


@dataclass
class IBKRContractSpec:
    """
    用于把“字符串symbol”映射成 IBKR Contract。
    最常见：美股股票/ETF -> Stock(symbol, exchange='SMART', currency='USD')
    """
    symbol: str
    exchange: str = "SMART"
    currency: str = "USD"


def _to_contract(spec: IBKRContractSpec) -> Contract:
    # 这里只实现最常用的 Stock；期货/外汇/期权可后续扩展
    return Stock(spec.symbol, spec.exchange, spec.currency)


class IBKRHistoryBarDataFeed(BaseDataFeed):
    """
    回测用：通过 reqHistoricalData 拉历史 K 线，然后按时间迭代输出。

    要点：
    - IBKR 历史bar接口可以拉日线/分钟线等（barSizeSetting）
    - 返回的是一组 BarData 列表（bar.date, open, high, low, close, volume）
    """

    def __init__(
        self,
        contracts: Dict[str, IBKRContractSpec],     # symbol -> spec
        duration_str: str = "1 Y",                  # e.g. "1 Y", "30 D"
        bar_size: str = "1 day",                    # e.g. "1 day", "1 hour", "5 mins"
        what_to_show: str = "TRADES",               # TRADES/MIDPOINT/BID/ASK
        use_rth: bool = True,                       # 只用 RTH（常规交易时段）
        end_datetime: str = "",                     # 空字符串表示“到现在”
        conn: IBKRConnConfig = IBKRConnConfig(),
    ) -> None:
        self.contracts = contracts
        self.duration_str = duration_str
        self.bar_size = bar_size
        self.what_to_show = what_to_show
        self.use_rth = use_rth
        self.end_datetime = end_datetime
        self.conn = conn

        self._last_market_data: Dict[str, MarketDataEvent] = {}
        self._bars_by_symbol: Dict[str, list] = {}
        self._load_all()

    @property
    def last_market_data(self) -> Dict[str, MarketDataEvent]:
        return self._last_market_data

    def _load_all(self) -> None:
        ib = IB()
        ib.connect(self.conn.host, self.conn.port, clientId=self.conn.client_id)
        try:
            for sym, spec in self.contracts.items():
                c = _to_contract(spec)
                bars = ib.reqHistoricalData(
                    c,
                    endDateTime=self.end_datetime,
                    durationStr=self.duration_str,
                    barSizeSetting=self.bar_size,
                    whatToShow=self.what_to_show,
                    useRTH=self.use_rth,
                    formatDate=1,
                    keepUpToDate=False,
                )
                self._bars_by_symbol[sym] = list(bars)
        finally:
            ib.disconnect()

    def __iter__(self) -> Iterator[Dict[str, MarketDataEvent]]:
        if not self._bars_by_symbol:
            return

        # 统一时间轴：用所有 symbol 的 bar.date 并集
        all_times = sorted(set().union(*[
            {b.date for b in bars} for bars in self._bars_by_symbol.values()
        ]))

        # 为加速，先把每个 symbol 的 bars 建一个 dict：time -> bar
        idx: Dict[str, Dict[Any, Any]] = {}
        for sym, bars in self._bars_by_symbol.items():
            idx[sym] = {b.date: b for b in bars}

        for ts in all_times:
            events: Dict[str, MarketDataEvent] = {}
            for sym, d in idx.items():
                if ts not in d:
                    continue
                b = d[ts]
                bar = Bar(
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(getattr(b, "volume", 0.0) or 0.0),
                )
                events[sym] = MarketDataEvent(
                    type=EventType.MARKET,
                    timestamp=ts,
                    symbol=sym,
                    bar=bar,
                    extra=None,
                )

            if events:
                self._last_market_data = events
                yield events


class IBKRLiveBarDataFeed(BaseDataFeed):
    """
    准实时用：使用 “历史bar + keepUpToDate” 的方式持续收到最新bar更新。

    注意：
    - keepUpToDate 需要 endDateTime="" 且 barSize >= 5 seconds；
      更新会通过 historicalDataUpdate 回来。
    - 所以如果你想要 1min bar，常见做法是：
      1) 用 5 secs 更新流
      2) 在策略/数据层做 resample 到 1min
    """

    def __init__(
        self,
        contracts: Dict[str, IBKRContractSpec],
        duration_str: str = "1 D",
        bar_size: str = "5 secs",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
        conn: IBKRConnConfig = IBKRConnConfig(),
        queue_size: int = 20000,
    ) -> None:
        self.contracts = contracts
        self.duration_str = duration_str
        self.bar_size = bar_size
        self.what_to_show = what_to_show
        self.use_rth = use_rth
        self.conn = conn

        self._last_market_data: Dict[str, MarketDataEvent] = {}
        self._q: "queue.Queue[Dict[str, MarketDataEvent]]" = queue.Queue(maxsize=queue_size)

        self._ib = IB()
        self._ib.connect(self.conn.host, self.conn.port, clientId=self.conn.client_id)

        # 订阅 keepUpToDate 的历史bar（每个 symbol 一个订阅）
        self._subscriptions = {}
        for sym, spec in self.contracts.items():
            c = _to_contract(spec)
            bars = self._ib.reqHistoricalData(
                c,
                endDateTime="",
                durationStr=self.duration_str,
                barSizeSetting=self.bar_size,
                whatToShow=self.what_to_show,
                useRTH=self.use_rth,
                formatDate=1,
                keepUpToDate=True,
            )
            # bars.updateEvent 每次更新会触发（ib_insync notebook 有示例思路）
            bars.updateEvent += self._make_on_update(sym, bars)
            self._subscriptions[sym] = bars

    @property
    def last_market_data(self) -> Dict[str, MarketDataEvent]:
        return self._last_market_data

    def close(self) -> None:
        try:
            # 取消订阅（简单处理：直接断开）
            self._ib.disconnect()
        except Exception:
            pass

    def _make_on_update(self, sym: str, bars):
        def _on_update(*_):
            # bars 里最后一个元素就是最新bar（可能是“正在形成的bar”）
            if not bars:
                return
            b = bars[-1]
            ts = b.date
            bar = Bar(
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=float(getattr(b, "volume", 0.0) or 0.0),
            )
            evt = MarketDataEvent(
                type=EventType.MARKET,
                timestamp=ts,
                symbol=sym,
                bar=bar,
                extra=None,
            )
            events = {sym: evt}
            try:
                self._q.put_nowait(events)
            except queue.Full:
                # 队列满就丢弃（或你可以改成覆盖旧数据）
                pass
        return _on_update

    def __iter__(self) -> Iterator[Dict[str, MarketDataEvent]]:
        try:
            while True:
                events = self._q.get(block=True)
                if events:
                    self._last_market_data = events
                    yield events
        finally:
            self.close()
