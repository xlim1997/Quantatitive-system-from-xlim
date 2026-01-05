# data/ibkr_feed.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Any
import queue

from ib_insync import IB, Stock, Contract, util  # type: ignore

from core.events import MarketDataEvent, EventType, Bar
from data.base import BaseDataFeed
from data.aggregators import BarAggregator, AggConfig


@dataclass
class IBKRConnConfig_Historical:
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
        conn: IBKRConnConfig_Historical = IBKRConnConfig_Historical(),
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
                # import ipdb; ipdb.set_trace()
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


@dataclass
class IBKRConnConfig_Live:
    host: str = "127.0.0.1"
    port: int = 4001
    client_id: int = 11


class IBKRRealtimeBarFeed(BaseDataFeed):
    """
    订阅 IBKR 5秒 realtime bars，并聚合到目标周期（默认 1min）。
    """
    def __init__(
        self,
        symbols: List[str],
        conn: IBKRConnConfig_Live = IBKRConnConfig_Live(),
        agg_rule: str = "1min",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
        queue_size: int = 20000,
    ) -> None:
        self.symbols = symbols
        self.conn = conn
        self.what_to_show = what_to_show
        self.use_rth = use_rth

        self._last_market_data: Dict[str, MarketDataEvent] = {}
        self._q: "queue.Queue[Dict[str, MarketDataEvent]]" = queue.Queue(maxsize=queue_size)

        self._agg = BarAggregator(AggConfig(rule=agg_rule))

        self._ib = IB()
        self._ib.connect(conn.host, conn.port, clientId=conn.client_id, readonly=True)

        self._subs = {}
        for sym in symbols:
            c = Stock(sym, "SMART", "USD")
            # bars = self._ib.reqRealTimeBars(c, 5, what_to_show, use_rth)
            bars = self._ib.reqHistoricalData(
                c,
                endDateTime="",
                durationStr="1800 S",        # 先拿最近 30min，避免太长触发 pacing
                barSizeSetting="5 secs",
                whatToShow=what_to_show,     # "TRADES" / "MIDPOINT" 等
                useRTH=use_rth,
                formatDate=1,
                keepUpToDate=True
            )
            bars.updateEvent += self._make_on_update(sym, bars)
            self._subs[sym] = bars

    @property
    def last_market_data(self) -> Dict[str, MarketDataEvent]:
        return self._last_market_data

    def close(self) -> None:
        try:
            self._ib.disconnect()
        except Exception:
            pass

    def _make_on_update(self, sym: str, bars):
        def _on_update(*_):
            if not bars:
                return
            b = bars[-1]
            ev = MarketDataEvent(
                type=EventType.MARKET,
                timestamp=b.time,
                symbol=sym,
                bar=Bar(
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(getattr(b, "volume", 0.0) or 0.0),
                ),
                extra={"src": "ibkr_realtime_5s"},
            )

            out = self._agg.push(ev)
            if out is not None:
                try:
                    self._q.put_nowait({sym: out})
                except queue.Full:
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