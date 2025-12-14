# data/futu_feed.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Any
import queue
import threading

import pandas as pd

from futu import (
    OpenQuoteContext,
    RET_OK,
    KLType,
    AuType,
    SubType,
    CurKlineHandlerBase,
)

from core.events import MarketDataEvent, EventType, Bar
from data.base import BaseDataFeed


# --------------------------
# Utils
# --------------------------
def _isna(x: Any) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return x is None


def _safe_str(x: Any) -> Optional[str]:
    if _isna(x):
        return None
    return str(x)


def _safe_float(x: Any) -> Optional[float]:
    if _isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    if _isna(x):
        return None
    try:
        # futu 有时给的是 numpy 类型/float
        return int(float(x))
    except Exception:
        return None


def _build_extra_from_row(row: pd.Series, code: str) -> Dict[str, Any]:
    """
    按你要求的字段填充到 event.extra（缺失字段自动 None）
    """
    return {
        "code": code,
        "name": _safe_str(row.get("name")),
        "time_key": _safe_str(row.get("time_key")),
        "open": _safe_float(row.get("open")),
        "close": _safe_float(row.get("close")),
        "high": _safe_float(row.get("high")),
        "low": _safe_float(row.get("low")),
        "pe_ratio": _safe_float(row.get("pe_ratio")),
        "turnover_rate": _safe_float(row.get("turnover_rate")),
        "volume": _safe_int(row.get("volume")),
        "turnover": _safe_float(row.get("turnover")),
        "change_rate": _safe_float(row.get("change_rate")),
        "last_close": _safe_float(row.get("last_close")),
    }


# --------------------------
# Config
# --------------------------
@dataclass(frozen=True)
class FutuConnConfig:
    host: str = "127.0.0.1"
    port: int = 11111


# --------------------------
# History feed (backtest)
# --------------------------
class FutuHistoryKlineDataFeed(BaseDataFeed):
    """
    回测用：通过 request_history_kline 拉历史K线，拉完后按时间迭代输出 MarketDataEvent 切片。
    """

    def __init__(
        self,
        codes: List[str],
        start: Optional[str] = None,  # "YYYY-MM-DD"
        end: Optional[str] = None,    # "YYYY-MM-DD"
        ktype: KLType = KLType.K_DAY,
        autype: AuType = AuType.QFQ,
        conn: Optional[FutuConnConfig] = None,
        max_count: int = 1000,
    ) -> None:
        self.codes = codes
        self.start = start
        self.end = end
        self.ktype = ktype
        self.autype = autype
        self.conn = conn if conn is not None else FutuConnConfig()
        self.max_count = max_count

        self._last_market_data: Dict[str, MarketDataEvent] = {}
        self._dfs: Dict[str, pd.DataFrame] = {}  # code -> df(index=time)

        self._load_all()

    @property
    def last_market_data(self) -> Dict[str, MarketDataEvent]:
        return self._last_market_data

    def close(self) -> None:
        # history 不持有长连接，这里留接口对齐
        return

    def __enter__(self) -> "FutuHistoryKlineDataFeed":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _load_all(self) -> None:
        quote_ctx = OpenQuoteContext(host=self.conn.host, port=self.conn.port)
        try:
            for code in self.codes:
                frames: List[pd.DataFrame] = []
                page_req_key = None

                while True:
                    ret, data, page_req_key = quote_ctx.request_history_kline(
                        code,
                        start=self.start,
                        end=self.end,
                        ktype=self.ktype,
                        autype=self.autype,
                        max_count=self.max_count,
                        page_req_key=page_req_key,
                    )
                    if ret != RET_OK:
                        raise RuntimeError(f"request_history_kline failed for {code}: {data}")

                    frames.append(data)
                    if page_req_key is None:
                        break

                df = pd.concat(frames, ignore_index=True)

                if "time_key" not in df.columns:
                    raise RuntimeError(f"history kline df missing time_key for {code}")

                # time_key -> datetime index
                df["time"] = pd.to_datetime(df["time_key"])
                df = df.sort_values("time").set_index("time")

                # 类型统一（存在则转，不存在就留空）
                for col in ["open", "high", "low", "close", "pe_ratio", "turnover_rate", "turnover", "change_rate", "last_close"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                if "volume" in df.columns:
                    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                else:
                    df["volume"] = 0

                self._dfs[code] = df

        finally:
            quote_ctx.close()

    def __iter__(self) -> Iterator[Dict[str, MarketDataEvent]]:
        if not self._dfs:
            return

        all_times = sorted(set().union(*[df.index for df in self._dfs.values()]))

        for ts in all_times:
            events: Dict[str, MarketDataEvent] = {}

            for code, df in self._dfs.items():
                if ts not in df.index:
                    continue

                row = df.loc[ts]
                # 可能同一时间多行，取最后一行
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]

                bar = Bar(
                    open=float(row.get("open", 0.0) if not _isna(row.get("open")) else 0.0),
                    high=float(row.get("high", 0.0) if not _isna(row.get("high")) else 0.0),
                    low=float(row.get("low", 0.0) if not _isna(row.get("low")) else 0.0),
                    close=float(row.get("close", 0.0) if not _isna(row.get("close")) else 0.0),
                    volume=float(row.get("volume", 0.0) if not _isna(row.get("volume")) else 0.0),
                )

                extra = _build_extra_from_row(row, code)

                events[code] = MarketDataEvent(
                    type=EventType.MARKET,
                    timestamp=ts,
                    symbol=code,
                    bar=bar,
                    extra=extra,   # ✅ 你要的字段都在这里
                )

            if events:
                self._last_market_data = events
                yield events


# --------------------------
# Live feed (paper/live)
# --------------------------
class _CurKlineQueueHandler(CurKlineHandlerBase):
    """
    把 futu push 的 DataFrame 放入队列。
    关键：callback 线程里不要阻塞（队列满就丢弃/丢旧）。
    """
    def __init__(self, q: "queue.Queue[Any]", drop_oldest_on_full: bool = True) -> None:
        super().__init__()
        self.q = q
        self.drop_oldest_on_full = drop_oldest_on_full

    def on_recv_rsp(self, rsp_pb):
        ret_code, data = super().on_recv_rsp(rsp_pb)
        if ret_code != RET_OK:
            return ret_code, data

        try:
            self.q.put_nowait(data)
        except queue.Full:
            if self.drop_oldest_on_full:
                try:
                    _ = self.q.get_nowait()  # 丢最旧
                except queue.Empty:
                    pass
                try:
                    self.q.put_nowait(data)
                except queue.Full:
                    pass
            # else: 丢最新（不做任何事）

        return RET_OK, data


class FutuLiveKlineDataFeed(BaseDataFeed):
    """
    实盘/准实时用：订阅 K线推送（callback），通过队列把数据变成 iterator。
    """

    _SENTINEL = object()

    def __init__(
        self,
        codes: List[str],
        subtype: SubType = SubType.K_1M,
        autype: AuType = AuType.QFQ,
        conn: Optional[FutuConnConfig] = None,
        queue_size: int = 10000,
        queue_get_timeout: float = 1.0,  # seconds
    ) -> None:
        self.codes = codes
        self.subtype = subtype
        self.autype = autype
        self.conn = conn if conn is not None else FutuConnConfig()
        self.queue_get_timeout = queue_get_timeout

        self._last_market_data: Dict[str, MarketDataEvent] = {}
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()

        self._quote_ctx = OpenQuoteContext(host=self.conn.host, port=self.conn.port)
        self._handler = _CurKlineQueueHandler(self._q, drop_oldest_on_full=True)
        self._quote_ctx.set_handler(self._handler)

        ret, msg = self._quote_ctx.subscribe(self.codes, [self.subtype])
        if ret != RET_OK:
            self._quote_ctx.close()
            raise RuntimeError(f"subscribe failed: {msg}")

    @property
    def last_market_data(self) -> Dict[str, MarketDataEvent]:
        return self._last_market_data

    def stop(self) -> None:
        """请求停止迭代（会唤醒阻塞 get）。"""
        self._stop.set()
        try:
            self._q.put_nowait(self._SENTINEL)
        except queue.Full:
            pass

    def close(self) -> None:
        try:
            self._quote_ctx.unsubscribe_all()
        except Exception:
            pass
        try:
            self._quote_ctx.close()
        except Exception:
            pass

    def __enter__(self) -> "FutuLiveKlineDataFeed":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
        self.close()

    def __iter__(self) -> Iterator[Dict[str, MarketDataEvent]]:
        try:
            while not self._stop.is_set():
                try:
                    item = self._q.get(timeout=self.queue_get_timeout)
                except queue.Empty:
                    continue

                if item is self._SENTINEL:
                    break

                df: pd.DataFrame = item
                df = df.copy()

                if "time_key" not in df.columns:
                    continue

                df["time"] = pd.to_datetime(df["time_key"])

                # 同一批推送里可能出现同 code 多行：按出现顺序保留最后一条
                last_row_by_code: Dict[str, pd.Series] = {}
                for _, r in df.iterrows():
                    c = _safe_str(r.get("code"))
                    if c is None:
                        continue
                    last_row_by_code[c] = r

                events: Dict[str, MarketDataEvent] = {}
                for code, row in last_row_by_code.items():
                    ts = row.get("time")
                    if _isna(ts):
                        continue

                    bar = Bar(
                        open=float(row.get("open", 0.0) if not _isna(row.get("open")) else 0.0),
                        high=float(row.get("high", 0.0) if not _isna(row.get("high")) else 0.0),
                        low=float(row.get("low", 0.0) if not _isna(row.get("low")) else 0.0),
                        close=float(row.get("close", 0.0) if not _isna(row.get("close")) else 0.0),
                        volume=float(row.get("volume", 0.0) if not _isna(row.get("volume")) else 0.0),
                    )

                    extra = _build_extra_from_row(row, code)

                    events[code] = MarketDataEvent(
                        type=EventType.MARKET,
                        timestamp=ts,
                        symbol=code,
                        bar=bar,
                        extra=extra,  # ✅ 你要的字段都在这里
                    )

                if events:
                    self._last_market_data = events
                    yield events

        finally:
            self.close()
