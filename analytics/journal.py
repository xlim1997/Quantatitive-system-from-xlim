# analytics/journal.py
from __future__ import annotations

import sqlite3
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Optional


def _to_jsonable(x: Any):
    if is_dataclass(x):
        return asdict(x)
    if hasattr(x, "__dict__"):
        return x.__dict__
    return x


class TradeJournal:
    """
    Write every step of the trading process to SQLite:
    - events: all events (type + payload)
    - bars: OHLCV bars (for Kline plot)
    - orders: order state transitions (optional but recommended)
    - fills: executions
    - snapshots: equity/cash/positions summary
    """
    def __init__(self, db_path: str, run_id: str):
        self.db_path = db_path
        self.run_id = run_id
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def close(self):
        self.conn.commit()
        self.conn.close()

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
          ts TEXT, run_id TEXT,
          etype TEXT,
          symbol TEXT,
          payload TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS bars (
          ts TEXT, run_id TEXT,
          symbol TEXT,
          open REAL, high REAL, low REAL, close REAL, volume REAL
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
          ts TEXT, run_id TEXT,
          order_id TEXT,
          symbol TEXT, side TEXT, qty REAL,
          order_type TEXT, limit_price REAL,
          status TEXT,
          payload TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS fills (
          ts TEXT, run_id TEXT,
          fill_id TEXT,
          order_id TEXT,
          symbol TEXT, side TEXT, qty REAL, price REAL,
          commission REAL,
          payload TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
          ts TEXT, run_id TEXT,
          equity REAL, cash REAL,
          gross_exposure REAL, net_exposure REAL,
          positions_json TEXT
        )
        """)
        self.conn.commit()

    def log_event(self, ts, etype: str, symbol: str = "", payload: Any = None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
            (str(ts), self.run_id, etype, symbol, json.dumps(_to_jsonable(payload), default=str)),
        )

    def log_bar(self, ts, symbol: str, o: float, h: float, l: float, c: float, v: float = 0.0):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO bars VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(ts), self.run_id, symbol, float(o), float(h), float(l), float(c), float(v)),
        )

    def log_order(self, ts, order_id: str, symbol: str, side: str, qty: float,
                  order_type: str, status: str, limit_price: Optional[float] = None, payload: Any = None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(ts), self.run_id, order_id, symbol, side, float(qty),
             order_type, float(limit_price) if limit_price is not None else None,
             status, json.dumps(_to_jsonable(payload), default=str)),
        )

    def log_fill(self, ts, fill_id: str, order_id: str, symbol: str, side: str,
                 qty: float, price: float, commission: float = 0.0, payload: Any = None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(ts), self.run_id, fill_id, order_id, symbol, side, float(qty), float(price),
             float(commission), json.dumps(_to_jsonable(payload), default=str)),
        )

    def log_snapshot(self, ts, equity: float, cash: float,
                     gross_exposure: float = 0.0, net_exposure: float = 0.0,
                     positions: Any = None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(ts), self.run_id, float(equity), float(cash),
             float(gross_exposure), float(net_exposure),
             json.dumps(_to_jsonable(positions), default=str)),
        )

    # 你可以在 engine 的 event loop 调这个：按你的事件字段映射即可
    def on_event(self, event: Any):
        etype = getattr(event, "type", event.__class__.__name__)
        ts = getattr(event, "ts", getattr(event, "time", ""))
        symbol = getattr(event, "symbol", "")

        self.log_event(ts=ts, etype=str(etype), symbol=str(symbol), payload=event)
