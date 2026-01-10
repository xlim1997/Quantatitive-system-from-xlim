# # analytics/trade_dashboard.py
# import sqlite3
# import json
# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# st.set_page_config(layout="wide")
# st.title("Trading Process Dashboard")

# # =========================
# # Config
# # =========================
# PLOTLY_CONFIG = {
#     "scrollZoom": True,          # ✅ 鼠标滚轮缩放
#     "displaylogo": False,
#     "responsive": True,
# }

# EVENT_ORDER = ["INSIGHT", "TARGET", "ADJ_TARGET", "RISK_ACTION", "ORDER", "FILL"]

# ETYPE_MAP = {
#     "INSIGHTS": "INSIGHT",
#     "TARGETS": "TARGET",
#     "ADJ_TARGETS": "ADJ_TARGET",
#     "RISK_ACTIONS": "RISK_ACTION",
#     "ORDERS": "ORDER",
#     "FILLS": "FILL",
# }


# # =========================
# # Helpers
# # =========================
# def _safe_json(x):
#     if x is None:
#         return None
#     if isinstance(x, (dict, list)):
#         return x
#     try:
#         return json.loads(x)
#     except Exception:
#         return None


# def _short_payload(x, n: int = 240) -> str:
#     if x is None:
#         return ""
#     try:
#         s = json.dumps(x, ensure_ascii=False)
#     except Exception:
#         s = str(x)
#     return s if len(s) <= n else s[:n] + "…"


# def normalize_etype(e: str) -> str:
#     if e is None:
#         return ""
#     e2 = str(e).strip().upper()
#     return ETYPE_MAP.get(e2, e2)


# def compute_rangebreaks(ts: pd.Series):
#     """
#     去掉非交易日空白：
#       - 周末
#       - 缺失的 business days（包含节假日缺口）
#     """
#     if ts is None or len(ts) == 0:
#         return []

#     s = pd.to_datetime(ts, errors="coerce").dropna()
#     if len(s) == 0:
#         return []

#     d0 = s.min().normalize()
#     d1 = s.max().normalize()

#     # 用 business day 作为“应该出现的交易日集合”，缺失的都作为 break
#     all_bdays = pd.date_range(d0, d1, freq="B")
#     have_days = pd.Index(pd.to_datetime(s.dt.normalize().unique()))
#     missing = all_bdays[~all_bdays.isin(have_days)]

#     # rangebreaks：
#     # 1) 周末
#     # 2) 缺失 business days
#     rbs = [
#         dict(bounds=["sat", "mon"]),
#     ]
#     if len(missing):
#         rbs.append(dict(values=missing.to_pydatetime().tolist()))
#     return rbs


# def apply_rangebreaks(fig: go.Figure, ts: pd.Series) -> go.Figure:
#     rbs = compute_rangebreaks(ts)
#     if rbs:
#         fig.update_xaxes(rangebreaks=rbs)
#     return fig


# def explode_events_by_symbol(events_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     解决“events.symbol 为空但 payload 是 list”的情况：
#     - 如果 payload 是 list，且每个 item 里有 symbol，则拆成多行
#     - etype 统一映射成单数（INSIGHTS->INSIGHT 等）
#     """
#     if events_df is None or len(events_df) == 0:
#         return events_df

#     df = events_df.copy()

#     if "ts" in df.columns:
#         df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

#     if "etype" not in df.columns:
#         df["etype"] = ""

#     if "symbol" not in df.columns:
#         df["symbol"] = ""

#     payload_col = df["payload"] if "payload" in df.columns else pd.Series([None] * len(df))
#     df["payload_json"] = payload_col.apply(_safe_json)
#     df["etype_norm"] = df["etype"].apply(normalize_etype)

#     rows = []
#     for _, r in df.iterrows():
#         et = r.get("etype_norm", "")
#         sym = r.get("symbol", "")
#         ts = r.get("ts", None)
#         payload = r.get("payload_json", None)

#         # list payload -> explode if possible
#         if isinstance(payload, list):
#             exploded = False
#             for item in payload:
#                 if isinstance(item, dict) and "symbol" in item and item["symbol"]:
#                     rows.append(
#                         {
#                             "ts": ts,
#                             "etype": et,
#                             "symbol": str(item.get("symbol", "")),
#                             "payload_json": item,
#                         }
#                     )
#                     exploded = True
#             if not exploded:
#                 rows.append({"ts": ts, "etype": et, "symbol": str(sym or ""), "payload_json": payload})
#         else:
#             # dict payload with symbol -> fill symbol if missing
#             if (not sym) and isinstance(payload, dict) and payload.get("symbol"):
#                 sym = payload.get("symbol")
#             rows.append({"ts": ts, "etype": et, "symbol": str(sym or ""), "payload_json": payload})

#     out = pd.DataFrame(rows)
#     out["hover"] = out.apply(
#         lambda rr: f"{rr.get('etype','')} {rr.get('symbol','')}\n{_short_payload(rr.get('payload_json'))}",
#         axis=1,
#     )

#     # 保留我们关心的类型 + 也允许其它类型（不直接丢弃）
#     return out.sort_values("ts")


# def make_price_trades_event_fig(
#     bars_df: pd.DataFrame,
#     fills_df: pd.DataFrame,
#     events_df: pd.DataFrame,
#     orders_df: pd.DataFrame | None,
#     symbol: str,
# ) -> go.Figure:
#     # 2 rows: price + timeline
#     fig = make_subplots(
#         rows=2,
#         cols=1,
#         shared_xaxes=True,
#         vertical_spacing=0.02,
#         row_heights=[0.78, 0.22],
#         specs=[[{"type": "candlestick"}], [{"type": "scatter"}]],
#     )

#     # --- Row1: Candles
#     fig.add_trace(
#         go.Candlestick(
#             x=bars_df["ts"],
#             open=bars_df["open"],
#             high=bars_df["high"],
#             low=bars_df["low"],
#             close=bars_df["close"],
#             name="OHLC",
#         ),
#         row=1,
#         col=1,
#     )

    

#     fig.update_yaxes(title_text="Price", row=1, col=1)
#     fig.update_layout(
#         height=900,  # ✅ 更大
#         xaxis_rangeslider_visible=False,
#         title=f"{symbol} - Price",
#         hovermode="x unified",  # ✅ 好用的 hover
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
#         margin=dict(l=20, r=20, t=60, b=20),
#     )

#     # ✅ 去掉周末/缺失交易日空白
#     fig = apply_rangebreaks(fig, bars_df["ts"])
#     return fig


# def compute_performance_metrics(snaps_df: pd.DataFrame) -> dict:
#     if snaps_df is None or len(snaps_df) < 2:
#         return {
#             "total_return": 0.0,
#             "cagr": 0.0,
#             "annual_vol": np.nan,
#             "sharpe": np.nan,
#             "max_dd": 0.0,
#             "calmar": np.nan,
#             "ending_equity": np.nan,
#         }

#     eq = snaps_df["equity"].astype(float).values
#     ts = pd.to_datetime(snaps_df["ts"], errors="coerce")
#     n = len(eq)

#     total_return = float(eq[-1] / eq[0] - 1.0) if eq[0] != 0 else 0.0

#     # daily returns
#     rets = pd.Series(eq).pct_change().dropna()
#     vol = float(np.sqrt(252) * rets.std()) if rets.std() and rets.std() > 0 else np.nan
#     sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() and rets.std() > 0 else np.nan

#     # CAGR (assume trading-day snapshots)
#     years = (ts.iloc[-1] - ts.iloc[0]).days / 365.25 if pd.notna(ts.iloc[-1]) and pd.notna(ts.iloc[0]) else (n / 252)
#     cagr = float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0) if years > 0 and eq[0] > 0 else 0.0

#     # drawdown
#     eq_s = pd.Series(eq)
#     peak = eq_s.cummax()
#     dd = eq_s / peak - 1.0
#     max_dd = float(dd.min()) if len(dd) else 0.0
#     calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan

#     return {
#         "total_return": total_return,
#         "cagr": cagr,
#         "annual_vol": vol,
#         "sharpe": sharpe,
#         "max_dd": max_dd,
#         "calmar": calmar,
#         "ending_equity": float(eq[-1]),
#     }


# def make_equity_drawdown_price_fig(snaps_df: pd.DataFrame, bars_df: pd.DataFrame, fills_df: pd.DataFrame, symbol: str) -> go.Figure:
#     """
#     1) Equity
#     2) Drawdown
#     3) Price+Trades
#     同一张图(3 rows)，并且去掉非交易日空白。
#     """
#     eq = snaps_df["equity"].astype(float)
#     peak = eq.cummax()
#     dd = eq / peak - 1.0
#     dd_df = pd.DataFrame({"ts": snaps_df["ts"], "drawdown": dd})

#     fig = make_subplots(
#         rows=3,
#         cols=1,
#         shared_xaxes=True,
#         vertical_spacing=0.03,
#         row_heights=[0.35, 0.20, 0.45],
#         specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "candlestick"}]],
#         subplot_titles=("Equity", "Drawdown", f"{symbol} Price + Trades"),
#     )

#     fig.add_trace(
#         go.Scatter(x=snaps_df["ts"], y=snaps_df["equity"], mode="lines", name="Equity"),
#         row=1,
#         col=1,
#     )
#     fig.add_trace(
#         go.Scatter(x=dd_df["ts"], y=dd_df["drawdown"], mode="lines", name="Drawdown"),
#         row=2,
#         col=1,
#     )

#     fig.add_trace(
#         go.Candlestick(
#             x=bars_df["ts"],
#             open=bars_df["open"],
#             high=bars_df["high"],
#             low=bars_df["low"],
#             close=bars_df["close"],
#             name="OHLC",
#         ),
#         row=3,
#         col=1,
#     )

#     if fills_df is not None and len(fills_df):
#         ff = fills_df.copy()
#         ff["side_u"] = ff["side"].astype(str).str.upper()
#         buys = ff[ff["side_u"] == "BUY"]
#         sells = ff[ff["side_u"] == "SELL"]

#         if len(buys):
#             fig.add_trace(
#                 go.Scatter(
#                     x=buys["ts"],
#                     y=buys["price"],
#                     mode="markers",
#                     name="BUY",
#                     marker=dict(symbol="triangle-up", size=10),
#                     hovertemplate="%{x|%Y-%m-%d}<br>BUY @%{y:.4f}<extra></extra>",
#                 ),
#                 row=3,
#                 col=1,
#             )
#         if len(sells):
#             fig.add_trace(
#                 go.Scatter(
#                     x=sells["ts"],
#                     y=sells["price"],
#                     mode="markers",
#                     name="SELL",
#                     marker=dict(symbol="triangle-down", size=10),
#                     hovertemplate="%{x|%Y-%m-%d}<br>SELL @%{y:.4f}<extra></extra>",
#                 ),
#                 row=3,
#                 col=1,
#             )

#     fig.update_layout(
#         height=950,
#         xaxis_rangeslider_visible=False,
#         hovermode="x unified",
#         margin=dict(l=20, r=20, t=60, b=20),
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
#     )

#     # 用 bars 的 trading calendar 去压缩非交易日空白
#     fig = apply_rangebreaks(fig, bars_df["ts"] if bars_df is not None and len(bars_df) else snaps_df["ts"])
#     return fig


# # =========================
# # Load DB
# # =========================
# DB = st.sidebar.text_input("DB path", "artifacts/trades.db")
# conn = sqlite3.connect(DB)

# runs = pd.read_sql("SELECT DISTINCT run_id FROM events ORDER BY run_id DESC", conn)
# run_id = st.sidebar.selectbox("Run ID", runs["run_id"].tolist() if len(runs) else ["run_001"])

# symbols_df = pd.read_sql(
#     "SELECT DISTINCT symbol FROM bars WHERE run_id=? AND symbol!='' ORDER BY symbol",
#     conn,
#     params=(run_id,),
# )
# symbol = st.sidebar.selectbox("Symbol", symbols_df["symbol"].tolist() if len(symbols_df) else [""])

# bars = pd.read_sql(
#     "SELECT * FROM bars WHERE run_id=? AND symbol=? ORDER BY ts",
#     conn,
#     params=(run_id, symbol),
# )
# fills = pd.read_sql(
#     "SELECT * FROM fills WHERE run_id=? AND symbol=? ORDER BY ts",
#     conn,
#     params=(run_id, symbol),
# )
# snaps = pd.read_sql(
#     "SELECT * FROM snapshots WHERE run_id=? ORDER BY ts",
#     conn,
#     params=(run_id,),
# )
# events = pd.read_sql(
#     "SELECT * FROM events WHERE run_id=? ORDER BY ts",
#     conn,
#     params=(run_id,),
# )

# # optional orders table
# orders = None
# try:
#     orders = pd.read_sql(
#         "SELECT * FROM orders WHERE run_id=? AND symbol=? ORDER BY ts",
#         conn,
#         params=(run_id, symbol),
#     )
# except Exception:
#     orders = None

# for df in (bars, fills, snaps, events):
#     if len(df) and "ts" in df.columns:
#         df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

# if orders is not None and len(orders) and "ts" in orders.columns:
#     orders["ts"] = pd.to_datetime(orders["ts"], errors="coerce")

# # explode + normalize events once
# events_ex = explode_events_by_symbol(events)

# tab1, tab2, tab3, tab4 = st.tabs(["Price + Trades", "Trade Timeline", "Positions/Equity", "Tables"])


# # =========================
# # TAB 1: Price + Trades + Events
# # =========================
# with tab1:
#     if len(bars):
#         fig = make_price_trades_event_fig(
#             bars_df=bars,
#             fills_df=fills,
#             events_df=events_ex,
#             orders_df=orders,
#             symbol=symbol,
#         )
#         st.plotly_chart(fig, width="stretch", key=f"tab1_price_{run_id}_{symbol}", config=PLOTLY_CONFIG)
#     else:
#         st.info("No bars found for this symbol. Make sure you log_bar() from MarketDataEvent.")


# # =========================
# # TAB 2: Trade Timeline (same as your previous; kept minimal here)
# # =========================
# with tab2:
#     st.subheader("Raw Event Timeline (per bar)")
#     if len(events_ex):
#         # 这里用 scatter 展示所有事件，symbol 过滤可选
#         ev2 = events_ex.copy()
#         ev2 = ev2[(ev2["symbol"].astype(str) == str(symbol)) | (ev2["symbol"].astype(str) == "")]
#         fig = go.Figure()
#         for et in EVENT_ORDER:
#             sub = ev2[ev2["etype"] == et]
#             if len(sub):
#                 fig.add_trace(
#                     go.Scattergl(
#                         x=sub["ts"],
#                         y=[et] * len(sub),
#                         mode="markers",
#                         name=et,
#                         marker=dict(size=7),
#                         hovertext=sub["hover"],
#                         hoverinfo="text",
#                     )
#                 )
#         fig.update_layout(height=520, hovermode="closest", title="Events (exploded & normalized)")
#         fig.update_yaxes(type="category", categoryorder="array", categoryarray=EVENT_ORDER)
#         fig = apply_rangebreaks(fig, bars["ts"] if len(bars) else snaps["ts"])
#         st.plotly_chart(fig, width="stretch", key=f"tab2_events_{run_id}_{symbol}", config=PLOTLY_CONFIG)
#     else:
#         st.info("No events found.")


# # =========================
# # TAB 3: Positions/Equity (+ Sharpe etc) + Price under Drawdown
# # =========================
# with tab3:
#     if len(snaps):
#         # ---- Metrics ----
#         m = compute_performance_metrics(snaps)

#         c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
#         c1.metric("Total Return", f"{m['total_return']:.2%}")
#         c2.metric("CAGR", f"{m['cagr']:.2%}")
#         c3.metric("Sharpe", f"{m['sharpe']:.3f}" if np.isfinite(m["sharpe"]) else "NA")
#         c4.metric("Annual Vol", f"{m['annual_vol']:.2%}" if np.isfinite(m["annual_vol"]) else "NA")
#         c5.metric("Max Drawdown", f"{m['max_dd']:.2%}")
#         c6.metric("Calmar", f"{m['calmar']:.3f}" if np.isfinite(m["calmar"]) else "NA")
#         c7.metric("Ending Equity", f"{m['ending_equity']:,.2f}" if np.isfinite(m["ending_equity"]) else "NA")

#         if len(bars):
#             fig = make_equity_drawdown_price_fig(snaps, bars, fills, symbol)
#             st.plotly_chart(fig, width="stretch", key=f"tab3_perf_{run_id}_{symbol}", config=PLOTLY_CONFIG)
#         else:
#             # fallback: only equity/drawdown
#             eq = snaps["equity"].astype(float)
#             peak = eq.cummax()
#             dd = eq / peak - 1.0

#             fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.04)
#             fig.add_trace(go.Scatter(x=snaps["ts"], y=snaps["equity"], mode="lines", name="Equity"), row=1, col=1)
#             fig.add_trace(go.Scatter(x=snaps["ts"], y=dd, mode="lines", name="Drawdown"), row=2, col=1)
#             fig.update_layout(height=780, hovermode="x unified")
#             fig = apply_rangebreaks(fig, snaps["ts"])
#             st.plotly_chart(fig, width="stretch", key=f"tab3_perf_fallback_{run_id}", config=PLOTLY_CONFIG)
#     else:
#         st.info("No snapshots found. Make sure you log_snapshot() each bar.")


# # =========================
# # TAB 4: Tables
# # =========================
# with tab4:
#     st.subheader("Fills")
#     st.dataframe(fills, width="stretch", height=220)

#     st.subheader("Orders")
#     if orders is not None:
#         st.dataframe(orders, width="stretch", height=220)
#     else:
#         st.info("No orders table or no orders logged yet.")

#     st.subheader("Events (raw)")
#     st.dataframe(events.tail(300), width="stretch", height=260)

#     st.subheader("Events (exploded & normalized)")
#     if len(events_ex):
#         st.dataframe(events_ex.tail(300), width="stretch", height=260)




# analytics/trade_dashboard.py
import sqlite3
import json
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("Trading Process Dashboard")

# Plotly config: wheel zoom + pan
PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
}

EVENT_ORDER = ["INSIGHT", "TARGET", "ADJ_TARGET", "RISK_ACTION", "ORDER", "FILL"]

ETYPE_MAP = {
    "INSIGHTS": "INSIGHT",
    "INSIGHT": "INSIGHT",
    "TARGETS": "TARGET",
    "TARGET": "TARGET",
    "ADJ_TARGETS": "ADJ_TARGET",
    "ADJ_TARGET": "ADJ_TARGET",
    "RISK_ACTIONS": "RISK_ACTION",
    "RISK_ACTION": "RISK_ACTION",
    "ORDERS": "ORDER",
    "ORDER": "ORDER",
    "FILLS": "FILL",
    "FILL": "FILL",
}


def _safe_json(x):
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return None


def _short_payload(x, n: int = 240) -> str:
    if x is None:
        return ""
    try:
        s = json.dumps(x, ensure_ascii=False)
    except Exception:
        s = str(x)
    return s if len(s) <= n else s[:n] + "…"


def normalize_etype(e: str) -> str:
    if e is None:
        return ""
    e2 = str(e).strip().upper()
    return ETYPE_MAP.get(e2, e2)


def norm_sym(s: str) -> str:
    return "" if s is None else str(s).strip().upper()


def sym_match(selected: str, candidate: str) -> bool:
    """Robust symbol matching.

    NVDA should match: NVDA, NVDA.US, STK.NVDA, NVDA-USD, etc.
    """
    sel = norm_sym(selected)
    cand = norm_sym(candidate)
    if not sel or not cand:
        return False
    if cand == sel:
        return True
    return bool(re.match(rf"^{re.escape(sel)}(\b|[\.\-_/:\\s])", cand))


def to_ts(s: pd.Series) -> pd.Series:
    s2 = pd.to_datetime(s, errors="coerce")
    # tz-aware -> tz-naive
    try:
        if getattr(s2.dt, "tz", None) is not None:
            s2 = s2.dt.tz_convert(None)
    except Exception:
        pass
    return s2


def compute_rangebreaks(ts: pd.Series):
    """Remove non-trading gaps (weekends + missing business days)."""
    if ts is None or len(ts) == 0:
        return []
    s = to_ts(ts).dropna()
    if len(s) == 0:
        return []

    d0 = s.min().normalize()
    d1 = s.max().normalize()

    all_bdays = pd.date_range(d0, d1, freq="B")
    have_days = pd.Index(pd.to_datetime(s.dt.normalize().unique()))
    missing = all_bdays[~all_bdays.isin(have_days)]

    rbs = [dict(bounds=["sat", "mon"]) ]
    if len(missing):
        rbs.append(dict(values=missing.to_pydatetime().tolist()))
    return rbs


def apply_rangebreaks(fig: go.Figure, ts: pd.Series) -> go.Figure:
    rbs = compute_rangebreaks(ts)
    if rbs:
        fig.update_xaxes(rangebreaks=rbs)
    return fig


def explode_events_by_symbol(events_df: pd.DataFrame) -> pd.DataFrame:
    """Explode events so INSIGHT/TARGET/ADJ_TARGET can be per-symbol rows.

    This prevents 'no events' when events.symbol is empty but payload is a list.
    """
    if events_df is None or len(events_df) == 0:
        return pd.DataFrame(columns=["ts", "etype", "symbol", "payload_json", "hover"])

    df = events_df.copy()
    if "ts" in df.columns:
        df["ts"] = to_ts(df["ts"])
    else:
        df["ts"] = pd.NaT

    if "etype" not in df.columns:
        df["etype"] = ""
    if "symbol" not in df.columns:
        df["symbol"] = ""

    payload_col = df["payload"] if "payload" in df.columns else pd.Series([None] * len(df))
    df["payload_json"] = payload_col.apply(_safe_json)
    df["etype"] = df["etype"].apply(normalize_etype)

    rows = []
    for _, r in df.iterrows():
        ts = r.get("ts")
        et = r.get("etype")
        sym0 = r.get("symbol")
        payload = r.get("payload_json")

        if isinstance(payload, list):
            exploded_any = False
            for item in payload:
                if isinstance(item, dict):
                    sym = item.get("symbol") or item.get("sym") or item.get("ticker")
                    if sym:
                        rows.append({"ts": ts, "etype": et, "symbol": str(sym), "payload_json": item})
                        exploded_any = True
            if not exploded_any:
                rows.append({"ts": ts, "etype": et, "symbol": str(sym0 or ""), "payload_json": payload})
        else:
            if isinstance(payload, dict):
                sym = payload.get("symbol") or payload.get("sym") or payload.get("ticker")
                if (not sym0) and sym:
                    sym0 = sym
            rows.append({"ts": ts, "etype": et, "symbol": str(sym0 or ""), "payload_json": payload})

    out = pd.DataFrame(rows)
    out["ts"] = to_ts(out["ts"])
    out = out.dropna(subset=["ts"]).sort_values("ts")

    out["hover"] = out.apply(
        lambda rr: f"{rr.get('etype','')} {rr.get('symbol','')}\n{_short_payload(rr.get('payload_json'))}",
        axis=1,
    )
    return out


def build_event_panel_df(
    events_ex: pd.DataFrame,
    fills_df: pd.DataFrame,
    orders_df: Optional[pd.DataFrame],
    selected_symbol: str,
) -> pd.DataFrame:
    """Events for the bottom row in tab1: INSIGHT/TARGET/ADJ_TARGET/RISK_ACTION + ORDER + FILL."""
    evs = events_ex.copy() if events_ex is not None and len(events_ex) else pd.DataFrame(
        columns=["ts", "etype", "symbol", "payload_json", "hover"]
    )

    if len(evs):
        # keep events for this symbol, plus symbol-empty (portfolio-level) events
        mask = evs["symbol"].astype(str).apply(lambda s: (not str(s).strip()) or sym_match(selected_symbol, s))
        evs = evs[mask].copy()

    extra_rows = []

    if orders_df is not None and len(orders_df):
        od = orders_df.copy()
        if "ts" in od.columns:
            od["ts"] = to_ts(od["ts"])
        od = od.dropna(subset=["ts"])
        for _, r in od.iterrows():
            payload = r.to_dict()
            extra_rows.append({
                "ts": r["ts"],
                "etype": "ORDER",
                "symbol": selected_symbol,
                "payload_json": payload,
                "hover": f"ORDER {selected_symbol}\n{_short_payload(payload)}",
            })

    if fills_df is not None and len(fills_df):
        fd = fills_df.copy()
        if "ts" in fd.columns:
            fd["ts"] = to_ts(fd["ts"])
        fd = fd.dropna(subset=["ts"])
        for _, r in fd.iterrows():
            payload = r.to_dict()
            extra_rows.append({
                "ts": r["ts"],
                "etype": "FILL",
                "symbol": selected_symbol,
                "payload_json": payload,
                "hover": f"FILL {selected_symbol}\n{_short_payload(payload)}",
            })

    if extra_rows:
        evs = pd.concat([evs, pd.DataFrame(extra_rows)], ignore_index=True)

    evs["etype"] = evs["etype"].apply(normalize_etype)
    evs["ts"] = to_ts(evs["ts"])
    evs = evs.dropna(subset=["ts"]).sort_values("ts")
    return evs


def make_price_trades_event_fig(
    bars_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    ev_panel_df: pd.DataFrame,
    selected_symbol: str,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.82, 0.18],
        specs=[[{"type": "candlestick"}], [{"type": "scatter"}]],
    )

    # Row 1: Candlestick
    fig.add_trace(
        go.Candlestick(
            x=bars_df["ts"],
            open=bars_df["open"],
            high=bars_df["high"],
            low=bars_df["low"],
            close=bars_df["close"],
            name="OHLC",
            increasing_line_width=1.2,
            decreasing_line_width=1.2,
        ),
        row=1,
        col=1,
    )

    # Row 1: BUY/SELL markers
    if fills_df is not None and len(fills_df):
        ff = fills_df.copy()
        ff["ts"] = to_ts(ff["ts"])
        ff = ff.dropna(subset=["ts"])
        ff["side_u"] = ff.get("side", "").astype(str).str.upper()
        buys = ff[ff["side_u"].str.contains("BUY", na=False)]
        sells = ff[ff["side_u"].str.contains("SELL", na=False)]

        if len(buys):
            fig.add_trace(
                go.Scatter(
                    x=buys["ts"],
                    y=buys["price"].astype(float),
                    mode="markers",
                    name="BUY",
                    marker=dict(symbol="triangle-up", size=11),
                    hovertemplate="%{x|%Y-%m-%d}<br>BUY @%{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        if len(sells):
            fig.add_trace(
                go.Scatter(
                    x=sells["ts"],
                    y=sells["price"].astype(float),
                    mode="markers",
                    name="SELL",
                    marker=dict(symbol="triangle-down", size=11),
                    hovertemplate="%{x|%Y-%m-%d}<br>SELL @%{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # Row 2: Event timeline markers
    evs = ev_panel_df.copy() if ev_panel_df is not None and len(ev_panel_df) else pd.DataFrame(
        columns=["ts", "etype", "hover"]
    )

    if len(evs):
        evs["etype"] = evs["etype"].apply(normalize_etype)
        # Only keep known event types to avoid messy legend
        evs = evs[evs["etype"].isin(EVENT_ORDER)].copy()

    for et in EVENT_ORDER:
        sub = evs[evs["etype"] == et] if len(evs) else evs
        if len(sub):
            fig.add_trace(
                go.Scattergl(
                    x=sub["ts"],
                    y=[et] * len(sub),
                    mode="markers",
                    name=et,
                    marker=dict(size=6),
                    hovertext=sub.get("hover", None),
                    hoverinfo="text",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(
        title_text="Events",
        type="category",
        categoryorder="array",
        categoryarray=EVENT_ORDER[::-1],  # top->bottom
        row=2,
        col=1,
    )

    # Better interaction
    fig.update_layout(
        height=980,
        xaxis_rangeslider_visible=False,
        title=f"{selected_symbol} - Price + Trades + Event Timeline",
        hovermode="x unified",
        dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # Remove non-trading gaps
    fig = apply_rangebreaks(fig, bars_df["ts"])

    # Force x range to data (prevents weird extension to 2030)
    try:
        xmin = pd.to_datetime(bars_df["ts"]).min()
        xmax = pd.to_datetime(bars_df["ts"]).max()
        if pd.notna(xmin) and pd.notna(xmax):
            fig.update_xaxes(range=[xmin, xmax])
    except Exception:
        pass

    return fig


def reconstruct_trades_from_fills(fills_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct round-trip long trades from fills (BUY...SELL back to flat).

    This assumes long-only or at least positions return to 0.
    """
    if fills_df is None or len(fills_df) == 0:
        return pd.DataFrame(columns=[
            "entry_ts", "exit_ts", "qty", "entry_avg", "exit_avg", "pnl", "holding_days"
        ])

    ff = fills_df.copy().sort_values("ts")
    ff["ts"] = to_ts(ff["ts"])
    ff = ff.dropna(subset=["ts"])

    def signed_qty(row):
        side = str(row.get("side", "")).lower()
        q = float(row.get("quantity", 0.0))
        return q if "buy" in side else (-q if "sell" in side else 0.0)

    ff["signed_qty"] = ff.apply(signed_qty, axis=1)
    ff["pos"] = ff["signed_qty"].cumsum()

    rows = []
    in_trade = False
    entry_idx = None

    for i, r in ff.iterrows():
        cur_pos = float(r["pos"])
        if (not in_trade) and cur_pos > 0:
            in_trade = True
            entry_idx = i
        elif in_trade and cur_pos == 0:
            sl = ff.loc[entry_idx:i]
            buys = sl[sl["signed_qty"] > 0]
            sells = sl[sl["signed_qty"] < 0]

            entry_ts = buys["ts"].iloc[0] if len(buys) else sl["ts"].iloc[0]
            exit_ts = sells["ts"].iloc[-1] if len(sells) else sl["ts"].iloc[-1]

            entry_qty = float(buys["signed_qty"].sum()) if len(buys) else float(sl["signed_qty"].sum())
            exit_qty = float((-sells["signed_qty"]).sum()) if len(sells) else float(entry_qty)
            qty = min(entry_qty, exit_qty)

            entry_avg = float((buys["price"].astype(float) * buys["signed_qty"].astype(float)).sum() / max(entry_qty, 1e-12)) if len(buys) else float("nan")
            exit_avg = float((sells["price"].astype(float) * (-sells["signed_qty"]).astype(float)).sum() / max(exit_qty, 1e-12)) if len(sells) else float("nan")

            pnl = (exit_avg - entry_avg) * qty if (not np.isnan(entry_avg) and not np.isnan(exit_avg)) else np.nan
            hold_days = (exit_ts - entry_ts).days if pd.notna(entry_ts) and pd.notna(exit_ts) else None

            rows.append({
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "qty": qty,
                "entry_avg": entry_avg,
                "exit_avg": exit_avg,
                "pnl": pnl,
                "holding_days": hold_days,
            })

            in_trade = False
            entry_idx = None

    return pd.DataFrame(rows)


def attach_trade_triggers(
    trades_df: pd.DataFrame,
    events_ex: pd.DataFrame,
    selected_symbol: str,
) -> pd.DataFrame:
    if trades_df is None or len(trades_df) == 0:
        return trades_df

    evs = events_ex.copy() if events_ex is not None else pd.DataFrame()
    if len(evs):
        mask = evs["symbol"].astype(str).apply(lambda s: sym_match(selected_symbol, s))
        evs = evs[mask].copy()
        evs = evs.sort_values("ts")

    def nearest_before(ts, etype: str):
        if len(evs) == 0 or pd.isna(ts):
            return None
        sub = evs[evs["etype"] == etype]
        sub = sub[sub["ts"] <= ts + pd.Timedelta(days=1)]
        if len(sub) == 0:
            return None
        return sub.iloc[-1]

    out = trades_df.copy()
    entry_reasons = []
    exit_reasons = []

    for _, r in out.iterrows():
        e = r.get("entry_ts")
        x = r.get("exit_ts")

        ins = nearest_before(e, "INSIGHT")
        if ins is not None:
            entry_reasons.append(_short_payload(ins.get("payload_json"), 380))
        else:
            entry_reasons.append("")

        ra = nearest_before(x, "RISK_ACTION")
        if ra is not None:
            exit_reasons.append(_short_payload(ra.get("payload_json"), 380))
        else:
            exit_reasons.append("")

    out["entry_reason"] = entry_reasons
    out["exit_reason"] = exit_reasons
    return out


def perf_metrics_from_snapshots(snaps: pd.DataFrame):
    if snaps is None or len(snaps) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "max_dd_duration_bars": 0,
            "periods": 0,
        }

    df = snaps.copy().sort_values("ts")
    df["ts"] = to_ts(df["ts"])
    df = df.dropna(subset=["ts"])
    eq = df["equity"].astype(float)

    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if float(eq.iloc[0]) != 0 else 0.0

    # daily returns
    rets = eq.pct_change().dropna()
    ann_vol = float(rets.std(ddof=0) * np.sqrt(252)) if len(rets) else 0.0
    sharpe = float((rets.mean() / (rets.std(ddof=0) + 1e-12)) * np.sqrt(252)) if len(rets) else 0.0

    # CAGR
    days = max((df["ts"].iloc[-1] - df["ts"].iloc[0]).days, 1)
    years = days / 365.25
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 and eq.iloc[0] > 0 else 0.0

    # drawdown
    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0

    # max dd duration in bars (time under peak)
    under = dd < 0
    # compute longest consecutive True stretch
    max_dur = 0
    cur = 0
    for v in under.tolist():
        if v:
            cur += 1
            max_dur = max(max_dur, cur)
        else:
            cur = 0

    return {
        "total_return": total_ret,
        "cagr": cagr,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_dd_duration_bars": int(max_dur),
        "periods": int(len(df)),
    }


def make_perf_fig(snaps: pd.DataFrame, bars_df: pd.DataFrame, fills_df: pd.DataFrame) -> go.Figure:
    df = snaps.copy().sort_values("ts")
    df["ts"] = to_ts(df["ts"])
    df = df.dropna(subset=["ts"])
    eq = df["equity"].astype(float)
    peak = eq.cummax()
    dd = eq / peak - 1.0

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.28, 0.22, 0.50],
        specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "candlestick"}]],
    )

    fig.add_trace(go.Scatter(x=df["ts"], y=eq, mode="lines", name="Equity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["ts"], y=dd, mode="lines", name="Drawdown"), row=2, col=1)

    # price candles on bottom
    b = bars_df.copy().sort_values("ts")
    b["ts"] = to_ts(b["ts"])
    b = b.dropna(subset=["ts"])
    fig.add_trace(
        go.Candlestick(
            x=b["ts"],
            open=b["open"],
            high=b["high"],
            low=b["low"],
            close=b["close"],
            name="OHLC",
            increasing_line_width=1.1,
            decreasing_line_width=1.1,
        ),
        row=3,
        col=1,
    )

    # fill markers
    if fills_df is not None and len(fills_df):
        ff = fills_df.copy().sort_values("ts")
        ff["ts"] = to_ts(ff["ts"])
        ff = ff.dropna(subset=["ts"])
        ff["side_u"] = ff.get("side", "").astype(str).str.upper()
        buys = ff[ff["side_u"].str.contains("BUY", na=False)]
        sells = ff[ff["side_u"].str.contains("SELL", na=False)]
        if len(buys):
            fig.add_trace(
                go.Scatter(
                    x=buys["ts"],
                    y=buys["price"].astype(float),
                    mode="markers",
                    name="BUY",
                    marker=dict(symbol="triangle-up", size=9),
                ),
                row=3,
                col=1,
            )
        if len(sells):
            fig.add_trace(
                go.Scatter(
                    x=sells["ts"],
                    y=sells["price"].astype(float),
                    mode="markers",
                    name="SELL",
                    marker=dict(symbol="triangle-down", size=9),
                ),
                row=3,
                col=1,
            )

    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=3, col=1)

    fig.update_layout(
        height=980,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        dragmode="pan",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    fig = apply_rangebreaks(fig, b["ts"])

    return fig


# =========================
# Load data
# =========================
DB = st.sidebar.text_input("DB path", "artifacts/trades.db")

conn = sqlite3.connect(DB)

runs = pd.read_sql("SELECT DISTINCT run_id FROM events ORDER BY run_id DESC", conn)
run_id = st.sidebar.selectbox("Run ID", runs["run_id"].tolist() if len(runs) else ["run_001"])

symbols_df = pd.read_sql(
    "SELECT DISTINCT symbol FROM bars WHERE run_id=? AND symbol!='' ORDER BY symbol",
    conn,
    params=(run_id,),
)
symbol = st.sidebar.selectbox("Symbol", symbols_df["symbol"].tolist() if len(symbols_df) else [""])

bars = pd.read_sql("SELECT * FROM bars WHERE run_id=? AND symbol=? ORDER BY ts", conn, params=(run_id, symbol))
fills = pd.read_sql("SELECT * FROM fills WHERE run_id=? AND symbol=? ORDER BY ts", conn, params=(run_id, symbol))
snaps = pd.read_sql("SELECT * FROM snapshots WHERE run_id=? ORDER BY ts", conn, params=(run_id,))
events = pd.read_sql("SELECT * FROM events WHERE run_id=? ORDER BY ts", conn, params=(run_id,))

# orders table might not exist
orders = None
try:
    orders = pd.read_sql("SELECT * FROM orders WHERE run_id=? AND symbol=? ORDER BY ts", conn, params=(run_id, symbol))
except Exception:
    orders = None

# Parse timestamps
for df in (bars, fills, snaps, events):
    if len(df) and "ts" in df.columns:
        df["ts"] = to_ts(df["ts"])
if orders is not None and len(orders) and "ts" in orders.columns:
    orders["ts"] = to_ts(orders["ts"])

# Explode & decode events once
events_ex = explode_events_by_symbol(events)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["Price + Trades", "Trade Timeline", "Positions/Equity", "Tables"])


with tab1:
    if len(bars):
        ev_panel = build_event_panel_df(events_ex, fills, orders, symbol)
        fig = make_price_trades_event_fig(bars, fills, ev_panel, symbol)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"tab1_price_{run_id}_{symbol}")

        with st.expander("Debug: counts"):
            st.write({
                "fills": int(len(fills)),
                "orders": int(len(orders) if orders is not None else 0),
                "events_ex": int(len(events_ex)),
                "events_panel": int(len(ev_panel)),
                "events_panel_by_type": ev_panel["etype"].value_counts().to_dict() if len(ev_panel) else {},
            })
    else:
        st.info("No bars found for this symbol. Make sure you log_bar() from MarketDataEvent.")


with tab2:
    st.subheader("Trade Timeline (Human-readable)")

    # reconstruct trades from fills
    trades = reconstruct_trades_from_fills(fills)
    trades = attach_trade_triggers(trades, events_ex, symbol)

    if len(trades):
        st.dataframe(trades, width="stretch", height=260, key=f"tab2_trades_{run_id}_{symbol}")

        st.markdown("### Chronological log")
        for _, tr in trades.iterrows():
            e = tr["entry_ts"].date() if pd.notna(tr["entry_ts"]) else tr["entry_ts"]
            x = tr["exit_ts"].date() if pd.notna(tr["exit_ts"]) else tr["exit_ts"]
            st.markdown(
                f"- **{e}** BUY @{tr['entry_avg']:.4f} qty={tr['qty']:.0f}\n"
                f"  Trigger: {tr.get('entry_reason','')}"
            )
            st.markdown(
                f"  **{x}** EXIT @{tr['exit_avg']:.4f}  PnL={tr['pnl']:.2f}\n"
                f"  Trigger: {tr.get('exit_reason','')}"
            )
    else:
        st.info("No complete round-trip trades reconstructed from fills yet (need buy then sell back to flat).")

    st.markdown("---")
    st.subheader("Raw Event Timeline (per bar)")
    # raw event scatter (symbol-filtered)
    evs = events_ex.copy()
    if len(evs):
        evs = evs[evs["symbol"].astype(str).apply(lambda s: sym_match(symbol, s))].copy()
        evs = evs[evs["etype"].isin(["INSIGHT", "TARGET", "ADJ_TARGET", "RISK_ACTION"])].copy()

    if len(evs):
        fig = px.scatter(evs, x="ts", y="etype", hover_data=["symbol"], title="Raw Event Timeline")
        fig = apply_rangebreaks(fig, bars["ts"] if len(bars) else evs["ts"])
        fig.update_layout(height=520, hovermode="closest")
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"tab2_rawevents_{run_id}_{symbol}")
    else:
        st.info("No symbol-matched events found.")


with tab3:
    if len(snaps):
        m = perf_metrics_from_snapshots(snaps)

        # add trade stats
        trades = reconstruct_trades_from_fills(fills)
        num_trades = int(len(trades))
        win_rate = float((trades["pnl"] > 0).mean()) if len(trades) else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return", f"{m['total_return']:.2%}")
        c2.metric("CAGR", f"{m['cagr']:.2%}")
        c3.metric("Sharpe", f"{m['sharpe']:.2f}")
        c4.metric("Max Drawdown", f"{m['max_drawdown']:.2%}")
        c5.metric("#Trades / Win", f"{num_trades} / {win_rate:.0%}")

        fig = make_perf_fig(snaps, bars, fills)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"tab3_perf_{run_id}_{symbol}")

        with st.expander("More metrics"):
            st.write(m)
    else:
        st.info("No snapshots found. Make sure you log_snapshot() each bar.")


with tab4:
    st.subheader("Fills")
    st.dataframe(fills, width="stretch", height=220, key=f"tab4_fills_{run_id}_{symbol}")

    st.subheader("Orders")
    if orders is not None:
        st.dataframe(orders, width="stretch", height=220, key=f"tab4_orders_{run_id}_{symbol}")
    else:
        st.info("No orders table or no orders logged yet.")

    st.subheader("Events (raw)")
    st.dataframe(events.tail(400), width="stretch", height=260, key=f"tab4_events_{run_id}_{symbol}")
