# analytics/trade_dashboard.py
import sqlite3
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("Trading Process Dashboard")

# =========================
# Config
# =========================
PLOTLY_CONFIG = {
    "scrollZoom": True,          # ✅ 鼠标滚轮缩放
    "displaylogo": False,
    "responsive": True,
}

EVENT_ORDER = ["INSIGHT", "TARGET", "ADJ_TARGET", "RISK_ACTION", "ORDER", "FILL"]

ETYPE_MAP = {
    "INSIGHTS": "INSIGHT",
    "TARGETS": "TARGET",
    "ADJ_TARGETS": "ADJ_TARGET",
    "RISK_ACTIONS": "RISK_ACTION",
    "ORDERS": "ORDER",
    "FILLS": "FILL",
}


# =========================
# Helpers
# =========================
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


def compute_rangebreaks(ts: pd.Series):
    """
    去掉非交易日空白：
      - 周末
      - 缺失的 business days（包含节假日缺口）
    """
    if ts is None or len(ts) == 0:
        return []

    s = pd.to_datetime(ts, errors="coerce").dropna()
    if len(s) == 0:
        return []

    d0 = s.min().normalize()
    d1 = s.max().normalize()

    # 用 business day 作为“应该出现的交易日集合”，缺失的都作为 break
    all_bdays = pd.date_range(d0, d1, freq="B")
    have_days = pd.Index(pd.to_datetime(s.dt.normalize().unique()))
    missing = all_bdays[~all_bdays.isin(have_days)]

    # rangebreaks：
    # 1) 周末
    # 2) 缺失 business days
    rbs = [
        dict(bounds=["sat", "mon"]),
    ]
    if len(missing):
        rbs.append(dict(values=missing.to_pydatetime().tolist()))
    return rbs


def apply_rangebreaks(fig: go.Figure, ts: pd.Series) -> go.Figure:
    rbs = compute_rangebreaks(ts)
    if rbs:
        fig.update_xaxes(rangebreaks=rbs)
    return fig


def explode_events_by_symbol(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    解决“events.symbol 为空但 payload 是 list”的情况：
    - 如果 payload 是 list，且每个 item 里有 symbol，则拆成多行
    - etype 统一映射成单数（INSIGHTS->INSIGHT 等）
    """
    if events_df is None or len(events_df) == 0:
        return events_df

    df = events_df.copy()

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    if "etype" not in df.columns:
        df["etype"] = ""

    if "symbol" not in df.columns:
        df["symbol"] = ""

    payload_col = df["payload"] if "payload" in df.columns else pd.Series([None] * len(df))
    df["payload_json"] = payload_col.apply(_safe_json)
    df["etype_norm"] = df["etype"].apply(normalize_etype)

    rows = []
    for _, r in df.iterrows():
        et = r.get("etype_norm", "")
        sym = r.get("symbol", "")
        ts = r.get("ts", None)
        payload = r.get("payload_json", None)

        # list payload -> explode if possible
        if isinstance(payload, list):
            exploded = False
            for item in payload:
                if isinstance(item, dict) and "symbol" in item and item["symbol"]:
                    rows.append(
                        {
                            "ts": ts,
                            "etype": et,
                            "symbol": str(item.get("symbol", "")),
                            "payload_json": item,
                        }
                    )
                    exploded = True
            if not exploded:
                rows.append({"ts": ts, "etype": et, "symbol": str(sym or ""), "payload_json": payload})
        else:
            # dict payload with symbol -> fill symbol if missing
            if (not sym) and isinstance(payload, dict) and payload.get("symbol"):
                sym = payload.get("symbol")
            rows.append({"ts": ts, "etype": et, "symbol": str(sym or ""), "payload_json": payload})

    out = pd.DataFrame(rows)
    out["hover"] = out.apply(
        lambda rr: f"{rr.get('etype','')} {rr.get('symbol','')}\n{_short_payload(rr.get('payload_json'))}",
        axis=1,
    )

    # 保留我们关心的类型 + 也允许其它类型（不直接丢弃）
    return out.sort_values("ts")


def make_price_trades_event_fig(
    bars_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    events_df: pd.DataFrame,
    orders_df: pd.DataFrame | None,
    symbol: str,
) -> go.Figure:
    # 2 rows: price + timeline
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.78, 0.22],
        specs=[[{"type": "candlestick"}], [{"type": "scatter"}]],
    )

    # --- Row1: Candles
    fig.add_trace(
        go.Candlestick(
            x=bars_df["ts"],
            open=bars_df["open"],
            high=bars_df["high"],
            low=bars_df["low"],
            close=bars_df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_layout(
        height=900,  # ✅ 更大
        xaxis_rangeslider_visible=False,
        title=f"{symbol} - Price",
        hovermode="x unified",  # ✅ 好用的 hover
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # ✅ 去掉周末/缺失交易日空白
    fig = apply_rangebreaks(fig, bars_df["ts"])
    return fig


def compute_performance_metrics(snaps_df: pd.DataFrame) -> dict:
    if snaps_df is None or len(snaps_df) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_dd": 0.0,
            "calmar": np.nan,
            "ending_equity": np.nan,
        }

    eq = snaps_df["equity"].astype(float).values
    ts = pd.to_datetime(snaps_df["ts"], errors="coerce")
    n = len(eq)

    total_return = float(eq[-1] / eq[0] - 1.0) if eq[0] != 0 else 0.0

    # daily returns
    rets = pd.Series(eq).pct_change().dropna()
    vol = float(np.sqrt(252) * rets.std()) if rets.std() and rets.std() > 0 else np.nan
    sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() and rets.std() > 0 else np.nan

    # CAGR (assume trading-day snapshots)
    years = (ts.iloc[-1] - ts.iloc[0]).days / 365.25 if pd.notna(ts.iloc[-1]) and pd.notna(ts.iloc[0]) else (n / 252)
    cagr = float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0) if years > 0 and eq[0] > 0 else 0.0

    # drawdown
    eq_s = pd.Series(eq)
    peak = eq_s.cummax()
    dd = eq_s / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "ending_equity": float(eq[-1]),
    }


def make_equity_drawdown_price_fig(snaps_df: pd.DataFrame, bars_df: pd.DataFrame, fills_df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    1) Equity
    2) Drawdown
    3) Price+Trades
    同一张图(3 rows)，并且去掉非交易日空白。
    """
    eq = snaps_df["equity"].astype(float)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    dd_df = pd.DataFrame({"ts": snaps_df["ts"], "drawdown": dd})

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.20, 0.45],
        specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "candlestick"}]],
        subplot_titles=("Equity", "Drawdown", f"{symbol} Price + Trades"),
    )

    fig.add_trace(
        go.Scatter(x=snaps_df["ts"], y=snaps_df["equity"], mode="lines", name="Equity"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dd_df["ts"], y=dd_df["drawdown"], mode="lines", name="Drawdown"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Candlestick(
            x=bars_df["ts"],
            open=bars_df["open"],
            high=bars_df["high"],
            low=bars_df["low"],
            close=bars_df["close"],
            name="OHLC",
        ),
        row=3,
        col=1,
    )

    if fills_df is not None and len(fills_df):
        ff = fills_df.copy()
        ff["side_u"] = ff["side"].astype(str).str.upper()
        buys = ff[ff["side_u"] == "BUY"]
        sells = ff[ff["side_u"] == "SELL"]

        if len(buys):
            fig.add_trace(
                go.Scatter(
                    x=buys["ts"],
                    y=buys["price"],
                    mode="markers",
                    name="BUY",
                    marker=dict(symbol="triangle-up", size=10),
                    hovertemplate="%{x|%Y-%m-%d}<br>BUY @%{y:.4f}<extra></extra>",
                ),
                row=3,
                col=1,
            )
        if len(sells):
            fig.add_trace(
                go.Scatter(
                    x=sells["ts"],
                    y=sells["price"],
                    mode="markers",
                    name="SELL",
                    marker=dict(symbol="triangle-down", size=10),
                    hovertemplate="%{x|%Y-%m-%d}<br>SELL @%{y:.4f}<extra></extra>",
                ),
                row=3,
                col=1,
            )

    fig.update_layout(
        height=950,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    # 用 bars 的 trading calendar 去压缩非交易日空白
    fig = apply_rangebreaks(fig, bars_df["ts"] if bars_df is not None and len(bars_df) else snaps_df["ts"])
    return fig


# =========================
# Load DB
# =========================
DB = st.sidebar.text_input("DB path", "artifacts/trades.db")
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
conn = sqlite3.connect(DB)
runs = pd.read_sql("SELECT DISTINCT run_id FROM events ORDER BY run_id DESC", conn)
run_id = st.sidebar.selectbox("Run ID", runs["run_id"].tolist() if len(runs) else ["run_001"])

symbols_df = pd.read_sql("SELECT DISTINCT symbol FROM bars WHERE run_id=? AND symbol!='' ORDER BY symbol",
                         conn, params=(run_id,))
symbol = st.sidebar.selectbox("Symbol", symbols_df["symbol"].tolist() if len(symbols_df) else [""])

bars = pd.read_sql(
    "SELECT * FROM bars WHERE run_id=? AND symbol=? ORDER BY ts",
    conn, params=(run_id, symbol)
)
fills = pd.read_sql(
    "SELECT * FROM fills WHERE run_id=? AND symbol=? ORDER BY ts",
    conn, params=(run_id, symbol)
)
snaps = pd.read_sql(
    "SELECT * FROM snapshots WHERE run_id=? ORDER BY ts",
    conn, params=(run_id,)
)
events = pd.read_sql(
    "SELECT * FROM events WHERE run_id=? ORDER BY ts",
    conn, params=(run_id,)
)


# optional orders table
orders = None
try:
    orders = pd.read_sql(
        "SELECT * FROM orders WHERE run_id=? AND symbol=? ORDER BY ts",
        conn,
        params=(run_id, symbol),
    )
except Exception:
    orders = None

for df in (bars, fills, snaps, events):
    if len(df) and "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

if orders is not None and len(orders) and "ts" in orders.columns:
    orders["ts"] = pd.to_datetime(orders["ts"], errors="coerce")

# explode + normalize events once
events_ex = explode_events_by_symbol(events)
import ipdb; ipdb.set_trace()
tab1, tab2, tab3, tab4 = st.tabs(["Price + Trades", "Trade Timeline", "Positions/Equity", "Tables"])


# =========================
# TAB 1: Price + Trades + Events
# =========================
with tab1:
    if len(bars):
        fig = make_price_trades_event_fig(
            bars_df=bars,
            fills_df=fills,
            events_df=events_ex,
            orders_df=orders,
            symbol=symbol,
        )
        st.plotly_chart(fig, width="stretch", key=f"tab1_price_{run_id}_{symbol}", config=PLOTLY_CONFIG)
    else:
        st.info("No bars found for this symbol. Make sure you log_bar() from MarketDataEvent.")


# =========================
# TAB 2: Trade Timeline (same as your previous; kept minimal here)
# =========================
with tab2:
    st.subheader("Raw Event Timeline (per bar)")
    if len(events_ex):
        # 这里用 scatter 展示所有事件，symbol 过滤可选
        ev2 = events_ex.copy()
        ev2 = ev2[(ev2["symbol"].astype(str) == str(symbol)) | (ev2["symbol"].astype(str) == "")]
        import ipdb; ipdb.set_trace()
        fig = go.Figure()
        for et in EVENT_ORDER:
            sub = ev2[ev2["etype"] == et]
            if len(sub):
                fig.add_trace(
                    go.Scattergl(
                        x=sub["ts"],
                        y=[et] * len(sub),
                        mode="markers",
                        name=et,
                        marker=dict(size=7),
                        hovertext=sub["hover"],
                        hoverinfo="text",
                    )
                )
        fig.update_layout(height=520, hovermode="closest", title="Events (exploded & normalized)")
        fig.update_yaxes(type="category", categoryorder="array", categoryarray=EVENT_ORDER)
        fig = apply_rangebreaks(fig, bars["ts"] if len(bars) else snaps["ts"])
        st.plotly_chart(fig, width="stretch", key=f"tab2_events_{run_id}_{symbol}", config=PLOTLY_CONFIG)
    else:
        st.info("No events found.")


# =========================
# TAB 3: Positions/Equity (+ Sharpe etc) + Price under Drawdown
# =========================
with tab3:
    if len(snaps):
        # ---- Metrics ----
        m = compute_performance_metrics(snaps)

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Total Return", f"{m['total_return']:.2%}")
        c2.metric("CAGR", f"{m['cagr']:.2%}")
        c3.metric("Sharpe", f"{m['sharpe']:.3f}" if np.isfinite(m["sharpe"]) else "NA")
        c4.metric("Annual Vol", f"{m['annual_vol']:.2%}" if np.isfinite(m["annual_vol"]) else "NA")
        c5.metric("Max Drawdown", f"{m['max_dd']:.2%}")
        c6.metric("Calmar", f"{m['calmar']:.3f}" if np.isfinite(m["calmar"]) else "NA")
        c7.metric("Ending Equity", f"{m['ending_equity']:,.2f}" if np.isfinite(m["ending_equity"]) else "NA")

        if len(bars):
            fig = make_equity_drawdown_price_fig(snaps, bars, fills, symbol)
            st.plotly_chart(fig, width="stretch", key=f"tab3_perf_{run_id}_{symbol}", config=PLOTLY_CONFIG)
        else:
            # fallback: only equity/drawdown
            eq = snaps["equity"].astype(float)
            peak = eq.cummax()
            dd = eq / peak - 1.0

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.04)
            fig.add_trace(go.Scatter(x=snaps["ts"], y=snaps["equity"], mode="lines", name="Equity"), row=1, col=1)
            fig.add_trace(go.Scatter(x=snaps["ts"], y=dd, mode="lines", name="Drawdown"), row=2, col=1)
            fig.update_layout(height=780, hovermode="x unified")
            fig = apply_rangebreaks(fig, snaps["ts"])
            st.plotly_chart(fig, width="stretch", key=f"tab3_perf_fallback_{run_id}", config=PLOTLY_CONFIG)
    else:
        st.info("No snapshots found. Make sure you log_snapshot() each bar.")


# =========================
# TAB 4: Tables
# =========================
with tab4:
    st.subheader("Fills")
    st.dataframe(fills, width="stretch", height=220)

    st.subheader("Orders")
    if orders is not None:
        st.dataframe(orders, width="stretch", height=220)
    else:
        st.info("No orders table or no orders logged yet.")

    st.subheader("Events (raw)")
    st.dataframe(events.tail(300), width="stretch", height=260)

    st.subheader("Events (exploded & normalized)")
    if len(events_ex):
        st.dataframe(events_ex.tail(300), width="stretch", height=260)



