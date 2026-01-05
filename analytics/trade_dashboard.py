# analytics/trade_dashboard.py
import sqlite3
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Trading Process Dashboard")

DB = st.sidebar.text_input("DB path", "artifacts/trades.db")

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

for df in (bars, fills, snaps, events):
    if len(df) and "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

tab1, tab2, tab3, tab4 = st.tabs(["Price + Trades", "Timeline", "Positions/Equity", "Tables"])

with tab1:
    if len(bars):
        fig = go.Figure(data=[go.Candlestick(
            x=bars["ts"], open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"],
            name="OHLC"
        )])

        if len(fills):
            buys = fills[fills["side"].str.upper() == "BUY"]
            sells = fills[fills["side"].str.upper() == "SELL"]

            fig.add_trace(go.Scatter(
                x=buys["ts"], y=buys["price"],
                mode="markers", name="BUY",
                marker=dict(symbol="triangle-up", size=10)
            ))
            fig.add_trace(go.Scatter(
                x=sells["ts"], y=sells["price"],
                mode="markers", name="SELL",
                marker=dict(symbol="triangle-down", size=10)
            ))

        fig.update_layout(height=520, xaxis_rangeslider_visible=False, title=f"{symbol} - Trades Overlay")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No bars found for this symbol. Make sure you log_bar() from MarketDataEvent.")

with tab2:
    if len(events):
        # 用散点时间线复盘：不同 etype 不同颜色
        fig = px.scatter(events, x="ts", y="etype", hover_data=["symbol"])
        fig.update_layout(height=520, title="Event Timeline (Signal/Order/Fill/Snapshot...)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No events found.")

with tab3:
    if len(snaps):
        c1, c2 = st.columns(2)

        with c1:
            fig = px.line(snaps, x="ts", y="equity", title="Equity")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            eq = snaps["equity"].astype(float)
            peak = eq.cummax()
            dd = eq / peak - 1.0
            dd_df = pd.DataFrame({"ts": snaps["ts"], "drawdown": dd})
            fig = px.line(dd_df, x="ts", y="drawdown", title="Drawdown")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No snapshots found. Make sure you log_snapshot() each bar.")

with tab4:
    st.subheader("Fills")
    st.dataframe(fills, use_container_width=True, height=220)
    st.subheader("Orders")
    try:
        orders = pd.read_sql("SELECT * FROM orders WHERE run_id=? AND symbol=? ORDER BY ts",
                             conn, params=(run_id, symbol))
        if len(orders):
            orders["ts"] = pd.to_datetime(orders["ts"], errors="coerce")
        st.dataframe(orders, use_container_width=True, height=220)
    except Exception:
        st.info("No orders table or no orders logged yet.")

    st.subheader("Events (raw)")
    st.dataframe(events.tail(300), use_container_width=True, height=260)
