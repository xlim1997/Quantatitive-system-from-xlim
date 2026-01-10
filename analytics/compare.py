# analytics/trade_dashboard.py
import sqlite3
import json
import numpy as np
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

tab1, tab2, tab3, tab4 = st.tabs(["Price + Trades", "Trade Timeline", "Positions/Equity", "Tables"])

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

    # ---- Decode event payloads (INSIGHT meta / RISK_ACTION reasons) ----
    if len(events):
        def _safe_json(x):
            if x is None:
                return {}
            if isinstance(x, (dict, list)):
                return x
            try:
                return json.loads(x)
            except Exception:
                return {}

        events2 = events.copy()
        events2["payload_json"] = events2.get("payload", pd.Series([None]*len(events2))).apply(_safe_json)

        insight_df = events2[events2["etype"] == "INSIGHT"].copy()
        risk_df    = events2[events2["etype"] == "RISK_ACTION"].copy()

        # parse INSIGHT direction + meta
        def _get(d, k, default=None):
            return d.get(k, default) if isinstance(d, dict) else default

        if len(insight_df):
            insight_df["direction"] = insight_df["payload_json"].apply(lambda d: _get(d, "direction"))
            insight_df["meta"]      = insight_df["payload_json"].apply(lambda d: _get(d, "meta", {}))
            insight_df["source"]    = insight_df["payload_json"].apply(lambda d: _get(d, "source"))

        if len(risk_df):
            risk_df["reason"] = risk_df["payload_json"].apply(lambda d: _get(d, "reason"))
            risk_df["action"] = risk_df["payload_json"].apply(lambda d: _get(d, "action"))

        # ---- Build trades from fills ----
        trade_rows = []
        if len(fills):
            ff = fills.copy()
            ff = ff.sort_values("ts")
            # signed qty
            def _signed_qty(row):
                side = str(row.get("side","")).lower()
                q = float(row.get("quantity",0.0))
                return q if side == "buy" else (-q if side == "sell" else 0.0)
            ff["signed_qty"] = ff.apply(_signed_qty, axis=1)
            ff["pos"] = ff["signed_qty"].cumsum()

            in_trade = False
            entry_i = None
            entry_pos = 0.0

            for i, r in ff.iterrows():
                prev_pos = entry_pos if in_trade else 0.0
                cur_pos = float(r["pos"])

                if (not in_trade) and prev_pos == 0.0 and cur_pos > 0.0:
                    in_trade = True
                    entry_i = i
                    entry_pos = cur_pos
                elif in_trade and cur_pos == 0.0:
                    # close trade at this fill
                    entry_slice = ff.loc[entry_i:i]
                    buys = entry_slice[entry_slice["signed_qty"] > 0]
                    sells = entry_slice[entry_slice["signed_qty"] < 0]

                    entry_ts = buys["ts"].iloc[0] if len(buys) else entry_slice["ts"].iloc[0]
                    exit_ts  = sells["ts"].iloc[-1] if len(sells) else entry_slice["ts"].iloc[-1]

                    entry_qty = float(buys["signed_qty"].sum()) if len(buys) else float(entry_slice["signed_qty"].sum())
                    exit_qty  = float((-sells["signed_qty"]).sum()) if len(sells) else float(entry_qty)

                    entry_avg = float((buys["price"].astype(float) * buys["signed_qty"].astype(float)).sum() / max(entry_qty, 1e-12)) if len(buys) else float("nan")
                    exit_avg  = float((sells["price"].astype(float) * (-sells["signed_qty"].astype(float))).sum() / max(exit_qty, 1e-12)) if len(sells) else float("nan")

                    pnl = (exit_avg - entry_avg) * min(entry_qty, exit_qty) if (not np.isnan(entry_avg) and not np.isnan(exit_avg)) else np.nan
                    hold_days = (exit_ts - entry_ts).days if pd.notna(entry_ts) and pd.notna(exit_ts) else None

                    # attach entry trigger: nearest INSIGHT UP around entry_ts (same/previous day)
                    entry_reason = ""
                    if len(insight_df) and pd.notna(entry_ts):
                        cand = insight_df[(insight_df["symbol"] == symbol) & (insight_df["ts"] <= entry_ts + pd.Timedelta(days=1))]
                        cand = cand.sort_values("ts")
                        if len(cand):
                            last = cand.iloc[-1]
                            meta = last.get("meta", {})
                            if isinstance(meta, dict) and meta:
                                conds = meta.get("conds", {})
                                vals = meta.get("values", {})
                                entry_reason = f'{meta.get("rule","")} | conds={conds} | values={ {k: vals.get(k) for k in ["fisher_slope","fisher_slope_eps","ema5_slope","ema5_slope_eps","ema_fast","ema_mid","ema_slow","close"]} }'

                    # attach exit trigger: nearest RISK_ACTION before exit
                    exit_reason = ""
                    if len(risk_df) and pd.notna(exit_ts):
                        cand = risk_df[(risk_df["symbol"] == symbol) & (risk_df["ts"] <= exit_ts + pd.Timedelta(days=1))]
                        cand = cand.sort_values("ts")
                        if len(cand):
                            last = cand.iloc[-1]
                            exit_reason = str(last.get("payload_json"))
                    # fallback: if strategy itself emits FLAT, show that reason
                    if (not exit_reason) and len(insight_df) and pd.notna(exit_ts):
                        cand2 = insight_df[(insight_df["symbol"] == symbol) & (insight_df["ts"] <= exit_ts + pd.Timedelta(days=1))]
                        cand2 = cand2.sort_values("ts")
                        if len(cand2):
                            last2 = cand2.iloc[-1]
                            if str(last2.get("direction","")).upper() in ["FLAT","0"]:
                                meta2 = last2.get("meta", {})
                                exit_reason = f"STRATEGY_FLAT | meta={meta2}"


                    trade_rows.append({
                        "symbol": symbol,
                        "entry_ts": entry_ts,
                        "entry_avg": entry_avg,
                        "qty": min(entry_qty, exit_qty),
                        "entry_reason": entry_reason,
                        "exit_ts": exit_ts,
                        "exit_avg": exit_avg,
                        "exit_reason": exit_reason,
                        "pnl": pnl,
                        "holding_days": hold_days,
                    })

                    in_trade = False
                    entry_i = None
                    entry_pos = 0.0

        st.subheader("Trade Timeline (Human-readable)")
        if trade_rows:
            tdf = pd.DataFrame(trade_rows)
            st.dataframe(tdf, use_container_width=True, height=260)

            st.markdown("### Chronological log")
            for _, tr in tdf.iterrows():
                e = tr["entry_ts"].date() if pd.notna(tr["entry_ts"]) else tr["entry_ts"]
                x = tr["exit_ts"].date() if pd.notna(tr["exit_ts"]) else tr["exit_ts"]
                st.markdown(
                    f"- **{e}** BUY @{tr['entry_avg']:.4f} qty={tr['qty']:.0f}\n"
                    f"  Trigger: {tr['entry_reason']}"
                )
                st.markdown(
                    f"  **{x}** EXIT @{tr['exit_avg']:.4f}  PnL={tr['pnl']:.2f}\n"
                    f"  Trigger: {tr['exit_reason']}"
                )
        else:
            st.info("No complete round-trip trades reconstructed from fills yet (need buy then sell back to flat).")

        # ---- Raw event timeline scatter (optional) ----
        fig = px.scatter(events2, x="ts", y="etype", hover_data=["symbol"])
        fig.update_layout(height=420, title="Raw Event Timeline (per bar)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No events found.")

with tab3:
    if len(snaps):
        # --- Key stats ---
        eq = snaps["equity"].astype(float)
        peak = eq.cummax()
        dd = eq / peak - 1.0
        max_dd = float(dd.min()) if len(dd) else 0.0
        total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 and float(eq.iloc[0]) != 0.0 else 0.0

        s1, s2, s3 = st.columns(3)
        s1.metric("Total Return", f"{total_ret:.2%}")
        s2.metric("Max Drawdown", f"{max_dd:.2%}")
        s3.metric("Ending Equity", f"{float(eq.iloc[-1]):,.2f}")

        c1, c2 = st.columns(2)

        with c1:
            fig = px.line(snaps, x="ts", y="equity", title="Equity")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
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