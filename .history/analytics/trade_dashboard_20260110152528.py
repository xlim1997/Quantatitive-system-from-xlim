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



