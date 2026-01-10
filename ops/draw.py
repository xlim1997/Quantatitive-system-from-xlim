import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

def plot_candles_with_volume(df_all: pd.DataFrame, symbol: str,
                             start=None, end=None, title=None):
    """
    df_all: MultiIndex (date, symbol) OR a single-symbol df indexed by date
            columns need: open, high, low, close, volume
    """
    # --- select symbol if MultiIndex ---
    if isinstance(df_all.index, pd.MultiIndex):
        df = df_all.xs(symbol, level="symbol").copy()
    else:
        df = df_all.copy()

    df = df.sort_index()

    if start is not None:
        df = df.loc[pd.to_datetime(start):]
    if end is not None:
        df = df.loc[:pd.to_datetime(end)]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ensure datetime index
    df.index = pd.to_datetime(df.index)

    # convert dates to matplotlib numbers
    x = mdates.date2num(df.index.to_pydatetime())
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)

    # candle width: based on median spacing
    if len(x) >= 2:
        w = np.median(np.diff(x)) * 0.7
    else:
        w = 0.6  # fallback

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    # --- candlesticks ---
    for xi, oi, hi, li, ci in zip(x, o, h, l, c):
        # wick
        ax1.vlines(xi, li, hi)

        # body
        y0 = min(oi, ci)
        body_h = abs(ci - oi)
        # if body_h == 0, draw a tiny body so it is visible
        if body_h == 0:
            body_h = (hi - li) * 0.001 if hi > li else 0.001

        # Use filled vs hollow to indicate down/up WITHOUT explicitly setting colors:
        # up (close>=open): hollow; down: filled
        is_up = (ci >= oi)
        rect = Rectangle(
            (xi - w/2, y0),
            w,
            body_h,
            fill=not is_up
        )
        ax1.add_patch(rect)

    # --- volume ---
    ax2.bar(x, v, width=w)

    # formatting
    ax1.set_ylabel("Price")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")

    ax1.xaxis_date()
    ax2.xaxis_date()
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))

    plt.setp(ax1.get_xticklabels(), visible=False)

    if title is None:
        title = f"{symbol} Candles + Volume"
    ax1.set_title(title)

    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

def plot_tv_style(df_all: pd.DataFrame,
                  symbol: str,
                  start=None,
                  end=None,
                  vol_sma_window: int = 9,
                  price_ma_window: int | None = None):
    """
    df_all: MultiIndex (date, symbol) or single-symbol df indexed by date
            columns: open, high, low, close, volume
    """

    # --- pick symbol ---
    if isinstance(df_all.index, pd.MultiIndex):
        df = df_all.xs(symbol, level="symbol").copy()
    else:
        df = df_all.copy()

    df = df.sort_index()
    df.index = pd.to_datetime(df.index)

    if start is not None:
        df = df.loc[pd.to_datetime(start):]
    if end is not None:
        df = df.loc[:pd.to_datetime(end)]

    req = {"open", "high", "low", "close", "volume"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # --- colors (TradingView-like) ---
    up_color = "#26a69a"    # green-teal
    down_color = "#ef5350"  # red
    grid_color = "#e9ecef"

    # --- data to mpl ---
    x = mdates.date2num(df.index.to_pydatetime())
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)

    is_up = c >= o
    colors = np.where(is_up, up_color, down_color)

    # candle width based on time spacing
    if len(x) >= 2:
        w = np.median(np.diff(x)) * 0.70
    else:
        w = 0.6

    # --- smooth lines ---
    vol_sma = pd.Series(v, index=df.index).rolling(vol_sma_window, min_periods=1).mean().to_numpy()

    if price_ma_window is not None:
        price_ma = pd.Series(c, index=df.index).rolling(price_ma_window, min_periods=1).mean().to_numpy()
    else:
        price_ma = None

    # --- figure layout ---
    fig = plt.figure(figsize=(14, 7), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.04)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    # grid like TradingView
    for ax in (ax1, ax2):
        ax.grid(True, which="major", linestyle="-", linewidth=0.6, color=grid_color)
        ax.tick_params(axis="both", which="both", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    # --- candlesticks ---
    wick_lw = 0.8
    body_lw = 0.8

    for xi, oi, hi, li, ci, col in zip(x, o, h, l, c, colors):
        # wick
        ax1.vlines(xi, li, hi, linewidth=wick_lw, color=col, alpha=0.95)

        # body
        y0 = min(oi, ci)
        bh = abs(ci - oi)
        if bh == 0:
            bh = (hi - li) * 0.001 if hi > li else 0.001

        rect = Rectangle(
            (xi - w / 2, y0),
            w,
            bh,
            facecolor=col,
            edgecolor=col,
            linewidth=body_lw
        )
        ax1.add_patch(rect)

    # optional price MA (if you want even more "smooth")
    if price_ma is not None:
        ax1.plot(x, price_ma, linewidth=1.2, alpha=0.9)

    ax1.set_title(f"{symbol}", loc="left", fontsize=12, pad=8)
    ax1.set_ylabel("Price", fontsize=10)

    # --- volume (colored) + SMA line ---
    ax2.bar(x, v, width=w, color=colors, alpha=0.35, align="center")
    ax2.plot(x, vol_sma, linewidth=1.4, alpha=0.95)  # smooth line
    ax2.set_ylabel("Volume", fontsize=10)

    # x-axis formatting
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.setp(ax1.get_xticklabels(), visible=False)

    # margins
    ax1.margins(x=0)
    ax2.margins(x=0)

    plt.show()
    
def plot_tv_style_with_mas(df_all: pd.DataFrame,
                           symbol: str,
                           start=None,
                           end=None,
                           vol_sma_window: int = 9,
                           ma_windows=(5, 10, 20)):
    # --- pick symbol ---
    if isinstance(df_all.index, pd.MultiIndex):
        df = df_all.xs(symbol, level="symbol").copy()
    else:
        df = df_all.copy()

    df = df.sort_index()
    df.index = pd.to_datetime(df.index)

    if start is not None:
        df = df.loc[pd.to_datetime(start):]
    if end is not None:
        df = df.loc[:pd.to_datetime(end)]

    req = {"open", "high", "low", "close", "volume"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # --- colors (TradingView-like) ---
    up_color = "#26a69a"
    down_color = "#ef5350"
    grid_color = "#e9ecef"

    # --- data to mpl ---
    x = mdates.date2num(df.index.to_pydatetime())
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)

    is_up = c >= o
    colors = np.where(is_up, up_color, down_color)

    # candle width
    if len(x) >= 2:
        w = np.median(np.diff(x)) * 0.70
    else:
        w = 0.6

    # --- smooth lines ---
    vol_sma = pd.Series(v, index=df.index).rolling(vol_sma_window, min_periods=1).mean()

    mas = {}
    for win in ma_windows:
        mas[win] = pd.Series(c, index=df.index).rolling(win, min_periods=1).mean()

    # --- figure layout ---
    fig = plt.figure(figsize=(14, 7), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.04)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    # grid + clean spines
    for ax in (ax1, ax2):
        ax.grid(True, which="major", linestyle="-", linewidth=0.6, color=grid_color)
        ax.tick_params(axis="both", which="both", length=0)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)

    # --- candlesticks ---
    for xi, oi, hi, li, ci, col in zip(x, o, h, l, c, colors):
        ax1.vlines(xi, li, hi, linewidth=0.8, color=col, alpha=0.95)

        y0 = min(oi, ci)
        bh = abs(ci - oi)
        if bh == 0:
            bh = (hi - li) * 0.001 if hi > li else 0.001

        ax1.add_patch(Rectangle(
            (xi - w/2, y0),
            w,
            bh,
            facecolor=col,
            edgecolor=col,
            linewidth=0.8
        ))

    # --- MA lines (5/10/20) ---
    # 不强制指定颜色，让 matplotlib 自己分配（你要固定颜色我也能给）
    for win in ma_windows:
        ax1.plot(df.index, mas[win], linewidth=1.2, label=f"MA{win}", alpha=0.95)

    ax1.set_title(f"{symbol}", loc="left", fontsize=12, pad=8)
    ax1.set_ylabel("Price", fontsize=10)

    # legend（放左上角，尽量不挡图）
    ax1.legend(loc="upper left", frameon=False, fontsize=9)

    # --- volume + SMA line ---
    ax2.bar(df.index, v, width=w, color=colors, alpha=0.35, align="center")
    ax2.plot(df.index, vol_sma, linewidth=1.4, alpha=0.95, label=f"Vol SMA {vol_sma_window}")
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    ax2.set_ylabel("Volume", fontsize=10)

    # x-axis formatting
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax1.margins(x=0)
    ax2.margins(x=0)
    plt.show()


# 用法：
# plot_tv_style(data_info, "NVDA", vol_sma_window=9)
# 加个价格均线也更“smooth”：
# plot_tv_style(data_info, "NVDA", vol_sma_window=9, price_ma_window=20)

import pandas as pd

def _enum_to_str(x):
    if x is None:
        return None
    return getattr(x, "name", None) or str(x)

def insights_to_long_df(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    results_df: 回测 df（必须有 timestamp 和 insights(list)）
    return: 长表（每条 insight 一行），包含 date + symbol 列，并 set_index=(date,symbol)
    """
    # ✅ 这里把 symbol（如果存在）也先保留；没有也不影响
    keep_cols = [c for c in ["timestamp", "symbol", "insights"] if c in results_df.columns]
    tmp = results_df[keep_cols].copy()

    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
    tmp["insights"] = tmp["insights"].apply(lambda x: x if isinstance(x, list) else [])

    ex = tmp.explode("insights")
    ex = ex.dropna(subset=["insights"])

    def one(ins):
        return {
            "symbol": getattr(ins, "symbol", None),  # ✅ 从 Insight 提取 symbol（关键）
            "direction": _enum_to_str(getattr(ins, "direction", None)),
            "confidence": getattr(ins, "confidence", None),
            "magnitude": getattr(ins, "magnitude", None),
            "weight": getattr(ins, "weight", None),
            "expiry": getattr(ins, "expiry", None),
            "source": getattr(ins, "source", None),
        }

    norm = pd.json_normalize(ex["insights"].map(one))

    out = pd.concat(
        [
            ex[["timestamp"]].reset_index(drop=True),
            norm.reset_index(drop=True),
        ],
        axis=1
    )

    out = out.rename(columns={"timestamp": "date"})
    out = out.dropna(subset=["symbol"])  # ✅ 确保有 symbol

    # ✅ 你要“包含 symbol”：这里既作为列保留，也作为 index 一部分
    out = out.set_index(["date", "symbol"]).sort_index()
    out = out.reset_index()  # 先变回列（方便你查看/merge）
    return out

def merge_price_with_insights(price_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    price_df: data_info (index=(date,symbol))
    results_df: 回测 df（timestamp + insights）
    """
    ins_long = insights_to_long_df(results_df)  # columns: date, symbol, direction, ...

    # 如果同一天同一symbol 多条 insight -> 先聚合，否则 join 会产生重复行
    ins_agg = (
        ins_long
        .set_index(["date", "symbol"])
        .groupby(level=["date", "symbol"])
        .first()
    )

    merged = price_df.join(ins_agg, how="left")
    merged["has_insight"] = merged["direction"].notna()
    return merged


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


# -------------------------
# Helpers: direction -> side
# -------------------------
def _enum_to_str(x):
    if x is None:
        return None
    return getattr(x, "name", None) or str(x)

def _dir_to_side(direction) -> str | None:
    """
    把各种 direction（enum / str / repr）映射到: 'up' / 'down' / 'flat' / None
    你可以按你自己的 InsightDirection 名字再补充关键词。
    """
    if direction is None or (isinstance(direction, float) and np.isnan(direction)):
        return None

    s = _enum_to_str(direction).upper()

    # entry / long
    if any(k in s for k in ["UP", "LONG", "BUY", "BULL"]):
        return "up"

    # exit / short / sell
    if any(k in s for k in ["DOWN", "SHORT", "SELL", "BEAR"]):
        return "down"

    # flat / close position / no position
    if any(k in s for k in ["FLAT", "CLOSE", "EXIT", "NONE", "NEUTRAL"]):
        return "flat"

    return None

def _side_from_cell(x) -> str | None:
    """
    df['direction'] 可能是:
      - 一个值（enum/str）
      - 或者 list/tuple（一天多条 insight 聚合后）
    这里取“第一条能识别的 side”。
    """
    if isinstance(x, (list, tuple)):
        for item in x:
            side = _dir_to_side(item)
            if side is not None:
                return side
        return None
    return _dir_to_side(x)


# -------------------------
# State-machine filter
# -------------------------
def filter_entry_exit_signals(
    sides: pd.Series,
    entry_side: str = "up",
    exit_sides=("flat", "down"),
):
    """
    只画“第一次 entry”，直到遇到 exit，再允许下次 entry。
    """
    holding = False
    entry_mask = pd.Series(False, index=sides.index)
    exit_mask  = pd.Series(False, index=sides.index)

    for dt, s in sides.items():
        if s is None:
            continue

        if (not holding) and (s == entry_side):
            entry_mask.loc[dt] = True
            holding = True
        elif holding and (s in exit_sides):
            exit_mask.loc[dt] = True
            holding = False

    return entry_mask, exit_mask


# -------------------------
# Plot
# -------------------------
def plot_tv_style_with_mas_and_insights(
    df_all: pd.DataFrame,
    symbol: str,
    start=None,
    end=None,
    ma_windows=(5, 10, 20),
    vol_sma_window=9,
    show_insight_text=False,
):
    """
    df_all: MultiIndex (date, symbol) 的 merged_df
            columns 至少包含 open/high/low/close/volume
            可选包含 direction（用于画买卖点）
    """

    # --- pick symbol ---
    if isinstance(df_all.index, pd.MultiIndex):
        df = df_all.xs(symbol, level="symbol").copy()
    else:
        df = df_all.copy()

    df = df.sort_index()
    df.index = pd.to_datetime(df.index)

    if start is not None:
        df = df.loc[pd.to_datetime(start):]
    if end is not None:
        df = df.loc[:pd.to_datetime(end)]

    req = {"open", "high", "low", "close", "volume"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # TradingView-like colors
    up_color = "#26a69a"
    down_color = "#ef5350"
    grid_color = "#e9ecef"

    # --- data to mpl ---
    x = mdates.date2num(df.index.to_pydatetime())
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)

    is_up = c >= o
    colors = np.where(is_up, up_color, down_color)

    # candle width (in "days")
    w = (np.median(np.diff(x)) * 0.70) if len(x) >= 2 else 0.6

    # MAs
    ma_series = {
        win: pd.Series(c, index=df.index).rolling(win, min_periods=1).mean()
        for win in ma_windows
    }

    # Volume SMA
    vol_sma = pd.Series(v, index=df.index).rolling(vol_sma_window, min_periods=1).mean()

    # --- layout ---
    fig = plt.figure(figsize=(14, 7), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.04)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.grid(True, which="major", linestyle="-", linewidth=0.6, color=grid_color)
        ax.tick_params(axis="both", which="both", length=0)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)

    # --- candles ---
    for xi, oi, hi, li, ci, col in zip(x, o, h, l, c, colors):
        # wick
        ax1.vlines(xi, li, hi, linewidth=0.8, color=col, alpha=0.95)

        # body
        y0 = min(oi, ci)
        bh = abs(ci - oi)
        if bh == 0:
            bh = (hi - li) * 0.001 if hi > li else 0.001

        ax1.add_patch(
            Rectangle(
                (xi - w / 2, y0),
                w,
                bh,
                facecolor=col,
                edgecolor=col,
                linewidth=0.8,
            )
        )

    # --- MA lines ---
    for win in ma_windows:
        ax1.plot(df.index, ma_series[win], linewidth=1.2, alpha=0.95, label=f"MA{win}")

    ax1.set_title(f"{symbol}", loc="left", fontsize=12, pad=8)
    ax1.set_ylabel("Price", fontsize=10)

    # --- insights markers (state-machine filtered) ---
    if "direction" in df.columns:
        sides = df["direction"].map(_side_from_cell)

        # ✅ 只画：第一次 UP（entry） -> 直到遇到 FLAT/DOWN（exit） -> 再允许下一次 UP
        entry_mask, exit_mask = filter_entry_exit_signals(
            sides,
            entry_side="up",
            exit_sides=("flat", "down"),
        )
        # import ipdb; ipdb.set_trace()

        buy_y = df["low"] * 0.995
        sell_y = df["high"] * 1.005

        ax1.scatter(
            df.index[entry_mask],
            buy_y[entry_mask],
            marker="^",
            s=70,
            color=up_color,
            edgecolors="none",
            label="Entry (first UP)",
            zorder=5,
        )
        ax1.scatter(
            df.index[exit_mask],
            sell_y[exit_mask],
            marker="v",
            s=70,
            color=down_color,
            edgecolors="none",
            label="Exit (FLAT/DOWN)",
            zorder=5,
        )

        if show_insight_text:
            for dt in df.index[entry_mask]:
                ax1.annotate(
                    "BUY",
                    (dt, float(buy_y.loc[dt])),
                    xytext=(0, 6),
                    textcoords="offset points",
                    fontsize=8,
                    ha="center",
                )
            for dt in df.index[exit_mask]:
                ax1.annotate(
                    "SELL",
                    (dt, float(sell_y.loc[dt])),
                    xytext=(0, -10),
                    textcoords="offset points",
                    fontsize=8,
                    ha="center",
                )

    # legend（合并重复项）
    handles, labels = ax1.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax1.legend(uniq.values(), uniq.keys(), loc="upper left", frameon=False, fontsize=9)

    # --- volume + sma ---
    ax2.bar(x, v, width=w, color=colors, alpha=0.35, align="center")
    ax2.plot(df.index, vol_sma, linewidth=1.4, alpha=0.95, label=f"Vol SMA {vol_sma_window}")
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    ax2.set_ylabel("Volume", fontsize=10)

    # x axis formatting
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax1.margins(x=0)
    ax2.margins(x=0)
    plt.show()


# -------------------------
# Example
# -------------------------
# merged_df = merge_price_with_insights(data_info, df[:400])
# plot_tv_style_with_mas_and_insights(merged_df, "NVDA", vol_sma_window=9)
# plot_tv_style_with_mas_and_insights(merged_df, "NVDA", start="2024-01-01", end="2024-12-31")
