# run_volume_contraction_algo_backtest.py
from __future__ import annotations

from backtesting.backtest import Backtest
from backtesting.performance import compute_performance

from portfolio.construction import WeightedByHintPC
from portfolio.execution import ImmediateExecutionModel

from strategies.fisher_slope_ema_long_entry import (
    FisherSlopeEmaEntryConfig,
    FisherSlopeEmaLongEntryStrategy
)
from data.ibkr_feed import IBKRHistoryBarDataFeed, IBKRContractSpec, IBKRConnConfig_Historical
from backtesting.trade_stats import (
    compute_turnover,
    summarize_trades,
    make_buy_and_hold_equity,
    compute_active_return_and_ir,
    compute_alpha_beta,
)
from portfolio.risk import (
    ChainRiskManagementModel,
    StopLossRiskModel,
    MaxPositionWeightRiskModel,
    MaxGrossExposureRiskModel,
    PortfolioMaxDrawdownRiskModel,
)
from portfolio.state import Portfolio
from brokerage.paper import PaperBrokerage
from universe.symbol_list import symbols_nasdaq_100, symbols_sp500
import pandas as pd
import numpy as np
from ops.draw import merge_price_with_insights,insights_to_long_df,plot_tv_style_with_mas,plot_tv_style_with_mas_and_insights


pd.set_option("display.max_rows", None)          # 不截断行
pd.set_option("display.max_colwidth", None)      # 不截断每个单元格字符串
def main():
    # 1) Universe（先用 5-20 个股票跑通，后面再扩）
    universe = symbols_nasdaq_100[:1] # nasdaq100, SP500, dow30, custom
    universe = ["MSTR"]
    print("Universe:", universe)
    # 2) DataFeed for selected universe
    contracts = {sym: IBKRContractSpec(symbol=sym) for sym in universe}
    data_feed = IBKRHistoryBarDataFeed(
        contracts=contracts,
        duration_str="6 Y",
        bar_size="1 day",
        what_to_show="TRADES",
        use_rth=True,
        end_datetime="",
        conn=IBKRConnConfig_Historical(
            host="127.0.0.1",
            port=7497,      # TWS paper常见 7497；live常见 7496（看你设置）
            client_id=1,
        ),
    )

    
    cfg = FisherSlopeEmaEntryConfig(
        universe=universe,
    
    )

    # 2) Strategy
    algo = FisherSlopeEmaLongEntryStrategy(cfg)
    # 3) 三模型
    pc_model = WeightedByHintPC(normalize_gross=True, gross_cap=1.0)
    risk_model = ChainRiskManagementModel([
        StopLossRiskModel(stop_loss_pct=0.05),  # 风控层退出
        PortfolioMaxDrawdownRiskModel(max_drawdown=0.1),             # 组合回撤保护
        MaxPositionWeightRiskModel(max_weight=1.0),                   # 单票限制
        MaxGrossExposureRiskModel(max_gross_exposure=1.0),            # 总曝险限制
    ])
    # 你也可以叠加单票限制：先用 MaxPositionWeightRiskModel 放在 risk.py 里串联（后面我给你做组合 risk chain）
    exec_model = exec_model = ImmediateExecutionModel(
        min_trade_value=65.0,
        cash_buffer_pct=0.002,       # 留 0.2% 现金防止误差（可调）
        commission_per_share=0.005,  # IBKR fixed 常见 0.005/share
        min_commission=1.0,          # min $1 / order
        allow_margin=False,
    )
    
    #
    # 
    
    # 5) Backtest 引擎
    
    bt = Backtest(
        algorithm=algo,
        data_feed=data_feed,
        symbol_to_path=None,  # 已经传 data_feed 了就不需要 symbol_to_path
        pc_model=pc_model,
        risk_model=risk_model,
        exec_model=exec_model,
        initial_cash=5_000.0,
        slippage=0.0005,
        commission_model="value_pct",
        commission_rate=0.0002,   # 例如 0.02%
        fixed_commission=0.0,
        keep_insights_active=True,
    )



    df = bt.run()
    # import ipdb; ipdb.set_trace()
    # ✅ 交易统计（需要 engine logs + portfolio trade log）
    # import ipdb; ipdb.set_trace()
    def bars_by_symbol_to_df(bars_by_symbol: dict) -> pd.DataFrame:
        def one_symbol(item):
            sym, bars = item
            df = pd.DataFrame.from_records([
                {
                    "date": pd.to_datetime(getattr(b, "date", None)),
                    "open": float(getattr(b, "open", np.nan)),
                    "high": float(getattr(b, "high", np.nan)),
                    "low": float(getattr(b, "low", np.nan)),
                    "close": float(getattr(b, "close", np.nan)),
                    "volume": float(getattr(b, "volume", np.nan)),
                    "average": float(getattr(b, "average", np.nan)) if getattr(b, "average", None) is not None else np.nan,
                    "barCount": int(getattr(b, "barCount", np.nan)) if getattr(b, "barCount", None) is not None else np.nan,
                    "symbol": sym,
                }
                for b in bars
            ])
            return df

        df = pd.concat(map(one_symbol, bars_by_symbol.items()), ignore_index=True)
        df = df.set_index(["date", "symbol"]).sort_index()
        return df
    
    data_info = bars_by_symbol_to_df(data_feed._bars_by_symbol)

    #conver data_feed bars to dataframe
    merged_df = merge_price_with_insights(data_info, df)
    # plot_tv_style_with_mas_and_insights(merged_df, universe[0], vol_sma_window=9)
    # import ipdb; ipdb.set_trace()
    turn = compute_turnover(bt.fill_log, df["equity"])
    trade_summary = summarize_trades(bt.portfolio.trade_log)
    
    
    print("\n=== Turnover ===")
    print(turn)
    print("\n=== Trade Stats ===")
    print(trade_summary)
    # ✅ 绩效指标
    perf = compute_performance(df)
    print("\n=== Performance ===")
    print(perf)
   
    import ipdb; ipdb.set_trace()

    

if __name__ == "__main__":
    main()
