# run_momentum_backtest.py
from __future__ import annotations

from backtesting.backtest import Backtest
from backtesting.performance import compute_performance

from portfolio.construction import WeightedByHintPC
from portfolio.execution import ImmediateExecutionModel

from strategies.CrossSectionalMomentumStrategy import CrossSectionalMomentumStrategy, MomentumConfig
from data.ibkr_feed import IBKRHistoryBarDataFeed, IBKRContractSpec, IBKRConnConfig_Historical
from data.futu_feed import FutuHistoryKlineDataFeed, FutuConnConfig
from backtesting.trade_stats import compute_turnover, summarize_trades
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
def main():
    # 1) Universe（先用 5-20 个股票跑通，后面再扩）
    universe = [
        "AAPL", "MSFT", "AMZN", "GOOG", "META","NFLX","NVDA","TSLA",
        # "NDX"
        # 你可以继续加：NVDA, TSLA, AMD ...
    ]
    # universe = symbols_nasdaq_100[:5]
    # 2) Strategy
    cfg = MomentumConfig(
        universe=universe,
        lookback=60,
        rebalance_every=5,
        top_k=3,
        bottom_k=0,           # 先只做多；想多空就设成 3
        weight_mode="equal",  # 或 "score"
        min_mom_abs=0.0,
    )
    algo = CrossSectionalMomentumStrategy(cfg)

    # 3) 三模型
    pc_model = WeightedByHintPC(normalize_gross=True, gross_cap=1.0)
    risk_model = ChainRiskManagementModel([
        StopLossRiskModel(stop_loss_pct=0.02),  # 风控层退出
        PortfolioMaxDrawdownRiskModel(max_drawdown=0.08),             # 组合回撤保护
        MaxPositionWeightRiskModel(max_weight=0.5),                  # 单票限制
        MaxGrossExposureRiskModel(max_gross_exposure=1.0),            # 总曝险限制
    ])
    # 你也可以叠加单票限制：先用 MaxPositionWeightRiskModel 放在 risk.py 里串联（后面我给你做组合 risk chain）
    exec_model = ImmediateExecutionModel(min_trade_value=65.0)
    
    #
    # 4) DataFeed
    # IBKR DATA FEED
    contracts = {symbol: IBKRContractSpec(symbol=symbol) for symbol in universe}
    # 4) DataFeed (IBKR)
    # durationStr: Time span of all the bars. Examples:
    #             '60 S', '30 D', '13 W', '6 M', '10 Y'.
    # barSizeSetting: Time period of one bar. Must be one of:
    #             '1 secs', '5 secs', '10 secs' 15 secs', '30 secs',
    #             '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins',
    #             '20 mins', '30 mins',
    #             '1 hour', '2 hours', '3 hours', '4 hours', '8 hours',
    #             '1 day', '1 week', '1 month'.
    contracts = {sym: IBKRContractSpec(symbol=sym) for sym in universe}
    data_feed = IBKRHistoryBarDataFeed(
        contracts=contracts,
        duration_str="2 Y",
        bar_size="1 day",
        what_to_show="TRADES",
        use_rth=True,
        end_datetime="",  # "" means now
        conn=IBKRConnConfig_Historical(
            host="127.0.0.1",
            port=7497,      # TWS paper常见 7497；live常见 7496（看你设置）
            client_id=1,
        ),
    )
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
        commission_rate=0.005,
        fixed_commission=0.0,
        keep_insights_active=True,
    )



    df = bt.run()
    # import ipdb; ipdb.set_trace()
    # ✅ 交易统计（需要 engine logs + portfolio trade log）
    turn = compute_turnover(bt.fill_log, df["equity"])
    trade_summary = summarize_trades(bt.portfolio.trade_log)
    
    
    print("\n=== Turnover ===")
    for k, v in turn.items():
        print(f"{k:>20}: {v}")

    print("\n=== Trade Stats ===")
    for k, v in trade_summary.items():
        print(f"{k:>20}: {v}")
    # ✅ 绩效指标
    perf = compute_performance(df)
    print("\n=== Performance ===")
    for k, v in perf.items():
        print(f"{k:>20}: {v}")
    print("\n=== Equity Curve ===")
    # print(df[["timestamp", "equity"]])
    print(df)

if __name__ == "__main__":
    main()
