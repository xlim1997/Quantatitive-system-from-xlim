# run_momentum_backtest.py
from __future__ import annotations

from backtesting.engine import BacktestEngine,Engine
from backtesting.performance import compute_performance

from portfolio.construction import WeightedByHintPC
from portfolio.execution import ImmediateExecutionModel

from strategies.momentum.main import CrossSectionalMomentumStrategy, MomentumConfig

from backtesting.trade_stats import compute_turnover, summarize_trades
from portfolio.risk import (
    ChainRiskManagementModel,
    StopLossRiskModel,
    MaxPositionWeightRiskModel,
    MaxGrossExposureRiskModel,
    PortfolioMaxDrawdownRiskModel,
)

def main():
    # 1) Universe（先用 5-20 个股票跑通，后面再扩）
    universe = [
        "AAPL", "MSFT", "AMZN", "GOOG", "META",
        # 你可以继续加：NVDA, TSLA, AMD ...
    ]

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
        StopLossRiskModel(stop_loss_pct=0.10, take_profit_pct=None),  # 风控层退出
        PortfolioMaxDrawdownRiskModel(max_drawdown=0.25),             # 组合回撤保护
        MaxPositionWeightRiskModel(max_weight=0.25),                  # 单票限制
        MaxGrossExposureRiskModel(max_gross_exposure=1.0),            # 总曝险限制
    ])
    # 你也可以叠加单票限制：先用 MaxPositionWeightRiskModel 放在 risk.py 里串联（后面我给你做组合 risk chain）
    exec_model = ImmediateExecutionModel(min_trade_value=50.0)

    # 4) 数据（先用 CSV 示例；你换成 data_feed=FutuHistoryKlineDataFeed / IBKRHistoryBarDataFeed 即可）
    symbol_to_path = {sym: f"data/{sym.lower()}_daily.csv" for sym in universe}

    bt = Engine(
        algorithm=algo,
        symbol_to_path=symbol_to_path,  # 若你改成可注入 data_feed，这里就传 data_feed=...
        pc_model=pc_model,
        risk_model=risk_model,
        exec_model=exec_model,
        initial_cash=100_000.0,
        slippage=0.0005,
        commission_rate=0.0001,
        fixed_commission=0.0,
        keep_insights_active=True,
    )

    df = bt.run()

    # ✅ 交易统计（需要 engine logs + portfolio trade log）
    turn = compute_turnover(bt.fill_log, df["equity"])
    trade_summary = summarize_trades(bt.portfolio.trade_log)
    print("\n=== Turnover ===")
    for k, v in turn.items():
        print(f"{k:>20}: {v}")

    print("\n=== Trade Stats ===")
    for k, v in trade_summary.items():
        print(f"{k:>20}: {v}")


if __name__ == "__main__":
    main()
