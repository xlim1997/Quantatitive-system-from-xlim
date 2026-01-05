# run_live_ibkr.py
from __future__ import annotations

from core.engine import Engine
from core.live_runner import LiveRunner

from data.ibkr_feed import IBKRRealtimeBarFeed, IBKRConnConfig_Live as IBDataConn
from brokerage.ibkr_brokerage import IBKRBrokerage, IBKRConnConfig as IBBrokerConn

from portfolio.state import Portfolio
from portfolio.construction import WeightedByHintPC
from portfolio.execution import ImmediateExecutionModel
from portfolio.risk import ChainRiskManagementModel, StopLossRiskModel, MaxPositionWeightRiskModel, MaxGrossExposureRiskModel, PortfolioMaxDrawdownRiskModel
from portfolio.kill_switch import KillSwitchRiskModel, KillSwitchConfig

from strategies.CrossSectionalMomentumStrategy import CrossSectionalMomentumStrategy, MomentumConfig


def main():
    universe = ["AAPL", "MSFT", "AMZN", "GOOG", "META"]

    # Live data (5s -> 1min)
    data_feed = IBKRRealtimeBarFeed(
        symbols=universe,
        conn=IBDataConn(host="127.0.0.1", port=7497, client_id=11),
        agg_rule="1min",
        what_to_show="TRADES",
        use_rth=False,
    )

    # Live brokerage (paper recommended)
    brokerage = IBKRBrokerage(IBBrokerConn(host="127.0.0.1", port=7497, client_id=21))

    initial_cash = 100_000.0
    portfolio = Portfolio(cash=initial_cash)

    algo = CrossSectionalMomentumStrategy(
        MomentumConfig(
            universe=universe,
            lookback=60,
            rebalance_every=5,  # 5分钟调一次（因为我们聚合到 1min）
            top_k=2,
            bottom_k=0,
            weight_mode="equal",
            stop_loss_pct=0.10,
            cooldown_bars=10,
        )
    )

    pc = WeightedByHintPC(normalize_gross=True, gross_cap=1.0)
    exec_model = ImmediateExecutionModel(min_trade_value=50.0)

    risk = ChainRiskManagementModel([
        KillSwitchRiskModel(KillSwitchConfig(max_total_loss=3000.0), initial_cash=initial_cash),
        StopLossRiskModel(stop_loss_pct=0.10, take_profit_pct=None),
        PortfolioMaxDrawdownRiskModel(max_drawdown=0.25),
        MaxPositionWeightRiskModel(max_weight=0.30),
        MaxGrossExposureRiskModel(max_gross_exposure=1.0),
    ])

    engine = Engine(
        algorithm=algo,
        data_feed=data_feed,
        brokerage=brokerage,
        portfolio=portfolio,
        pc_model=pc,
        risk_model=risk,
        exec_model=exec_model,
        keep_insights_active=True,
    )
    engine.halt_trading = False

    runner = LiveRunner(engine, state_path="state/ibkr_live.pkl", save_every=1, log_path="logs/ibkr_live_equity.csv")
    runner.restore_if_exists()
    runner.run_forever()


if __name__ == "__main__":
    main()
