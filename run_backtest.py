# run_backtest.py
from __future__ import annotations

from backtesting.engine import BacktestEngine
from strategies.sample_buy_and_hold.main import BuyAndHoldOnceStrategy

from portfolio.construction import WeightedByHintPC
from portfolio.risk import MaxGrossExposureRiskModel, MaxPositionWeightRiskModel
from portfolio.execution import ImmediateExecutionModel
from data.futu_feed import FutuHistoryKlineDataFeed
from futu import KLType, AuType
from data.ibkr_feed import IBKRHistoryBarDataFeed, IBKRConnConfig, IBKRContractSpec

def main():
    # 1) 选择策略
    algo = BuyAndHoldOnceStrategy(symbol="AAPL")

    # 2) 数据路径（你需要准备对应 CSV 文件）
    symbol_to_path = {
        "AAPL": "data/aapl_daily.csv",
    }
    
    data_feed = IBKRHistoryBarDataFeed(
        contracts={"AAPL": IBKRContractSpec("AAPL")},
        duration_str="1 Y",
        bar_size="1 day",
        conn=IBKRConnConfig(port=4002, client_id=7),  # TWS paper 默认 7497 
    )

    # 3) 三模型
    pc_model = WeightedByHintPC(normalize_gross=True, gross_cap=1.0)   # 用 weight_hint 构建目标
    risk_model = MaxGrossExposureRiskModel(max_gross_exposure=1.0)     # 不加杠杆
    exec_model = ImmediateExecutionModel(min_trade_value=50.0)         # 忽略太小的调整

    # 4) 回测引擎
    bt = BacktestEngine(
        algorithm=algo,
        symbol_to_path=symbol_to_path,
        data_feed=data_feed,
        pc_model=pc_model,
        risk_model=risk_model,
        exec_model=exec_model,
        initial_cash=100_000.0,
        slippage=0.0005,            # 5 bps
        commission_rate=0.0001,     # 1 bps
        fixed_commission=0.0,
        keep_insights_active=True,  # 启用 active insight 缓存（推荐）
    )

    df = bt.run_to_dataframe()
    print(df.tail(5))

    # 你后续可以自己画 equity 曲线：
    # import matplotlib.pyplot as plt
    # plt.plot(df["timestamp"], df["equity"])
    # plt.show()


if __name__ == "__main__":
    main()
