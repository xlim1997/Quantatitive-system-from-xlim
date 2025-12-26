# Quantatitive-system-from-xlim
Quantatitive trading system from scratch

# My Algo Engine 🇨🇳/🇺🇸
简化版 QuantConnect Lean 风格的事件驱动量化交易引擎（个人量化交易者友好）

A lightweight, Lean-style, event-driven algorithmic trading engine,  
designed for **individual quantitative traders** using **Python**.

---

## 1. 环境要求 / Requirements

- 操作系统 / OS
  - Linux / macOS / Windows 均可（推荐 Linux 或 WSL2）
- Python 版本 / Python Version
  - **Python 3.10+**（建议 3.10 或 3.11）
- 工具 / Tools
  - `git`
  - 包管理：`conda` 或 `python -m venv` + `pip`

---

## 2. 获取代码 / Clone the Repository

```bash
git@github.com:xlim1997/Quantatitive-system-from-xlim.git
cd Quantatitive-system-from-xlim

conda create -n my_algo_env python=3.10 -y
conda activate my_algo_env

pip install -r requirements

## 架构总览 / Framework Overview

本项目是一个 **简化版 QuantConnect Lean 风格** 的事件驱动量化交易引擎，核心思想：

> 一切都是事件（Events），  
> 策略只表达观点（Insights），  
> 组合/风控/执行负责把“观点”变成“订单”。

### 1. 核心模块 / Core Modules

从下到上，主要分为几层：

1. **核心事件系统（`core/events.py`）**  
   - 定义所有模块之间沟通的“通用语言”：  
     - `MarketDataEvent` : 行情事件（类似 Lean 的 `Slice` 中每个 symbol 的数据）  
     - `OrderEvent`      : 下单请求（Algorithm/Execution → Brokerage）  
     - `FillEvent`       : 成交回报（Brokerage → Portfolio）  
     - 预留类型：`BrokerStatusEvent`, `ErrorEvent`, `SCHEDULED` 等  
   - 好处：  
     - 所有模块只依赖统一的数据结构，耦合度低  
     - 以后要增加新的数据源/券商/风控/执行逻辑，只要遵守这套“语言”，就能无缝接入。

2. **策略与三模型（Algorithm & Portfolio / Risk / Execution）**  
   - `algorithm/`  
     - `BaseAlgorithm`：策略基类  
     - 策略只做一件事：**在 `on_data()` 里根据行情产生 `Insight` 列表**  
   - `portfolio/models.py`  
     - `Insight`：策略对单个标的的“观点”（看多/看空/中性 + 期望权重）  
     - `PortfolioTarget`：组合构建后的“目标权重”（例如 AAPL 20%，MSFT 10%）  
   - `portfolio/construction.py`  
     - `BasePortfolioConstructionModel`：把多个 `Insight` 转成一组合合理的 `PortfolioTarget`  
     - 示例：等权多头、按信号强度加权、目标波动率等  
   - `portfolio/risk.py`  
     - `BaseRiskManagementModel`：在风险约束下调整/过滤 `PortfolioTarget`  
     - 示例：限制最大单票权重、限制总杠杆等  
   - `portfolio/execution.py`  
     - `BaseExecutionModel`：负责把目标权重变成具体订单（`OrderEvent`）  
     - 示例：一次性市价下单、分批 TWAP/VWAP 下单

3. **组合状态（`portfolio/state.py`）**  
   - `Portfolio` / `Position`  
   - 职责单一：根据 `FillEvent` 更新现金和持仓，并提供当前净值/仓位快照。  
   - 不直接参与策略逻辑，也不做风控/执行，仅仅“记账”。

4. **数据与券商适配层（DataFeed & Brokerage）**  
   - `data/base.py` 定义统一接口，`data/local_csv.py` 是回测用的 CSV 数据源实现。  
   - `brokerage/base.py` 定义统一接口，`brokerage/paper.py` 是纸上回测撮合实现。  
   - 将来可以很容易扩展：  
     - `FutuDataFeed` / `IBDataFeed`  
     - `FutuBrokerage` / `IBKRBrokerage` / `BinanceBrokerage`

5. **引擎层（`core/engine.py` + `backtesting/engine.py`）**  
   - `Engine` 负责事件循环与模块编排：  
     1. 从 DataFeed 取出一批 `MarketDataEvent`  
     2. 调用 `Algorithm.on_data()` 得到 `Insights`  
     3. 交给 PortfolioConstructionModel → 得到 `PortfolioTargets`  
     4. 交给 RiskManagementModel → 得到风险调整后的目标  
     5. 交给 ExecutionModel → 生成一批 `OrderEvent`，发送给 Brokerage  
     6. 从 Brokerage 获取 `FillEvent`，更新 `Portfolio`  
   - `backtesting/engine.py` 封装了一个 `BacktestEngine`：  
     - 自动用 CSV 数据源 + 纸上撮合 + 组合模型，  
     - 方便你一行代码跑完整个回测。

### 2. 事件流示意 / Event Flow

下面是一张简化的事件流示意图，帮助理解各模块之间的数据流：

```text
[ DataFeed ] --MarketDataEvent--> [ Engine ] --传给--> [ Algorithm ]
                                                   |
                                                   v (Insights)
                                              [ PortfolioConstruction ]
                                                   |
                                                   v (PortfolioTargets)
                                              [ RiskManagement ]
                                                   |
                                                   v (Adjusted Targets)
                                              [ ExecutionModel ]
                                                   |
                                                   v (OrderEvent)
                                              [ Brokerage ]
                                                   |
                                                   v (FillEvent)
                                              [ Portfolio ]


### 3. Portfolio & Insights 模型说明

在这个框架中，**策略不直接控制“买多少股”**，而是遵循 QuantConnect Lean 式的三层结构：

1. 策略（Algorithm）输出 **Insights**  
2. 组合构建模型（PortfolioConstructionModel）把 Insights 转成 **目标权重（PortfolioTargets）**  
3. 风险模型（RiskModel）和执行模型（ExecutionModel）在此基础上做风险控制和下单细化  

对应的代码位置：

- `portfolio/models.py`
  - `InsightDirection`  
    - `UP` / `DOWN` / `FLAT`，对应看多 / 看空 / 中性  
  - `Insight`  
    - 策略对某个标的的“观点”：  
      - `symbol`：标的，例如 `"AAPL"`  
      - `direction`：`UP/DOWN/FLAT`  
      - `weight_hint`：希望的相对权重（例如 `+0.1` 表示想要 10% 多头）  
  - `PortfolioTarget`  
    - 组合构建后得到的目标持仓权重：  
      - `symbol`  
      - `target_percent`：目标权重（例如 `0.20` 表示 20% 多头）

- `portfolio/state.py`
  - `Position`  
    - 单个标的的持仓信息（数量 + 均价）  
  - `Portfolio`  
    - 整个组合状态（现金 + 所有持仓）  
    - 核心方法：  
      - `update_from_fill(fill)`：根据成交事件（`FillEvent`）更新现金和持仓  
      - `total_value(last_prices)`：根据最新价格计算当前组合净值  
      - `snapshot(last_prices)`：返回一个适合写日志/调试的组合快照

通过这种设计：

- **Algorithm** 只负责“生成观点（Insights）”  
- **PortfolioConstruction + Risk + Execution** 负责把观点变成可执行订单  
- **Portfolio** 负责“记账和估值”，不参与决策

这使得你可以在不改策略代码的情况下：

- 换一套组合构建逻辑（等权、多因子打分、风险平价等）  
- 换一套风险模型（更激进或更保守）  
- 换一套执行模型（一次性市价成交 vs 分批 TWAP）  

非常适合做系统化回测、策略对比实验和风控研究。


### 4. 策略 & 数据层说明（Algorithm & DataFeed）

#### 4.1 策略基类：`algorithm/base.py`

策略不直接“下单”，而是继承 `BaseAlgorithm`，通过 `on_data()` 返回一组 **Insights**：

- `BaseAlgorithm.initialize()`  
  在引擎开始运行前调用，用来：
  - 选择标的（`self.add_equity("AAPL")`）
  - 设置参数（窗口长度、因子权重等）
  - 初始化内部状态（价格缓存、指标等）

- `BaseAlgorithm.on_data(data)`  
  每个时间步由引擎调用，其中：
  - `data` 是一个字典：`{symbol: MarketDataEvent, ...}`  
  - 策略根据这些行情数据，返回一个 `List[Insight]`，例如：

    ```python
    [
      Insight(symbol="AAPL", direction=UP,   weight_hint=0.2),
      Insight(symbol="MSFT", direction=FLAT, weight_hint=0.0),
    ]
    ```

这些 Insights 会被后续的：

- `PortfolioConstructionModel` 转成目标权重（PortfolioTargets）
- `RiskManagementModel` 做风险过滤
- `ExecutionModel` 转成真实订单（OrderEvent）

#### 4.2 数据源接口：`data/base.py`

数据源（DataFeed）的职责是“按时间顺序提供行情切片”：

- 所有数据源都继承 `BaseDataFeed`，必须实现：
  - `__iter__(self) -> Iterator[Dict[str, MarketDataEvent]]`
    - 每次迭代返回某个时间点的多标的行情：
      `{symbol: MarketDataEvent, ...}`
  - `last_market_data` 属性：
    - 返回最近一次产生的行情切片，供撮合和组合估值使用。

通过这个接口，可以很方便地切换不同的数据源实现：

- 回测：`LocalCSVDataFeed`（从本地 CSV 读历史数据）
- 实盘：`LiveAPIDataFeed`（从 Futu / IBKR / Binance 等拉实时数据）

#### 4.3 本地 CSV 数据源：`data/local_csv.py`

`LocalCSVDataFeed` 是一个用于回测的简单数据源实现：

- 初始化时传入 `symbol_to_path` 字典：
  ```python
  symbol_to_path = {
      "AAPL": "data/aapl_daily.csv",
      "MSFT": "data/msft_daily.csv",
  }

### 5. Brokerage（券商/撮合）层说明

Brokerage 层负责把订单真正“执行”出来：

- 回测时：用 `PaperBrokerage` 做撮合（模拟成交）
- 实盘时：用 `IBKRBrokerage` / `FutuBrokerage` 连接真实券商 API

#### 5.1 统一接口：`brokerage/base.py`

所有 Brokerage 必须实现：

- `place_order(order: OrderEvent)`  
  接收订单请求  
- `get_fills() -> List[FillEvent]`  
  返回新产生的成交回报（FillEvent）

引擎会在每个时间步调用 `get_fills()`，并把成交传给 `Portfolio.update_from_fill()` 更新账户状态。

#### 5.2 回测撮合：`brokerage/paper.py`

`PaperBrokerage` 是最小可用的回测撮合器：

- 当前只支持 `MARKET` 市价单
- 立即按当前 `close` 成交
- 支持：
  - `slippage`（滑点，买更贵卖更便宜）
  - `commission_rate`（按成交额收取比例手续费）
  - `fixed_commission`（每笔固定手续费）

后续可以扩展更真实的撮合机制（限价单、部分成交、bid/ask spread 等）。


### 6. 三模型解耦（Portfolio / Risk / Execution）

本框架采用 Lean 风格的三模型解耦：

1) **PortfolioConstructionModel**
- 输入：策略产生的 `Insights`
- 输出：组合目标 `PortfolioTargets`（目标权重）

代码位置：`portfolio/construction.py`  
- `EqualWeightLongOnlyPC`：只做多、等权分配  
- `WeightedByHintPC`：按 `weight_hint` 加权（支持多空），并可做总曝险归一化

2) **RiskManagementModel**
- 输入：初始 `PortfolioTargets`
- 输出：风险调整后的 Targets

代码位置：`portfolio/risk.py`  
- `NoRiskModel`：不做风险调整  
- `MaxGrossExposureRiskModel`：限制总绝对权重（gross exposure）  
- `MaxPositionWeightRiskModel`：限制单票最大权重

3) **ExecutionModel**
- 输入：风险调整后的 Targets + 当前组合状态 + 最新价格
- 输出：订单 `OrderEvent` 并发送给 Brokerage

代码位置：`portfolio/execution.py`  
- `ImmediateExecutionModel`：一次性调仓到目标权重（市价单）

这种设计的好处：
- 策略只负责产生观点（Insights），不必关心交易细节  
- 风控与执行逻辑可独立替换，方便做消融实验和策略对比  
- 回测到实盘迁移时，只需替换数据源和券商适配层，核心逻辑复用


### 7. Engine（主循环调度器）

`core/engine.py` 定义了主引擎 `Engine`，它负责把所有模块串起来：

每个时间步（一个 bar）：
1. DataFeed 产生 `MarketDataEvent` 切片  
2. Algorithm 生成 `Insights`  
3. PortfolioConstructionModel：Insights → Targets  
4. RiskManagementModel：Targets → Adjusted Targets  
5. ExecutionModel：Adjusted Targets → Orders（OrderEvent）  
6. Brokerage 返回 `FillEvent`  
7. Portfolio 根据 `FillEvent` 更新现金与持仓  
8. Engine 记录净值、仓位快照用于回测分析

#### Active Insights（轻量版 InsightManager）

为了避免策略每个 bar 重复输出相同信号，引擎提供 active insight 缓存：

- 策略可以只在信号发生变化时输出 Insight  
- Engine 会缓存 active insights，直到：
  - 被 `FLAT` 覆盖（撤销观点）
  - 或到期（expiry，可选）

这样更接近 Lean 的 Insight 管理机制。


## Strategies（策略库）

### Momentum（横截面动量选股）
路径：`strategies/momentum/main.py`

- 维护每个标的 close 历史
- 动量 = close[t] / close[t-lookback] - 1
- 每 `rebalance_every` 根 bar：
  - 选 Top-K 做多
  - 可选 Bottom-K 做空
  - 对被踢出的旧标的显式输出 `FLAT`，确保 Engine 的 `active_insights` 不会遗留旧仓位

运行：
```bash
python run_momentum_backtest.py


