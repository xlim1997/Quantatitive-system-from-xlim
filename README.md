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


