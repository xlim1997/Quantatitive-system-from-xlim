# portfolio/models.py
"""
组合相关的核心“消息结构”
========================

本文件定义两种在 Portfolio / Risk / Execution 三模型之间传递的核心结构：

1. Insight
   - 策略（Algorithm）对单个标的的“观点”
   - 比如：AAPL 看多，建议权重 +0.1（想占组合 10%）

2. PortfolioTarget
   - 组合构建模型（PortfolioConstructionModel）输出的“目标仓位”
   - 比如：AAPL 目标持仓 20% 权重，MSFT 10%

这样，Algorithm 不直接决定“买多少股”，而是先表达“想持有多少权重”，
由组合/风控/执行三个模型一起把它变成具体订单。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# 1. InsightDirection：策略观点的方向（看多 / 看空 / 中性）
# ---------------------------------------------------------------------------

class InsightDirection(Enum):
    """
    策略对某个标的的方向性观点：

    - UP   : 看多（希望持有多头）
    - DOWN : 看空（希望持有空头或降低多头）
    - FLAT : 中性（希望不持仓 / 清仓）

    注意：
    - UP / DOWN 只是“方向”，实际持仓大小由 weight_hint 决定。
    """
    UP = auto()
    DOWN = auto()
    FLAT = auto()


# ---------------------------------------------------------------------------
# 2. Insight：策略对某个标的的“观点”
# ---------------------------------------------------------------------------

@dataclass
class Insight:
    """
    策略输出的“观点”（类似 Lean 的 Insight 类）：

    字段：
    - symbol      : 标的代码，如 "AAPL"
    - direction   : InsightDirection.UP / DOWN / FLAT
    - weight_hint : 对该标的的“希望权重”，范围通常在 [-1, 1]
                    例如：
                      +0.10 -> 想持有 10% 净值的多头（长仓）
                      -0.05 -> 想持有 5% 净值的空头（短仓或对冲）
                      0.0   -> 想不持有（或平仓）

    - expiry      : 观点的“过期时间”（可选），例如：
                    - 回测时：pandas.Timestamp
                    - 实盘时：datetime.datetime
                    当前版本可以先不使用，后续做“信号失效”时会用到。

    使用方式：
    - Algorithm.on_data(...) 返回 List[Insight]
    - PortfolioConstructionModel 根据这些 Insight 设计组合。
    """
    symbol: str
    direction: InsightDirection
    weight_hint: float
    expiry: Optional[object] = None  # 现在可以不管，后续高级用法


# ---------------------------------------------------------------------------
# 3. PortfolioTarget：组合构建后得到的目标权重
# ---------------------------------------------------------------------------

@dataclass
class PortfolioTarget:
    """
    组合构建模型输出的“目标头寸”（类似 Lean 的 PortfolioTarget）：

    字段：
    - symbol         : 标的代码
    - target_percent : 目标持仓权重（相对于组合净值），通常在 [-1, 1]

    示例：
    - AAPL,  +0.20 -> 希望 AAPL 占组合净值的 20%（多头）
    - SPY,   -0.10 -> 希望 SPY 占组合净值的 -10%（空头对冲）
    - CASH,   0.00 -> 表示不持仓

    说明：
    - 组合构建（PortfolioConstructionModel）负责从多个 Insight 产生一组
      PortfolioTarget（加起来可能 <= 1, 或者按杠杆要求放大/缩小）
    - RiskManagementModel 可以对这批目标进一步缩放/过滤，保证满足风险约束。
    """
    symbol: str
    target_percent: float
