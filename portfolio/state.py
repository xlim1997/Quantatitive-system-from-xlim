# portfolio/state.py
"""
组合状态（Portfolio State）
==========================

本文件负责描述并维护“当前账户状态”：

- Position：单个标的的持仓信息（数量 + 均价）
- Portfolio：整体组合（现金 + 所有持仓）

职责非常单一：
- 根据成交事件（FillEvent）更新现金和持仓
- 根据最新价格计算组合总资产（equity）

不做的事情：
- 不负责决定买什么 / 买多少（那是组合 & 风控 & 执行模型的工作）
- 不负责下单（那是 ExecutionModel + Brokerage 的工作）
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from core.events import FillEvent


@dataclass
class Position:
    """
    单个标的的持仓信息：

    字段：
    - symbol    : 标的代码，如 "AAPL"
    - quantity  : 当前持仓数量（>0 表示多头，<0 表示空头）
    - avg_price : 当前持仓的持仓成本均价

    示例：
    - quantity = 100, avg_price = 150 -> 持有 100 股，成本价 150
    - quantity = -50, avg_price = 120 -> 空头 50 股，建仓均价 120
    """
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    
# ---------------------------------------------------------------------------
# 2. Portfolio：整个账户的状态（现金 + 持仓）
# ---------------------------------------------------------------------------
@dataclass
class Portfolio:
    """
    组合状态：

    字段：
    - cash      : 当前现金余额
    - positions : 每个标的的持仓（symbol -> Position）

    关键方法：
    - update_from_fill(fill) : 根据成交更新仓位和现金
    - total_value(prices)    : 根据最新价格计算总资产净值
    - snapshot(prices)       : 返回一个适合 log/打印的快照 dict
    """
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    # -----------------------------
    # 根据成交事件更新组合
    # -----------------------------
    def update_from_fill(self, fill: FillEvent) -> None:
        """
        根据一条成交记录（FillEvent）更新组合状态：

        - 更新对应 symbol 的 Position（数量 + 均价）
        - 更新现金余额（买入扣现金，卖出加现金）

        注意：
        - fill.quantity > 0  表示买入
        - fill.quantity < 0  表示卖出
        """
        pos = self.positions.get(fill.symbol, Position(symbol=fill.symbol))
        old_qty = pos.quantity
        old_avg = pos.avg_price
        trade_qty = fill.quantity      # >0 买入，<0 卖出
        price = fill.price
        
        # 本次交易的“总金额”（不含手续费）：price * quantity
        trade_cash = fill.price * fill.quantity
        # 更新现金：买入扣钱，卖出加钱
        self.cash -= trade_cash + fill.commission
        # 1) 原来没仓位：直接开新仓
        if old_qty == 0:
            pos.quantity = trade_qty
            pos.avg_price = price

         # 2) 原来有仓位，且本次成交方向相同（加仓）
        elif old_qty * trade_qty > 0:
            new_qty = old_qty + trade_qty
            total_cost = old_avg * old_qty + price * trade_qty
            pos.quantity = new_qty
            pos.avg_price = total_cost / new_qty
        else:
            # 3) 本次成交与原方向相反：先平仓再看是否翻方向
            new_qty = old_qty + trade_qty

            # 3a) 只是部分平仓（方向不变）
            if old_qty * new_qty > 0:
                # 这里可以选择保持 avg_price 不变（常见做法）
                pos.quantity = new_qty
                # pos.avg_price 不变

            # 3b) 刚好全部平仓
            elif new_qty == 0:
                pos.quantity = 0
                pos.avg_price = 0.0

            # 3c) 超额平仓 -> 翻方向，建立新仓
            else:
                # 例：old_qty=+100, trade_qty=-150, new_qty=-50
                # 说明：先把 100 股多头全部平掉，再额外开 -50 股空头
                pos.quantity = new_qty
                pos.avg_price = price   # 新方向的仓位成本按当前成交价算

        # 更新回 positions 字典
        self.positions[fill.symbol] = pos
        


    # -----------------------------
    # 计算当前组合总资产
    # -----------------------------
    def total_value(self, last_prices: dict[str, float]) -> float:
        """
        根据最新价格（last_prices）计算组合净值。

        参数：
        - last_prices: {symbol: last_price}

        返回：
        - equity: 当前总资产（现金 + 所有持仓按市价估值）

        注意：
        - 如果某个 symbol 在 last_prices 中找不到，就退而求其次用 avg_price。
        """
        equity = self.cash
        for sym, pos in self.positions.items():
            price = last_prices.get(sym, pos.avg_price)
            equity += pos.quantity * price
        return equity

    # -----------------------------
    # 返回一个用于日志/调试的快照
    # -----------------------------
    def snapshot(self, last_prices: dict[str, float]) -> dict:
        """
        返回一个字典格式的组合快照，便于打印或写入日志/结果文件。

        返回示例：
        {
            "cash": 95000.0,
            "equity": 102000.0,
            "positions": {
                "AAPL": {"qty": 100, "avg_price": 150.0},
                "MSFT": {"qty": 50,  "avg_price": 300.0}
            }
        }
        """
        return {
            "cash": self.cash,
            "equity": self.total_value(last_prices),
            "positions": {
                sym: {"qty": p.quantity, "avg_price": p.avg_price}
                for sym, p in self.positions.items()
            },
        }