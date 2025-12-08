# core/events.py
"""
    事件系统（Event System）
    =======================

    这个文件定义了整个交易引擎中各个模块之间沟通使用的“通用语言”：

    - 行情事件（MarketDataEvent）
    - 下单事件（OrderEvent）
    - 成交事件（FillEvent）
    - 组合更新、定时任务、Broker 状态、错误等事件类型（预留）

    设计思路：
    - 所有事件都继承自基础类 Event，至少包含：
    - type: EventType     -> 事件类型
    - timestamp: Any      -> 事件发生的时间（datetime 或 int）
    - 具体的事件类型再额外增加自己需要的字段
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional

class EventType(Enum):
    
    """
        EventType 用来区分不同类型的事件。

        目前用到的核心类型：
        - MARKET      : 行情数据（K 线、Tick 等）
        - ORDER       : 策略/执行模型发出的下单请求
        - FILL        : 券商/撮合返回的成交信息
        - PORTFOLIO   : 组合状态更新（暂时可选，用于结果展示）
        - SCHEDULED   : 定时任务触发（例如每日收盘后回调）
        - BROKER_STATUS: 券商连接状态变化（如断线/重连）
        - ERROR       : 错误/告警事件（方便监控）

        好处：
        - 引擎或日志系统可以根据 type 做不同处理
        - 以后要扩展新的事件类型，只需在这里添加
    """

    MARKET = auto()
    ORDER = auto()
    FILL = auto()
    PORTFOLIO = auto()
    SCHEDULED = auto()
    BROKER_STATUS = auto()
    ERROR = auto()
    

# ---------------------------------------------------------------------------
# 2. 基础事件类：所有事件的共同父类
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """
    所有事件的基类。

    字段：
    - type      : EventType，事件类型（见上）
    - timestamp : 事件发生时间，可以是 datetime、int（bar 序号）等。

    说明：
    - 我们不强制 timestamp 一定是 datetime，以便：
      - 回测时可以用 pandas.Timestamp
      - 实盘时用 datetime.datetime
      - 也可以用简单的整数 index
    """
    type: EventType
    timestamp: Any

@dataclass
class MarketEvent(Event):
    symbol: str
    data: Dict[str, Any]  # K线或Tick数据

    def __init__(self, timestamp: Any, symbol: str, data: Dict[str, Any]):
        super().__init__(EventType.MARKET, timestamp)
        self.symbol = symbol
        self.data = data


# ---------------------------------------------------------------------------
# 3. 行情领域对象（Domain Model）：Bar / Tick 等
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    """
    标准 K 线（OHLCV）结构。

    字段：
    - open   : 开盘价
    - high   : 最高价
    - low    : 最低价
    - close  : 收盘价
    - volume : 成交量（没有可以用 0.0）

    说明：
    - 这是一个“领域模型（domain model）”，不包含时间戳和 symbol；
      时间戳和 symbol 信息由 MarketDataEvent 来承载。
    """
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


# 如果你以后需要 Tick / OrderBook，可以在这里扩展，例如：
#
# @dataclass
# class Tick:
#     last_price: float
#     last_size: float
#     bid: float
#     ask: float
#     bid_size: float
#     ask_size: float


# ---------------------------------------------------------------------------
# 4. 行情事件：MarketDataEvent（类似 Lean 的 Slice 中的每个 symbol）
# ---------------------------------------------------------------------------

@dataclass
class MarketDataEvent(Event):
    """
    行情事件：表示“某个标的在某一时刻的一条行情数据”。

    字段：
    - type      : 必须是 EventType.MARKET
    - timestamp : 时间（由父类 Event 提供）
    - symbol    : 证券代码，如 "AAPL"、"ESM4"
    - bar       : Bar 对象（OHLCV）
    - extra     : 额外信息（可选，如分笔数据、因子值等）

    说明：
    - 在引擎内部，我们通常用 Dict[str, MarketDataEvent] 来表示某个时刻
      多个 symbol 的行情切片，这个结构就类似 Lean 里的 Slice。
    """
    symbol: str
    bar: Bar
    extra: Dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# 5. 下单事件：OrderEvent（策略/执行模型 -> 券商/撮合）
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    """买卖方向：BUY / SELL。"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """订单类型：这里只定义几种常见的，方便后续扩展。"""
    MARKET = "market"   # 市价单
    LIMIT = "limit"     # 限价单
    STOP = "stop"       # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单


class TimeInForce(str, Enum):
    """订单有效期：可以先用最常用的几种。"""
    DAY = "day"         # 当日有效
    GTC = "gtc"         # Good-Till-Cancelled，直到取消
    IOC = "ioc"         # Immediate-Or-Cancel
    FOK = "fok"         # Fill-Or-Kill


@dataclass
class OrderEvent(Event):
    """
    下单事件：由 ExecutionModel 或策略发出，交给 Brokerage 执行。

    字段：
    - type        : EventType.ORDER
    - timestamp   : 触发下单的时间（下单时刻）
    - symbol      : 交易标的，如 "AAPL"
    - quantity    : 数量（>0 表示买，<0 表示卖）
    - side        : OrderSide.BUY / SELL（冗余但更清晰）
    - order_type  : OrderType.MARKET / LIMIT / STOP / STOP_LIMIT
    - limit_price : 限价单/止损限价单的价格（可选）
    - stop_price  : 止损单/止损限价单的触发价（可选）
    - time_in_force: 有效期（如 DAY / GTC）
    - tag         : 备注（例如 "enter long", "risk-reduction"）

    说明：
    - 对于单纯的回测，你可以只填：
      - order_type = MARKET
      - quantity / side
    - limit_price / stop_price 可以在以后需要时再真正使用。
    """
    symbol: str
    quantity: float
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    tag: Optional[str] = None


# ---------------------------------------------------------------------------
# 6. 成交事件：FillEvent（Brokerage -> Portfolio）
# ---------------------------------------------------------------------------

@dataclass
class FillEvent(Event):
    """
    成交事件：由 Brokerage 在订单成交/部分成交时产生。

    字段：
    - type        : EventType.FILL
    - timestamp   : 成交时间（撮合时间）
    - order_id    : 关联的订单 ID（由 Brokerage 生成）
    - symbol      : 标的
    - quantity    : 成交数量（>0 买入，<0 卖出）
    - price       : 成交价格
    - commission  : 手续费（可为 0）
    - slippage    : 滑点（可选，用于分析）
    - is_partial  : 是否部分成交（True = 部分成交）

    说明：
    - Portfolio 只关心 quantity / price / commission，用来更新现金和持仓。
    - 你也可以在回测时用 slippage 字段记录“理论价 vs 成交价”的差异。
    """
    order_id: str
    symbol: str
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    is_partial: bool = False


# ---------------------------------------------------------------------------
# 7. Broker 状态 & 错误事件（可选/预留）
# ---------------------------------------------------------------------------

class BrokerStatus(str, Enum):
    """Broker / 交易连接状态。"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"


@dataclass
class BrokerStatusEvent(Event):
    """
    券商状态事件，例如：
    - 断线
    - 重连成功
    - API 限流等

    用途：
    - 可以在监控模块/日志里记录
    - 也可以触发策略减少下单、强制平仓等安全动作（高级用法）
    """
    status: BrokerStatus
    message: Optional[str] = None


@dataclass
class ErrorEvent(Event):
    """
    错误/告警事件。

    字段：
    - severity : 严重程度（"warn" / "error" 等）
    - message  : 错误信息
    - details  : 附加信息（stack trace、上下文等）

    用途：
    - 可以统一由 Engine 记录到日志 / 发通知（邮件/Telegram）
    """
    severity: str
    message: str
    details: Optional[Dict[str, Any]] = None