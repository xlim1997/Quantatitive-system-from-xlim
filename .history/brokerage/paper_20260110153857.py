from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from core.events import OrderEvent, FillEvent, EventType, OrderType, OrderSide
from .base import BaseBrokerage


@dataclass(frozen=True)
class CommissionSpec:
    """
    手续费规格（尽量统一口径）
    """
    model: str = "value_pct"              # value_pct | per_share
    # value_pct
    commission_rate: float = 0.0          # 按成交额比例
    # per_share
    per_share: float = 0.0                # 每股佣金
    min_commission: float = 0.0           # 每单最低
    max_commission_pct: Optional[float] = None  # 每单封顶（成交额比例），例如 0.01
    # extra
    fixed_commission: float = 0.0         # 每单固定附加费（可选）


IBKR_PRESETS: Dict[str, CommissionSpec] = {
    # IBKR Pro Fixed (US Stocks/ETFs 常见结构)
    "ibkr_pro_fixed": CommissionSpec(
        model="per_share",
        per_share=0.005,
        min_commission=1.0,
        max_commission_pct=0.01,
        fixed_commission=0.0,
    ),
    # IBKR Pro Tiered（最低档示例；更高月成交量会更低）
    "ibkr_pro_tiered": CommissionSpec(
        model="per_share",
        per_share=0.0035,
        min_commission=0.35,
        max_commission_pct=0.01,
        fixed_commission=0.0,
    ),
    # IBKR Lite（近似 0 佣金）
    "ibkr_lite": CommissionSpec(
        model="value_pct",
        commission_rate=0.0,
        fixed_commission=0.0,
    ),
}


def _normalize_model_name(name: str) -> str:
    """
    允许你用更短/更直觉的名字：
      - ibkr_fixed -> ibkr_pro_fixed
      - ibkr_tiered -> ibkr_pro_tiered
      - ibkr_pro -> 默认映射为 ibkr_pro_tiered（你可按喜好改成 fixed）
    """
    key = (name or "").strip().lower()
    alias = {
        "ibkr_fixed": "ibkr_pro_fixed",
        "ibkr_tiered": "ibkr_pro_tiered",
        "ibkr_pro": "ibkr_pro_tiered",   # ✅ 这里你想让 ibkr_pro=Fixed 就改成 ibkr_pro_fixed
        "ibkr_lite": "ibkr_lite",
    }
    return alias.get(key, key)


class PaperBrokerage(BaseBrokerage):
    def __init__(
        self,
        slippage: float = 0.0,
        commission_model: str = "value_pct",
        # 兼容你原来的参数（value_pct）
        commission_rate: float = 0.0,
        fixed_commission: float = 0.0,
        # 通用 per_share 参数（当 commission_model="per_share" 时使用）
        per_share: float = 0.0,
        min_commission: float = 0.0,
        max_commission_pct: Optional[float] = None,
        enforce_cash: bool = True,  # 是否强制现金约束
        # 可选：按 symbol 覆盖手续费模型
        commission_by_symbol: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        
        self.enforce_cash = bool(enforce_cash)
        
        self.slippage = float(slippage)

        self._commission_by_symbol = commission_by_symbol or {}

        # 默认 spec（来自 commission_model + 你传入的参数）
        cm = _normalize_model_name(commission_model)
        if cm in IBKR_PRESETS:
            self._default_spec = IBKR_PRESETS[cm]
        else:
            # 自定义 spec
            if cm not in {"value_pct", "per_share"}:
                raise ValueError(
                    f"Unknown commission_model='{commission_model}'. "
                    f"Use one of: value_pct, per_share, ibkr_fixed, ibkr_pro, ibkr_tiered, ibkr_lite."
                )
            self._default_spec = CommissionSpec(
                model=cm,
                commission_rate=float(commission_rate),
                per_share=float(per_share),
                min_commission=float(min_commission),
                max_commission_pct=max_commission_pct,
                fixed_commission=float(fixed_commission),
            )

        self._pending_fills: List[FillEvent] = []
        self._order_counter = 0

    def _spec_for_symbol(self, symbol: str) -> CommissionSpec:
        name = self._commission_by_symbol.get(symbol)
        if not name:
            return self._default_spec
        key = _normalize_model_name(name)
        if key not in IBKR_PRESETS:
            raise ValueError(f"commission_by_symbol[{symbol}]='{name}' not in IBKR presets.")
        return IBKR_PRESETS[key]

    def _calc_commission(self, spec: CommissionSpec, signed_qty: float, price: float) -> float:
        qty = abs(float(signed_qty))
        trade_value = qty * abs(float(price))

        if spec.model == "value_pct":
            return trade_value * spec.commission_rate + spec.fixed_commission

        if spec.model == "per_share":
            commission = qty * spec.per_share + spec.fixed_commission
            if spec.min_commission > 0:
                commission = max(commission, spec.min_commission)
            if spec.max_commission_pct is not None:
                commission = min(commission, trade_value * float(spec.max_commission_pct))
            return commission

        raise ValueError(f"Unsupported spec.model={spec.model}")
    
    def _max_affordable_buy_qty(self, budget: float, price: float, desired_qty: int) -> int:
        if desired_qty <= 0 or budget <= 0 or price <= 0:
            return 0

        lo, hi = 0, desired_qty
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cost = mid * price + self._commission(mid, price)
            if cost <= budget + 1e-9:
                lo = mid
            else:
                hi = mid - 1
        return lo
    def place_order(self, order: OrderEvent) -> None:
        if self._engine is None:
            raise RuntimeError("PaperBrokerage has no engine attached.")
        if order.order_type != OrderType.MARKET:
            raise NotImplementedError("PaperBrokerage only supports MARKET orders.")

        md = self._engine.data_feed.last_market_data.get(order.symbol)
        if md is None:
            raise ValueError(f"No market data for symbol={order.symbol} at time={self._engine.current_time}.")

        close = md.bar.close

        # slippage
        if order.side == OrderSide.BUY:
            fill_price = close * (1.0 + self.slippage)
        else:
            fill_price = close * (1.0 - self.slippage)

        # ✅ signed qty：买 +，卖 -
        base_qty = abs(float(order.quantity))
        signed_qty = base_qty if order.side == OrderSide.BUY else -base_qty

        # ✅ enforce cash：BUY 不能把 cash 买成负数
        # self._reset_shadow_if_needed()
        # if self.enforce_cash and order.side == OrderSide.BUY:
        #     max_qty = self._max_affordable_buy_qty(self._shadow_cash, fill_price, base_qty)
        #     if max_qty <= 0:
        #         return  # 买不起：直接不成交（也可以改成打 log）
        #     signed_qty = max_qty  # BUY 正数
        
        spec = self._spec_for_symbol(order.symbol)
        commission = self._calc_commission(spec, signed_qty=signed_qty, price=fill_price)

        self._order_counter += 1
        order_id = f"paper-{self._order_counter}"

        fill = FillEvent(
            type=EventType.FILL,
            timestamp=self._engine.current_time,
            order_id=order_id,
            symbol=order.symbol,
            quantity=signed_qty,
            price=fill_price,
            commission=commission,
            slippage=self.slippage,
            is_partial=False,
        )
        self._pending_fills.append(fill)

    def get_fills(self) -> List[FillEvent]:
        fills = self._pending_fills
        self._pending_fills = []
        return fills
