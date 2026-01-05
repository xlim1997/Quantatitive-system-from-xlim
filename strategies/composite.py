# algorithm/composite.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional

from algorithm.base import BaseAlgorithm
from core.events import MarketDataEvent
from portfolio.models import Insight


@dataclass
class StrategySpec:
    algo: BaseAlgorithm
    weight: float = 1.0   # 给每个策略一个权重（默认 1）


class CompositeAlgorithm(BaseAlgorithm):
    """
    把多个 BaseAlgorithm 组合成一个 Algorithm。
    Engine 依旧只认识一个 algo，但内部可以运行 N 个策略。
    """

    def __init__(self, strategies: Sequence[StrategySpec], flat_threshold: float = 1e-6) -> None:
        super().__init__()
        self._strategies = list(strategies)
        self._flat_threshold = flat_threshold

    def set_engine(self, engine) -> None:
        super().set_engine(engine)
        # 把 engine 注入给子策略
        for spec in self._strategies:
            spec.algo.set_engine(engine)

    def initialize(self) -> None:
        # 让每个子策略初始化并收集 symbols（Engine 通常用 algo.symbols 来订阅/拉数据）
        all_syms: set[str] = set()
        for spec in self._strategies:
            spec.algo.initialize()
            all_syms.update(spec.algo.symbols)
        self.symbols = sorted(all_syms)

    def on_data(self, data: Dict[str, MarketDataEvent]) -> List[Insight]:
        raw: List[Insight] = []
        for spec in self._strategies:
            ins = spec.algo.on_data(data)
            # 将策略权重乘到 weight_hint 上（假设 weight_hint 是可线性叠加的“目标权重提示”）
            for i in ins:
                raw.append(self._scale_insight(i, spec.weight))

        return self._merge_insights(raw)

    def _scale_insight(self, ins: Insight, w: float) -> Insight:
        # 兼容旧 Insight：只要求有 symbol/direction/weight_hint
        return Insight(
            symbol=ins.symbol,
            direction=ins.direction,
            weight_hint=float(ins.weight_hint) * float(w),
            # 下面字段如果你还没加到 Insight，就先删掉；我在第 2 节给你一个“兼容增强版”
            source=getattr(ins, "source", "") or getattr(ins, "meta", {}).get("source", ""),
            confidence=float(getattr(ins, "confidence", 1.0)),
            horizon_bars=int(getattr(ins, "horizon_bars", 1)),
            meta=dict(getattr(ins, "meta", {})),
        )

    def _merge_insights(self, insights: List[Insight]) -> List[Insight]:
        """
        合并规则（先用“最稳的默认版本”）：
        - 同一 symbol：weight_hint 直接相加（策略投票/叠加）
        - direction 用合并后 weight_hint 的符号决定
        - 很小则 FLAT
        """
        by_symbol: Dict[str, List[Insight]] = {}
        for i in insights:
            by_symbol.setdefault(i.symbol, []).append(i)

        merged: List[Insight] = []
        for sym, lst in by_symbol.items():
            total = sum(float(i.weight_hint) for i in lst)
            if abs(total) < self._flat_threshold:
                direction = 0
                total = 0.0
            else:
                direction = 1 if total > 0 else -1

            # 合并一些元信息（可选）
            conf = 0.0
            for i in lst:
                conf += float(getattr(i, "confidence", 1.0))
            conf = min(conf / max(len(lst), 1), 1.0)

            meta = {"sources": [getattr(i, "source", "") for i in lst if getattr(i, "source", "")]}

            merged.append(Insight(
                symbol=sym,
                direction=direction,
                weight_hint=total,
                source="composite",
                confidence=conf,
                horizon_bars=max(int(getattr(i, "horizon_bars", 1)) for i in lst),
                meta=meta,
            ))
        return merged
