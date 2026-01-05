# core/state_store.py
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EngineState:
    engine_time: Any
    active_insights: Dict[str, Any]
    halt_trading: bool


@dataclass
class SnapshotState:
    portfolio: Any
    engine_state: EngineState
    algo_state: Optional[dict]


class StateStore:
    def __init__(self, path: str = "state/last.pkl") -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def save(self, engine) -> None:
        algo_state = None
        if hasattr(engine.algorithm, "get_state"):
            try:
                algo_state = engine.algorithm.get_state()
            except Exception:
                algo_state = None

        s = SnapshotState(
            portfolio=engine.portfolio,
            engine_state=EngineState(
                engine_time=engine.current_time,
                active_insights=getattr(engine, "_active_insights", {}),
                halt_trading=getattr(engine, "halt_trading", False) or getattr(engine.portfolio, "halt_trading", False),
            ),
            algo_state=algo_state,
        )

        with open(self.path, "wb") as f:
            pickle.dump(s, f)

    def load(self) -> Optional[SnapshotState]:
        if not os.path.exists(self.path):
            return None
        with open(self.path, "rb") as f:
            return pickle.load(f)

    def restore_into(self, engine) -> bool:
        s = self.load()
        if s is None:
            return False

        engine.portfolio = s.portfolio
        engine._active_insights = s.engine_state.active_insights
        engine.halt_trading = s.engine_state.halt_trading

        if s.algo_state is not None and hasattr(engine.algorithm, "set_state"):
            try:
                engine.algorithm.set_state(s.algo_state)
            except Exception:
                pass
        return True
