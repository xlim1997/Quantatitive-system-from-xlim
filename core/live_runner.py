# core/live_runner.py
from __future__ import annotations

import time
from typing import Optional
import pandas as pd

from core.state_store import StateStore


class LiveRunner:
    """
    Live 模式运行器：
    - 循环从 data_feed 读取 events
    - 调用 engine.step()
    - 每 N 步保存 state
    - 可选写 CSV 日志
    """
    def __init__(
        self,
        engine,
        state_path: str = "state/last.pkl",
        save_every: int = 1,
        log_path: Optional[str] = "logs/live_equity.csv",
    ) -> None:
        self.engine = engine
        self.store = StateStore(state_path)
        self.save_every = save_every
        self.log_path = log_path
        if log_path:
            import os
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def restore_if_exists(self) -> bool:
        return self.store.restore_into(self.engine)

    def run_forever(self) -> None:
        i = 0
        for market_slice in self.engine.data_feed:
            rec = self.engine.step(market_slice)
            if rec is None:
                continue

            i += 1
            if i % self.save_every == 0:
                self.store.save(self.engine)

            if self.log_path:
                df = pd.DataFrame([{
                    "timestamp": rec.timestamp,
                    "equity": rec.equity,
                    "cash": rec.cash,
                    "realized_pnl": rec.realized_pnl,
                }])
                # append
                header = False
                import os
                if not os.path.exists(self.log_path):
                    header = True
                df.to_csv(self.log_path, mode="a", header=header, index=False)
            if i % 10 == 0:
                print(f"[LiveRunner] Processed {i} market slices. Equity: {rec.equity:.2f}, Cash: {rec.cash:.2f}, Realized PnL: {rec.realized_pnl:.2f}")
            # 如果触发停机：退出循环
            if getattr(self.engine, "halt_trading", False) or getattr(self.engine.portfolio, "halt_trading", False):
                self.store.save(self.engine)
                break
