# universe/ibkr_scanner_universe.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from ib_insync import IB, ScannerSubscription, Stock, util


@dataclass
class IBKRScannerConfig:
    instrument: str = "STK"
    location_code: str = "STK.US.MAJOR"
    scan_code: str = "HOT_BY_VOLUME"

    # 可选：这些属于 ScannerSubscription 的过滤字段（是否需要订阅视你的账户而定）
    min_price: float = 5.0
    min_volume: int = 200_000

    # 取前多少个候选（scanner 本身通常最多 50）
    take: int = 50


def scan_universe(ib: IB, cfg: IBKRScannerConfig) -> List[Stock]:
    """
    返回 Stock contracts 列表（候选池）。
    注意：scanner 只返回合约，不含行情字段；你后面仍要 reqHistoricalData/reqMktData 去算因子。:contentReference[oaicite:5]{index=5}
    """
    sub = ScannerSubscription(
        instrument=cfg.instrument,
        locationCode=cfg.location_code,
        scanCode=cfg.scan_code,
    )
    # old-style filters:
    sub.abovePrice = cfg.min_price
    sub.aboveVolume = cfg.min_volume

    scanData = ib.reqScannerData(sub)  # ib_insync notebook 示例就是这么用的:contentReference[oaicite:6]{index=6}
    scanData = scanData[: cfg.take]

    contracts = []
    for sd in scanData:
        c = sd.contractDetails.contract
        # 用 SMART 路由，避免交易所细节
        contracts.append(Stock(c.symbol, "SMART", c.currency or "USD"))

    ib.qualifyContracts(*contracts)
    return contracts
