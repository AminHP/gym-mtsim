from enum import Enum
from datetime import datetime

import numpy as np

import MetaTrader5 as mt5
from MetaTrader5 import SymbolInfo as MtSymbolInfo


class Timeframe(Enum):
    M1 = mt5.TIMEFRAME_M1
    M2 = mt5.TIMEFRAME_M2
    M3 = mt5.TIMEFRAME_M3
    M4 = mt5.TIMEFRAME_M4
    M5 = mt5.TIMEFRAME_M5
    M6 = mt5.TIMEFRAME_M6
    M10 = mt5.TIMEFRAME_M10
    M12 = mt5.TIMEFRAME_M12
    M15 = mt5.TIMEFRAME_M15
    M20 = mt5.TIMEFRAME_M20
    M30 = mt5.TIMEFRAME_M30
    H1 = mt5.TIMEFRAME_H1
    H2 = mt5.TIMEFRAME_H2
    H4 = mt5.TIMEFRAME_H4
    H3 = mt5.TIMEFRAME_H3
    H6 = mt5.TIMEFRAME_H6
    H8 = mt5.TIMEFRAME_H8
    H12 = mt5.TIMEFRAME_H12
    D1 = mt5.TIMEFRAME_D1
    W1 = mt5.TIMEFRAME_W1
    MN1 = mt5.TIMEFRAME_MN1


def initialize() -> bool:
    return mt5.initialize()


def shutdown() -> None:
    mt5.shutdown()


def copy_rates_range(symbol: str, timeframe: Timeframe, date_from: datetime, date_to: datetime) -> np.ndarray:
    return mt5.copy_rates_range(symbol, timeframe.value, date_from, date_to)


def symbol_info(symbol: str) -> MtSymbolInfo:
    return mt5.symbol_info(symbol)
