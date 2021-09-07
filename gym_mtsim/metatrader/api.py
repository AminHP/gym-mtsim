from typing import Tuple

import pytz
import calendar
from datetime import datetime, timedelta

import pandas as pd

from . import interface as mt
from .symbol import SymbolInfo


def retrieve_data(
        symbol: str, from_dt: datetime, to_dt: datetime, timeframe: mt.Timeframe
    ) -> Tuple[SymbolInfo, pd.DataFrame]:

    if not mt.initialize():
        raise ConnectionError(f"MetaTrader cannot be initialized")

    symbol_info = _get_symbol_info(symbol)

    utc_from = _local2utc(from_dt)
    utc_to = _local2utc(to_dt)
    all_rates = []

    partial_from = utc_from
    partial_to = _add_months(partial_from, 1)

    while partial_from < utc_to:
        rates = mt.copy_rates_range(symbol, timeframe, partial_from, partial_to)
        all_rates.extend(rates)
        partial_from = _add_months(partial_from, 1)
        partial_to = min(_add_months(partial_to, 1), utc_to)

    all_rates = [list(r) for r in all_rates]

    rates_frame = pd.DataFrame(
        all_rates,
        columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', '_', '_'],
    )
    rates_frame['Time'] = pd.to_datetime(rates_frame['Time'], unit='s', utc=True)

    data = rates_frame[['Time', 'Open', 'Close', 'Low', 'High', 'Volume']].set_index('Time')
    data = data.loc[~data.index.duplicated(keep='first')]

    mt.shutdown()

    return symbol_info, data


def _get_symbol_info(symbol: str) -> SymbolInfo:
    info = mt.symbol_info(symbol)
    symbol_info = SymbolInfo(info)
    return symbol_info


def _local2utc(dt: datetime) -> datetime:
    return dt.astimezone(pytz.timezone('Etc/UTC'))


def _add_months(sourcedate: datetime, months: int) -> datetime:
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])

    return datetime(
        year, month, day,
        sourcedate.hour, sourcedate.minute, sourcedate.second,
        tzinfo=sourcedate.tzinfo
    )
