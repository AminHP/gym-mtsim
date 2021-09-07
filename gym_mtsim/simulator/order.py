from enum import IntEnum
from datetime import datetime


class OrderType(IntEnum):
    Sell = 0
    Buy = 1

    @property
    def sign(self) -> float:
        return 1. if self == OrderType.Buy else -1.

    @property
    def opposite(self) -> 'OrderType':
        if self == OrderType.Sell:
            return OrderType.Buy
        return OrderType.Sell


class Order:

    def __init__(
        self,
        id: int, type: OrderType, symbol: str, volume: float, fee: float,
        entry_time: datetime, entry_price: float,
        exit_time: datetime, exit_price: float
    ) -> None:

        self.id = id
        self.type = type
        self.symbol = symbol
        self.volume = volume
        self.fee = fee
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.profit = 0.
        self.margin = 0.
        self.closed = False
