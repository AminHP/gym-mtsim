from typing import Tuple

from .interface import MtSymbolInfo


class SymbolInfo:

    def __init__(self, info: MtSymbolInfo) -> None:
        self.name: str = info.name
        self.market: str = self._get_market(info)

        self.currency_margin: str = info.currency_margin
        self.currency_profit: str = info.currency_profit
        self.currencies: Tuple[str, ...] = tuple(set([self.currency_margin, self.currency_profit]))

        self.trade_contract_size: float = info.trade_contract_size
        self.margin_rate: float = 1.0  # MetaTrader info does not contain this value!

        self.volume_min: float = info.volume_min
        self.volume_max: float = info.volume_max
        self.volume_step: float = info.volume_step

    def __str__(self) -> str:
        return f'{self.market}/{self.name}'

    def _get_market(self, info: MtSymbolInfo) -> str:
        mapping = {
            'forex': 'Forex',
            'crypto': 'Crypto',
            'stock': 'Stock',
        }

        root = info.path.split('\\')[0]
        for k, v in mapping.items():
            if root.lower().startswith(k):
                return v

        return root
