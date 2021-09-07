from gym.envs.registration import register

from .metatrader import Timeframe, SymbolInfo
from .simulator import MtSimulator, OrderType, Order, SymbolNotFound, OrderNotFound
from .envs import MtEnv
from .data import FOREX_DATA_PATH, STOCKS_DATA_PATH, CRYPTO_DATA_PATH, MIXED_DATA_PATH


register(
    id='forex-hedge-v0',
    entry_point='gym_mtsim.envs:MtEnv',
    kwargs={
        'original_simulator': MtSimulator(symbols_filename=FOREX_DATA_PATH, hedge=True),
        'trading_symbols': ['EURUSD', 'GBPCAD', 'USDJPY'],
        'window_size': 10,
        'symbol_max_orders': 2,
        'fee': lambda symbol: 0.03 if 'JPY' in symbol else 0.0003
    }
)

register(
    id='forex-unhedge-v0',
    entry_point='gym_mtsim.envs:MtEnv',
    kwargs={
        'original_simulator': MtSimulator(symbols_filename=FOREX_DATA_PATH, hedge=False),
        'trading_symbols': ['EURUSD', 'GBPCAD', 'USDJPY'],
        'window_size': 10,
        'fee': lambda symbol: 0.03 if 'JPY' in symbol else 0.0003
    }
)

register(
    id='stocks-hedge-v0',
    entry_point='gym_mtsim.envs:MtEnv',
    kwargs={
        'original_simulator': MtSimulator(symbols_filename=STOCKS_DATA_PATH, hedge=True),
        'trading_symbols': ['GOGL', 'AAPL', 'TSLA', 'MSFT'],
        'window_size': 10,
        'symbol_max_orders': 2,
        'fee': 0.2
    }
)

register(
    id='stocks-unhedge-v0',
    entry_point='gym_mtsim.envs:MtEnv',
    kwargs={
        'original_simulator': MtSimulator(symbols_filename=STOCKS_DATA_PATH, hedge=False),
        'trading_symbols': ['GOGL', 'AAPL', 'TSLA', 'MSFT'],
        'window_size': 10,
        'fee': 0.2
    }
)

register(
    id='crypto-hedge-v0',
    entry_point='gym_mtsim.envs:MtEnv',
    kwargs={
        'original_simulator': MtSimulator(symbols_filename=CRYPTO_DATA_PATH, hedge=True),
        'trading_symbols': ['BTCUSD', 'ETHUSD', 'BCHUSD'],
        'window_size': 10,
        'symbol_max_orders': 2,
        'fee': lambda symbol: {
            'BTCUSD': 50.0,
            'ETHUSD': 3.0,
            'BCHUSD': 0.5,
        }[symbol]
    }
)

register(
    id='crypto-unhedge-v0',
    entry_point='gym_mtsim.envs:MtEnv',
    kwargs={
        'original_simulator': MtSimulator(symbols_filename=CRYPTO_DATA_PATH, hedge=False),
        'trading_symbols': ['BTCUSD', 'ETHUSD', 'BCHUSD'],
        'window_size': 10,
        'fee': lambda symbol: {
            'BTCUSD': 50.0,
            'ETHUSD': 3.0,
            'BCHUSD': 0.5,
        }[symbol]
    }
)

register(
    id='mixed-hedge-v0',
    entry_point='gym_mtsim.envs:MtEnv',
    kwargs={
        'original_simulator': MtSimulator(symbols_filename=MIXED_DATA_PATH, hedge=True),
        'trading_symbols': ['EURUSD', 'USDCAD', 'GOGL', 'AAPL', 'BTCUSD', 'ETHUSD'],
        'window_size': 10,
        'symbol_max_orders': 2,
        'fee': lambda symbol: {
            'EURUSD': 0.0002,
            'USDCAD': 0.0005,
            'GOGL': 0.15,
            'AAPL': 0.01,
            'BTCUSD': 50.0,
            'ETHUSD': 3.0,
        }[symbol]
    }
)

register(
    id='mixed-unhedge-v0',
    entry_point='gym_mtsim.envs:MtEnv',
    kwargs={
        'original_simulator': MtSimulator(symbols_filename=MIXED_DATA_PATH, hedge=False),
        'trading_symbols': ['EURUSD', 'USDCAD', 'GOGL', 'AAPL', 'BTCUSD', 'ETHUSD'],
        'window_size': 10,
        'fee': lambda symbol: {
            'EURUSD': 0.0002,
            'USDCAD': 0.0005,
            'GOGL': 0.15,
            'AAPL': 0.01,
            'BTCUSD': 50.0,
            'ETHUSD': 3.0,
        }[symbol]
    }
)
