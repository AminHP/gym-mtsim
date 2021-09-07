import os


DATA_DIR = os.path.dirname(os.path.abspath(__file__))

FOREX_DATA_PATH = os.path.join(DATA_DIR, 'symbols_forex.pkl')
STOCKS_DATA_PATH = os.path.join(DATA_DIR, 'symbols_stocks.pkl')
CRYPTO_DATA_PATH = os.path.join(DATA_DIR, 'symbols_crypto.pkl')
MIXED_DATA_PATH = os.path.join(DATA_DIR, 'symbols_mixed.pkl')
