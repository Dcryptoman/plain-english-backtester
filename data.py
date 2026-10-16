"""Daily price history from Yahoo Finance (free, no key needed)."""

import logging

import pandas as pd
import yfinance as yf

# yfinance prints its own noisy errors; we raise a friendly one instead.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


class DataError(Exception):
    pass


def load_prices(symbol: str, years: int) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=f"{years}y", interval="1d",
                         auto_adjust=True, progress=False)
    except Exception as error:
        raise DataError(f"Download failed for {symbol}: {error}")
    if df is None or df.empty:
        raise DataError(f"No data found for '{symbol}'. Check the ticker (Yahoo Finance format, e.g. AAPL, BTC-USD, EURUSD=X).")
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance adds a ticker level
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    if len(df) < 100:
        raise DataError(f"Only {len(df)} bars of data for {symbol} - not enough to test a strategy.")
    return df
