"""
Indicator maths. Each function takes a price table (columns Open, High, Low,
Close, Volume) and returns one number per bar. The first few bars are NaN
("not a number") until there is enough history - no trades happen there.
"""

import pandas as pd

from strategy_spec import Operand


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    # Wilder's RSI, the version TradingView and most platforms use.
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def operand_series(df: pd.DataFrame, o: Operand) -> pd.Series:
    """Turn one side of a rule (e.g. 'RSI(14)' or '30') into a series."""
    close = df["Close"]
    t = o.type
    if t == "number":
        return pd.Series(o.value, index=df.index)
    if t in ("close", "open", "high", "low", "volume"):
        return df[t.capitalize()]
    if t == "sma":
        return sma(close, o.period)
    if t == "ema":
        return ema(close, o.period)
    if t == "rsi":
        return rsi(close, o.period)
    if t == "atr":
        return atr(df, o.period)
    if t in ("macd", "macd_signal"):
        line = ema(close, 12) - ema(close, 26)
        return line if t == "macd" else line.ewm(span=9, adjust=False, min_periods=9).mean()
    if t in ("bb_upper", "bb_lower"):
        mid = sma(close, o.period)
        width = 2 * close.rolling(o.period).std(ddof=0)
        return mid + width if t == "bb_upper" else mid - width
    if t == "highest":
        # shift(1): today's breakout is measured against the bars BEFORE today
        return df["High"].rolling(o.period).max().shift(1)
    if t == "lowest":
        return df["Low"].rolling(o.period).min().shift(1)
    raise ValueError(f"Unknown operand type: {t}")
