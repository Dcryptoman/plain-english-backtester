"""
The strategy "rulebook" format.

Claude never writes code that we run. Instead it fills in this strict form,
and our own tested engine (backtest.py) runs it. That keeps every backtest
reliable, repeatable and safe, and lets the app show you exactly what the
AI understood before any number is calculated.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Everything a rule can look at. "number" means a fixed value like 30.
OperandType = Literal[
    "close", "open", "high", "low", "volume",
    "sma",          # simple moving average of close
    "ema",          # exponential moving average of close
    "rsi",          # relative strength index (0-100)
    "atr",          # average true range (volatility in price units)
    "macd",         # MACD line (12/26 EMA difference)
    "macd_signal",  # 9-period EMA of the MACD line
    "bb_upper",     # Bollinger upper band (period, 2 std devs)
    "bb_lower",     # Bollinger lower band
    "highest",      # highest high of the PREVIOUS `period` bars (breakouts)
    "lowest",       # lowest low of the PREVIOUS `period` bars
    "number",       # a fixed value, given in `value`
]


class Operand(BaseModel):
    type: OperandType
    period: Optional[int] = Field(
        None, description="Look-back length for sma/ema/rsi/atr/bb_*/highest/lowest. Null for prices, macd and number.")
    value: Optional[float] = Field(None, description="Only for type 'number'.")


class Condition(BaseModel):
    left: Operand
    op: Literal[">", "<", "crosses_above", "crosses_below"]
    right: Operand


class StopLoss(BaseModel):
    type: Literal["none", "percent", "atr"]
    value: Optional[float] = Field(
        None, description="percent: e.g. 2 means 2% below entry (above for shorts). atr: multiple of ATR(14) at entry.")


class TakeProfit(BaseModel):
    type: Literal["none", "percent", "atr", "r_multiple"]
    value: Optional[float] = Field(
        None, description="r_multiple: target = value x the stop distance (needs a stop).")


class Strategy(BaseModel):
    name: str = Field(description="Short title for the strategy.")
    symbol: str = Field(description="Yahoo Finance ticker, e.g. SPY, AAPL, BTC-USD, EURUSD=X, GC=F, ^GSPC.")
    years: int = Field(description="Years of daily history to test (1-20). Use 10 unless the user says otherwise.")
    direction: Literal["long", "short"]
    entry_conditions: list[Condition] = Field(
        description="ALL of these must be true at a bar's close to enter at the next bar's open.")
    exit_conditions: list[Condition] = Field(
        description="If ANY of these is true at a bar's close, exit at the next bar's open. May be empty.")
    stop_loss: StopLoss
    take_profit: TakeProfit
    max_hold_bars: Optional[int] = Field(None, description="Exit after this many bars if still open. Null = no limit.")
    assumptions: list[str] = Field(
        description="Every choice the user did NOT state that you had to make (market, periods, years...).")
    unsupported: list[str] = Field(
        description="Parts of the request this rulebook cannot express. Empty if everything fits.")


NEEDS_PERIOD = {"sma", "ema", "rsi", "atr", "bb_upper", "bb_lower", "highest", "lowest"}


def validate(s: Strategy) -> list[str]:
    """Problems that would make the backtest meaningless. Empty = OK."""
    problems = []
    if not s.entry_conditions:
        problems.append("No entry rule was found.")
    if not 1 <= s.years <= 20:
        problems.append("Years of history must be between 1 and 20.")
    for c in s.entry_conditions + s.exit_conditions:
        for o in (c.left, c.right):
            if o.type in NEEDS_PERIOD and not (o.period and o.period > 0):
                problems.append(f"{o.type} needs a look-back period.")
            if o.type == "number" and o.value is None:
                problems.append("A fixed number is missing its value.")
    for name, level in (("Stop loss", s.stop_loss), ("Take profit", s.take_profit)):
        if level.type != "none" and not (level.value and level.value > 0):
            problems.append(f"{name} needs a positive value.")
    if s.take_profit.type == "r_multiple" and s.stop_loss.type == "none":
        problems.append("An R-multiple target needs a stop loss to measure R from.")
    if s.max_hold_bars is not None and s.max_hold_bars < 1:
        problems.append("Max hold must be at least 1 bar.")
    return problems


def describe_operand(o: Operand) -> str:
    if o.type == "number":
        return f"{o.value:g}"
    names = {
        "close": "close", "open": "open", "high": "high", "low": "low", "volume": "volume",
        "sma": "SMA", "ema": "EMA", "rsi": "RSI", "atr": "ATR", "macd": "MACD",
        "macd_signal": "MACD signal", "bb_upper": "upper Bollinger", "bb_lower": "lower Bollinger",
        "highest": "highest high of last", "lowest": "lowest low of last",
    }
    label = names[o.type]
    if o.type in ("highest", "lowest"):
        return f"{label} {o.period} bars"
    return f"{label}({o.period})" if o.period else label


def describe_condition(c: Condition) -> str:
    op = {">": "is above", "<": "is below",
          "crosses_above": "crosses above", "crosses_below": "crosses below"}[c.op]
    return f"{describe_operand(c.left)} {op} {describe_operand(c.right)}"


def describe_rules(s: Strategy) -> list[str]:
    """The rulebook as plain-English lines, shown to the user to check."""
    lines = [f"Market: {s.symbol}, daily bars, last {s.years} years, {s.direction} only"]
    lines.append("Enter when " + " AND ".join(describe_condition(c) for c in s.entry_conditions))
    if s.exit_conditions:
        lines.append("Exit when " + " OR ".join(describe_condition(c) for c in s.exit_conditions))
    if s.stop_loss.type == "percent":
        lines.append(f"Stop loss: {s.stop_loss.value:g}% from entry")
    elif s.stop_loss.type == "atr":
        lines.append(f"Stop loss: {s.stop_loss.value:g} x ATR(14) from entry")
    tp = s.take_profit
    if tp.type == "percent":
        lines.append(f"Take profit: {tp.value:g}% from entry")
    elif tp.type == "atr":
        lines.append(f"Take profit: {tp.value:g} x ATR(14) from entry")
    elif tp.type == "r_multiple":
        lines.append(f"Take profit: {tp.value:g}R (risk multiples of the stop distance)")
    if s.max_hold_bars:
        lines.append(f"Time exit: after {s.max_hold_bars} bars")
    return lines
