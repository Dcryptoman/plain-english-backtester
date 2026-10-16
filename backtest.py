"""
The backtest engine: replays history bar by bar and applies the rulebook.

Rules of the simulation (kept deliberately conservative):
- Signals are checked at a bar's CLOSE; orders fill at the NEXT bar's OPEN.
  (You can't trade on a close you haven't seen yet.)
- Stops and targets are checked against each bar's high and low. If both
  could have been hit in the same bar, we assume the STOP was hit first.
- If the price gaps past a stop or target overnight, the fill is the open.
- Every trade pays FEE_PCT on the way in and on the way out.
- All money is in one position at a time (compounding), no leverage.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators import atr, operand_series
from strategy_spec import Condition, Strategy

FEE_PCT = 0.05  # per side, in percent: covers commission + spread + slippage
STOP_ATR_PERIOD = 14


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str
    bars_held: int
    return_pct: float          # after fees
    r_multiple: float | None   # result in units of the initial risk (needs a stop)


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series          # strategy account value, starting at 1.0
    buy_hold: pd.Series        # buy-and-hold account value, starting at 1.0
    metrics: dict
    prices: pd.DataFrame       # the price history that was tested


def condition_series(df: pd.DataFrame, c: Condition) -> pd.Series:
    left = operand_series(df, c.left)
    right = operand_series(df, c.right)
    if c.op == ">":
        result = left > right
    elif c.op == "<":
        result = left < right
    elif c.op == "crosses_above":
        result = (left > right) & (left.shift(1) <= right.shift(1))
    else:  # crosses_below
        result = (left < right) & (left.shift(1) >= right.shift(1))
    # Comparisons with NaN (not enough history yet) are already False.
    return result.fillna(False).astype(bool)


def all_true(df: pd.DataFrame, conditions: list[Condition]) -> pd.Series:
    out = pd.Series(True, index=df.index)
    for c in conditions:
        out &= condition_series(df, c)
    return out


def any_true(df: pd.DataFrame, conditions: list[Condition]) -> pd.Series:
    out = pd.Series(False, index=df.index)
    for c in conditions:
        out |= condition_series(df, c)
    return out


def _levels(strategy: Strategy, entry: float, atr_value: float, side: int):
    """Stop and target prices for a new trade (None = not used)."""
    stop = target = None
    sl, tp = strategy.stop_loss, strategy.take_profit
    if sl.type == "percent":
        stop = entry * (1 - side * sl.value / 100)
    elif sl.type == "atr" and not np.isnan(atr_value):
        stop = entry - side * sl.value * atr_value
    if tp.type == "percent":
        target = entry * (1 + side * tp.value / 100)
    elif tp.type == "atr" and not np.isnan(atr_value):
        target = entry + side * tp.value * atr_value
    elif tp.type == "r_multiple" and stop is not None:
        target = entry + tp.value * (entry - stop)
    return stop, target


def run_backtest(df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
    side = 1 if strategy.direction == "long" else -1
    fee = FEE_PCT / 100
    entries = all_true(df, strategy.entry_conditions).to_numpy()
    exits = any_true(df, strategy.exit_conditions).to_numpy()
    atr_values = atr(df, STOP_ATR_PERIOD).to_numpy()
    opens, highs, lows, closes = (df[c].to_numpy(dtype=float) for c in ("Open", "High", "Low", "Close"))
    dates = df.index

    trades: list[Trade] = []
    equity = np.ones(len(df))
    cash = 1.0
    in_trade = False
    entry_i = 0
    entry_price = stop = target = None
    pending_entry = pending_exit = False
    pending_reason = ""

    def close_trade(i: int, price: float, reason: str):
        nonlocal cash, in_trade
        gross = side * (price / entry_price - 1)
        ret = (1 - fee) * (1 + gross) * (1 - fee) - 1
        r = None
        if stop is not None and entry_price != stop:
            r = (price - entry_price) * side / abs(entry_price - stop)
        cash *= 1 + ret
        trades.append(Trade(dates[entry_i], entry_price, dates[i], price, reason,
                            i - entry_i + 1, ret * 100, r))
        in_trade = False

    for i in range(len(df)):
        # 1) Orders decided at the previous close fill at today's open.
        if pending_exit and in_trade:
            close_trade(i, opens[i], pending_reason)
        if pending_entry and not in_trade:
            in_trade = True
            entry_i = i
            entry_price = opens[i]
            stop, target = _levels(strategy, entry_price, atr_values[i - 1], side)
        pending_entry = pending_exit = False

        # 2) Intrabar stop / target (stop first if both are touched).
        if in_trade:
            hit_stop = stop is not None and (lows[i] <= stop if side == 1 else highs[i] >= stop)
            hit_target = target is not None and (highs[i] >= target if side == 1 else lows[i] <= target)
            if hit_stop:
                gapped = opens[i] <= stop if side == 1 else opens[i] >= stop
                close_trade(i, opens[i] if gapped else stop, "stop loss")
            elif hit_target:
                gapped = opens[i] >= target if side == 1 else opens[i] <= target
                close_trade(i, opens[i] if gapped else target, "take profit")

        # 3) Mark the account to today's close.
        if in_trade:
            equity[i] = cash * (1 - fee) * (1 + side * (closes[i] / entry_price - 1))
        else:
            equity[i] = cash

        # 4) Decide tomorrow's orders from today's close.
        last_bar = i == len(df) - 1
        if in_trade and not last_bar:
            if exits[i]:
                pending_exit, pending_reason = True, "exit rule"
            elif strategy.max_hold_bars and i - entry_i + 1 >= strategy.max_hold_bars:
                pending_exit, pending_reason = True, "time exit"
        elif not in_trade and not last_bar and entries[i]:
            pending_entry = True

    if in_trade:  # still open at the end of the data: close at the last close
        close_trade(len(df) - 1, closes[-1], "end of data")
        equity[-1] = cash

    equity_s = pd.Series(equity, index=dates, name="Strategy")
    buy_hold = pd.Series(closes / opens[0], index=dates, name="Buy & hold")
    if side == -1:
        buy_hold.name = "Buy & hold (long)"
    return BacktestResult(trades, equity_s, buy_hold, compute_metrics(trades, equity_s, buy_hold, df), df)


def max_drawdown_pct(values: pd.Series | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    peak = np.maximum.accumulate(values)
    return float(((values / peak) - 1).min() * 100)


def trade_stats(trades: list[Trade]) -> dict:
    """Stats that only need the list of trades (reused for the out-of-sample check)."""
    rets = np.array([t.return_pct for t in trades])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = float("inf") if gross_win > 0 else 0.0
    compounded = (np.prod(1 + rets / 100) - 1) * 100 if len(rets) else 0.0
    return {
        "trades": len(trades),
        "win_rate_pct": float(len(wins) / len(rets) * 100) if len(rets) else 0.0,
        "avg_win_pct": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_pct": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(profit_factor),
        "avg_trade_pct": float(rets.mean()) if len(rets) else 0.0,
        "total_return_pct": float(compounded),
    }


def compute_metrics(trades: list[Trade], equity: pd.Series, buy_hold: pd.Series, df: pd.DataFrame) -> dict:
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
    bars_per_year = len(df) / years

    def cagr(series: pd.Series) -> float:
        end = series.iloc[-1]
        return float((end ** (1 / years) - 1) * 100) if end > 0 else -100.0

    daily = equity.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(bars_per_year)) if daily.std() > 0 else 0.0
    r_values = [t.r_multiple for t in trades if t.r_multiple is not None]
    stats = trade_stats(trades)
    stats.update({
        "total_return_pct": float((equity.iloc[-1] - 1) * 100),
        "cagr_pct": cagr(equity),
        "max_drawdown_pct": max_drawdown_pct(equity),
        "sharpe": sharpe,
        "buy_hold_return_pct": float((buy_hold.iloc[-1] - 1) * 100),
        "buy_hold_cagr_pct": cagr(buy_hold),
        "buy_hold_max_drawdown_pct": max_drawdown_pct(buy_hold),
        "time_in_market_pct": float(sum(t.bars_held for t in trades) / len(df) * 100),
        "avg_bars_held": float(np.mean([t.bars_held for t in trades])) if trades else 0.0,
        "avg_r": float(np.mean(r_values)) if r_values else None,
        "years_tested": float(years),
        "first_date": str(df.index[0].date()),
        "last_date": str(df.index[-1].date()),
        "fee_pct_per_side": FEE_PCT,
    })
    return stats
