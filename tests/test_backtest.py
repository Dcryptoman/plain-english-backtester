"""
Tests for the engine, on tiny hand-made price tables where the right answer
can be worked out on paper. Run with:  python -m pytest
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backtest  # noqa: E402
from backtest import condition_series, run_backtest  # noqa: E402
from indicators import rsi  # noqa: E402
from strategy_spec import (Condition, Operand, StopLoss, Strategy,  # noqa: E402
                           TakeProfit, validate)


@pytest.fixture(autouse=True)
def no_fees(monkeypatch):
    monkeypatch.setattr(backtest, "FEE_PCT", 0.0)


def prices(opens, highs=None, lows=None, closes=None):
    n = len(opens)
    closes = closes or opens
    return pd.DataFrame({
        "Open": opens, "High": highs or [max(o, c) for o, c in zip(opens, closes)],
        "Low": lows or [min(o, c) for o, c in zip(opens, closes)],
        "Close": closes, "Volume": [1000] * n,
    }, index=pd.date_range("2020-01-01", periods=n, freq="D"))


def num(v):
    return Operand(type="number", value=v)


def close():
    return Operand(type="close")


def strategy(entry, exits=(), stop=None, target=None, direction="long", max_hold=None):
    return Strategy(
        name="test", symbol="TEST", years=1, direction=direction,
        entry_conditions=list(entry), exit_conditions=list(exits),
        stop_loss=stop or StopLoss(type="none"), take_profit=target or TakeProfit(type="none"),
        max_hold_bars=max_hold, assumptions=[], unsupported=[],
    )


def test_enters_next_open_and_exits_next_open():
    # Close goes above 100 on bar 1 -> buy bar 2 open (102).
    # Close drops below 100 on bar 3 -> sell bar 4 open (95).
    df = prices(opens=[99, 101, 102, 98, 95, 96], closes=[99, 101, 103, 98, 95, 96])
    s = strategy([Condition(left=close(), op=">", right=num(100))],
                 [Condition(left=close(), op="<", right=num(100))])
    result = run_backtest(df, s)
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.entry_price == 102 and t.exit_price == 95
    assert t.exit_reason == "exit rule"
    assert t.return_pct == pytest.approx((95 / 102 - 1) * 100)
    assert result.equity.iloc[-1] == pytest.approx(95 / 102)


def test_percent_stop_fills_at_stop_price():
    df = prices(opens=[100, 100, 100, 100], highs=[100, 101, 100, 100],
                lows=[100, 100, 94, 100], closes=[101, 100, 96, 100])
    s = strategy([Condition(left=close(), op=">", right=num(100))],
                 stop=StopLoss(type="percent", value=5))
    t = run_backtest(df, s).trades[0]
    assert t.exit_reason == "stop loss"
    assert t.exit_price == pytest.approx(95)
    assert t.r_multiple == pytest.approx(-1)


def test_stop_wins_when_stop_and_target_hit_same_bar():
    df = prices(opens=[100, 100, 100], highs=[100, 120, 100], lows=[100, 80, 100],
                closes=[101, 100, 100])
    s = strategy([Condition(left=close(), op=">", right=num(100))],
                 stop=StopLoss(type="percent", value=5),
                 target=TakeProfit(type="r_multiple", value=2))
    t = run_backtest(df, s).trades[0]
    assert t.exit_reason == "stop loss"


def test_r_multiple_target():
    # Entry 100, stop 5% -> risk 5, 2R target = 110.
    df = prices(opens=[100, 100, 105, 108], highs=[100, 104, 111, 108],
                lows=[100, 99, 104, 108], closes=[101, 103, 109, 108])
    s = strategy([Condition(left=close(), op=">", right=num(100))],
                 stop=StopLoss(type="percent", value=5),
                 target=TakeProfit(type="r_multiple", value=2))
    t = run_backtest(df, s).trades[0]
    assert t.exit_reason == "take profit"
    assert t.exit_price == pytest.approx(110)
    assert t.r_multiple == pytest.approx(2)


def test_gap_through_stop_fills_at_open():
    df = prices(opens=[100, 100, 90, 90], highs=[100, 100, 91, 90],
                lows=[100, 99, 89, 90], closes=[101, 100, 90, 90])
    s = strategy([Condition(left=close(), op=">", right=num(100))],
                 stop=StopLoss(type="percent", value=5))
    t = run_backtest(df, s).trades[0]
    assert t.exit_price == 90  # worse than the 95 stop: the market gapped


def test_short_profits_when_price_falls():
    df = prices(opens=[100, 100, 90, 80], closes=[99, 95, 85, 80])
    s = strategy([Condition(left=close(), op="<", right=num(100))], direction="short", max_hold=1)
    t = run_backtest(df, s).trades[0]
    assert t.entry_price == 100 and t.exit_price == 90
    assert t.return_pct == pytest.approx(10)


def test_time_exit_and_end_of_data():
    df = prices(opens=[100] * 6, closes=[101] * 6)
    s = strategy([Condition(left=close(), op=">", right=num(100))], max_hold=2)
    trades = run_backtest(df, s).trades
    assert [t.exit_reason for t in trades] == ["time exit", "end of data"]


def test_crosses_above_only_fires_on_the_cross():
    df = prices(opens=[1] * 5, closes=[99, 101, 102, 99, 103])
    fired = condition_series(df, Condition(left=close(), op="crosses_above", right=num(100)))
    assert fired.tolist() == [False, True, False, False, True]


def test_fees_reduce_returns(monkeypatch):
    monkeypatch.setattr(backtest, "FEE_PCT", 0.1)
    df = prices(opens=[100, 100, 100, 100], closes=[101, 100, 100, 100])
    s = strategy([Condition(left=close(), op=">", right=num(100))], max_hold=1)
    t = run_backtest(df, s).trades[0]
    assert t.return_pct == pytest.approx((0.999 * 0.999 - 1) * 100)


def test_rsi_extremes():
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100)
    assert math.isnan(rsi(up, 14).iloc[5])


def test_validate_catches_bad_rulebooks():
    bad = strategy([Condition(left=Operand(type="sma"), op=">", right=num(1))],
                   target=TakeProfit(type="r_multiple", value=2))
    problems = validate(bad)
    assert any("look-back" in p for p in problems)
    assert any("R-multiple" in p for p in problems)
