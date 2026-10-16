"""
Is the backtest result real, or luck? Three checks, then one score.

1. Out-of-sample: split the history 70% / 30%. A strategy that only works in
   the first part was probably fitted to old data.
2. Monte Carlo "luck check": shuffle the trades (drawing with replacement)
   thousands of times. If many reshuffles lose money, the good result
   depended on a lucky order of trades.
3. Buy-and-hold: was all this effort better than simply holding?

The robustness score is plain arithmetic on those numbers, NOT an AI
opinion, so the AI cannot hype a weak strategy.
"""

import numpy as np
import pandas as pd

from backtest import BacktestResult, trade_stats

IN_SAMPLE_SHARE = 0.7
MONTE_CARLO_RUNS = 2000
MIN_TRADES = 30  # fewer than this and the stats are mostly noise


def out_of_sample(result: BacktestResult) -> dict:
    dates = result.equity.index
    split_date = dates[int(len(dates) * IN_SAMPLE_SHARE)]
    first = [t for t in result.trades if t.entry_date < split_date]
    last = [t for t in result.trades if t.entry_date >= split_date]
    return {"split_date": str(split_date.date()),
            "in_sample": trade_stats(first), "out_of_sample": trade_stats(last)}


def monte_carlo(result: BacktestResult, runs: int = MONTE_CARLO_RUNS, seed: int = 7) -> dict:
    rets = np.array([t.return_pct for t in result.trades]) / 100
    if len(rets) < 2:
        return {"runs": 0}
    rng = np.random.default_rng(seed)  # fixed seed: same strategy, same answer
    samples = rng.choice(rets, size=(runs, len(rets)), replace=True)
    paths = np.cumprod(1 + samples, axis=1)
    final = (paths[:, -1] - 1) * 100
    peaks = np.maximum.accumulate(paths, axis=1)
    drawdowns = ((paths / peaks) - 1).min(axis=1) * 100
    bands = np.percentile(paths, [5, 50, 95], axis=0)
    return {
        "runs": runs,
        "prob_loss_pct": float((final < 0).mean() * 100),
        "return_p5_pct": float(np.percentile(final, 5)),
        "return_median_pct": float(np.median(final)),
        "return_p95_pct": float(np.percentile(final, 95)),
        "worst_drawdown_p95_pct": float(np.percentile(drawdowns, 5)),  # 1-in-20 bad case
        "bands": pd.DataFrame({"p5": bands[0], "median": bands[1], "p95": bands[2]},
                              index=np.arange(1, len(rets) + 1)),
    }


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def robustness_score(result: BacktestResult, oos: dict, mc: dict) -> dict:
    """0-100, five parts worth 20 points each. Each part is explained to the user."""
    m = result.metrics
    oos_stats = oos["out_of_sample"]
    parts = {}

    parts["Enough trades"] = 20 * _clip01(m["trades"] / MIN_TRADES)

    parts["Profit factor"] = 20 * _clip01((min(m["profit_factor"], 10) - 1) / 1.0)  # 2.0+ = full marks

    if oos_stats["trades"] >= 5:
        parts["Works on unseen data"] = 20 * _clip01((min(oos_stats["profit_factor"], 10) - 1) / 0.5)
    else:
        parts["Works on unseen data"] = 0.0

    parts["Survives the luck check"] = 20 * _clip01(1 - mc.get("prob_loss_pct", 100) / 100) if mc.get("runs") else 0.0

    # Return per unit of pain (CAGR / max drawdown) versus buy-and-hold.
    def efficiency(cagr, dd):
        return cagr / abs(dd) if dd < 0 else (cagr if cagr > 0 else 0)
    mine = efficiency(m["cagr_pct"], m["max_drawdown_pct"])
    hold = efficiency(m["buy_hold_cagr_pct"], m["buy_hold_max_drawdown_pct"])
    if mine <= 0:
        parts["Beats buy & hold (risk-adjusted)"] = 0.0
    elif hold <= 0:
        parts["Beats buy & hold (risk-adjusted)"] = 20.0
    else:
        parts["Beats buy & hold (risk-adjusted)"] = 20 * _clip01(mine / hold)

    score = round(sum(parts.values()))
    if score >= 70:
        grade = "Robust"
    elif score >= 45:
        grade = "Fragile"
    else:
        grade = "Doesn't hold up"
    return {"score": score, "grade": grade, "parts": {k: round(v, 1) for k, v in parts.items()}}
