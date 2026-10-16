"""
The whole pipeline, plus a terminal version:

    python run.py "Buy SPY when RSI(2) drops below 10 above the 200 SMA, exit when close > 5 SMA"

English -> rulebook (Claude) -> prices (Yahoo) -> backtest -> robustness -> verdict (Claude)
"""

import sys
from dataclasses import dataclass

import anthropic

import ai
from backtest import BacktestResult, run_backtest
from data import DataError, load_prices
from robustness import monte_carlo, out_of_sample, robustness_score
from strategy_spec import Strategy, describe_rules, validate


@dataclass
class Report:
    strategy: Strategy
    result: BacktestResult
    oos: dict
    mc: dict
    score: dict
    verdict: ai.Verdict | None


class StrategyError(Exception):
    """A problem the user can fix by rewording (shown as a friendly message)."""


def understand(text: str) -> Strategy:
    strategy = ai.parse_strategy(text)
    problems = validate(strategy)
    if problems:
        detail = " ".join(problems + strategy.unsupported)
        raise StrategyError(f"I couldn't turn that into a testable strategy: {detail}")
    return strategy


def test_strategy(strategy: Strategy, with_verdict: bool = True) -> Report:
    try:
        df = load_prices(strategy.symbol, strategy.years)
    except DataError as error:
        raise StrategyError(str(error))
    result = run_backtest(df, strategy)
    oos, mc = out_of_sample(result), monte_carlo(result)
    score = robustness_score(result, oos, mc)
    verdict = ai.write_verdict(strategy, result.metrics, oos, mc, score) if with_verdict and result.trades else None
    return Report(strategy, result, oos, mc, score, verdict)


def main():
    text = " ".join(sys.argv[1:]) or input("Describe your strategy: ").strip()
    if not text:
        return
    try:
        print("\nReading your strategy...")
        strategy = understand(text)
        print("\nWhat I understood:")
        for line in describe_rules(strategy):
            print("  -", line)
        for a in strategy.assumptions:
            print("  (assumed)", a)
        for u in strategy.unsupported:
            print("  (not modelled)", u)
        print("\nDownloading prices and backtesting...")
        report = test_strategy(strategy)
    except (StrategyError, ai.AIError, ai.MissingAPIKeyError) as error:
        print("\n" + str(error))
        return
    except anthropic.AuthenticationError:
        print("\nYour API key was rejected. Check ANTHROPIC_API_KEY in .env.")
        return
    except anthropic.APIConnectionError:
        print("\nCouldn't reach Claude. Check your internet connection.")
        return

    m = report.result.metrics
    print(f"\n{m['first_date']} to {m['last_date']}  |  {m['trades']} trades  |  win rate {m['win_rate_pct']:.0f}%")
    print(f"Return {m['total_return_pct']:+.1f}% (buy & hold {m['buy_hold_return_pct']:+.1f}%)  |  "
          f"max drawdown {m['max_drawdown_pct']:.1f}%  |  profit factor {m['profit_factor']:.2f}")
    print(f"\nROBUSTNESS: {report.score['score']}/100 - {report.score['grade']}")
    for name, points in report.score["parts"].items():
        print(f"  {points:4.1f}/20  {name}")
    v = report.verdict
    if v:
        print(f"\n{v.headline}")
        for title, items in (("What works", v.what_works), ("Why it could fail", v.why_it_could_fail),
                             ("Try next", v.try_next)):
            if items:
                print(f"\n{title}:")
                for item in items:
                    print("  -", item)
    else:
        print("\nNo trades were triggered, so there's nothing to judge. Try looser rules.")
    print("\nResearch only - not financial advice.")


if __name__ == "__main__":
    main()
