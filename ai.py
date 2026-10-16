"""
The two Claude calls:
1. parse_strategy: plain English -> a validated Strategy rulebook
   (structured output: Claude must fill in the exact form in strategy_spec.py).
2. write_verdict: the measured numbers -> an honest plain-English verdict.
Claude never sees or invents the numbers in step 2; they come from the engine.
"""

import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from strategy_spec import Strategy, describe_rules

load_dotenv(Path(__file__).parent / ".env")

MODEL = "claude-opus-5-5"


class MissingAPIKeyError(RuntimeError):
    pass


class AIError(RuntimeError):
    pass


def get_client() -> anthropic.Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MissingAPIKeyError(
            "No API key found. Copy .env.example to .env in this folder and put "
            "your key after ANTHROPIC_API_KEY= (see README.md).")
    return anthropic.Anthropic()


PARSER_PROMPT = """You turn a trader's plain-English strategy into a strict rulebook for a daily-bar backtester.

How the backtester works:
- Daily bars only. Entry rules are ANDed and checked at the close; the trade opens at the next open.
- Exit rules are ORed. Stops and targets are checked intrabar. One position at a time, long OR short.
- "highest"/"lowest" compare against the PREVIOUS N bars, so "close crosses above highest 20" or "close > highest 20" is a 20-day breakout.

Rules for you:
- Map the user's words faithfully. Common defaults when unstated: RSI 14, ATR 14, Bollinger 20, 10 years of history, SPY if no market is named. Record EVERY default you choose in `assumptions`, in plain English.
- Use Yahoo Finance tickers: Bitcoin BTC-USD, Ethereum ETH-USD, gold GC=F, oil CL=F, S&P 500 ^GSPC (or SPY), Nasdaq ^IXIC or QQQ, DAX ^GDAXI, FTSE ^FTSE, EUR/USD EURUSD=X.
- "2R" / "2:1 reward to risk" means take_profit r_multiple 2 and needs a stop.
- If something cannot be expressed (intraday times, news, position scaling, several markets at once, indicators not in the list), approximate it only if the approximation is honest, otherwise leave it out - and always list it in `unsupported`.
- If the text is not a trading strategy at all, return an empty entry_conditions list and explain in `unsupported`."""


def parse_strategy(text: str) -> Strategy:
    response = get_client().messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=PARSER_PROMPT,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": text}],
        output_format=Strategy,
    )
    if response.stop_reason == "refusal":
        raise AIError("Claude declined to read this request. Try rephrasing it as a trading rule.")
    if response.parsed_output is None:
        raise AIError("Claude's answer couldn't be read as a rulebook. Try rephrasing.")
    return response.parsed_output


class Verdict(BaseModel):
    headline: str = Field(description="One punchy, honest sentence summing up the result.")
    what_works: list[str] = Field(description="1-3 genuine strengths, each citing a number. Empty if none.")
    why_it_could_fail: list[str] = Field(description="2-4 concrete weaknesses or risks, each citing a number.")
    try_next: list[str] = Field(description="2-3 specific experiments to test next (rule changes to backtest, not trades to place).")


VERDICT_PROMPT = """You are a sceptical quant reviewing a backtest for a trader. Be honest and specific, in plain English a non-coder understands.
- Use ONLY the numbers provided; cite them. Never invent figures.
- The robustness score and grade are computed by code - explain them, do not override them.
- Judge with care: few trades = noise; a strong first 70% with a weak last 30% = probably overfitted; a strategy that loses to buy-and-hold on risk-adjusted terms adds little.
- Never tell the user to buy, sell or trade anything. This is research, not financial advice."""


def _jsonable(report: dict) -> str:
    def default(o):
        return round(o, 3) if isinstance(o, float) else str(o)
    return json.dumps(report, default=default, indent=1)


def write_verdict(strategy: Strategy, metrics: dict, oos: dict, mc: dict, score: dict) -> Verdict:
    report = {
        "rules": describe_rules(strategy),
        "assumptions": strategy.assumptions,
        "not_modelled": strategy.unsupported,
        "full_period": metrics,
        "out_of_sample_split": oos,
        "monte_carlo": {k: v for k, v in mc.items() if k != "bands"},
        "robustness": score,
    }
    response = get_client().messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=VERDICT_PROMPT,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": "Backtest report:\n" + _jsonable(report)}],
        output_format=Verdict,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise AIError("Claude couldn't write a verdict for this result.")
    return response.parsed_output
