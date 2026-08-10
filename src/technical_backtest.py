from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.optimizer import (
    DEFAULT_MARKET_DATA_PATH,
    evaluate_strategy,
    load_market_data,
)
from src.technical_indicators import (
    calculate_technical_snapshot,
)


def build_technical_signal_data(
    market_data_path: Path | str = DEFAULT_MARKET_DATA_PATH,
    entry_score: int = 2,
    exit_score: int = -2,
) -> pd.DataFrame:
    if not 1 <= entry_score <= 4:
        raise ValueError(
            "entry_score must be between 1 and 4."
        )

    if not -4 <= exit_score <= -1:
        raise ValueError(
            "exit_score must be between -4 and -1."
        )

    market_data = load_market_data(
        market_data_path
    )

    closes: list[float] = []

    rows: list[dict] = []

    in_position = False

    for row in market_data.itertuples(
        index=False
    ):
        close = float(
            row.Close
        )

        closes.append(
            close
        )

        technical = (
            calculate_technical_snapshot(
                closes
            )
        )

        score = (
            technical["score"]
        )

        signal = 0

        if (
            technical["ready"]
            and score is not None
        ):
            if (
                not in_position
                and score >= entry_score
            ):
                signal = 1
                in_position = True

            elif (
                in_position
                and score <= exit_score
            ):
                signal = -1
                in_position = False

        macd = (
            technical["macd"]
        )

        bollinger = (
            technical["bollinger"]
        )

        rows.append(
            {
                "Date":
                    row.Date,

                "Close":
                    close,

                "TechnicalScore":
                    score,

                "TechnicalRating":
                    technical["rating"],

                "RSI14":
                    technical["rsi14"],

                "MACD":
                    macd["macd"],

                "MACD_SIGNAL":
                    macd["signal"],

                "MACD_HISTOGRAM":
                    macd["histogram"],

                "BB_UPPER":
                    bollinger["upper"],

                "BB_MIDDLE":
                    bollinger["middle"],

                "BB_LOWER":
                    bollinger["lower"],

                "Signal":
                    signal,
            }
        )

    return pd.DataFrame(
        rows
    )


def run_technical_backtest(
    market_data_path: Path | str = DEFAULT_MARKET_DATA_PATH,
    initial_capital: float = 10_000.0,
    transaction_fee_pct: float = 0.10,
    slippage_pct: float = 0.05,
    entry_score: int = 2,
    exit_score: int = -2,
) -> dict:
    signal_data = (
        build_technical_signal_data(
            market_data_path=market_data_path,
            entry_score=entry_score,
            exit_score=exit_score,
        )
    )

    metrics = evaluate_strategy(
        signal_data,
        initial_capital=initial_capital,
        transaction_fee_pct=transaction_fee_pct,
        slippage_pct=slippage_pct,
    )

    buy_signals = int(
        (
            signal_data["Signal"]
            == 1
        ).sum()
    )

    sell_signals = int(
        (
            signal_data["Signal"]
            == -1
        ).sum()
    )

    ready_rows = int(
        signal_data[
            "TechnicalScore"
        ].notna().sum()
    )

    return {
        "strategy":
            "Multi-Indicator Technical Score",

        "entry_score":
            entry_score,

        "exit_score":
            exit_score,

        "ready_rows":
            ready_rows,

        "buy_signals":
            buy_signals,

        "sell_signals":
            sell_signals,

        **metrics,
    }


if __name__ == "__main__":
    result = (
        run_technical_backtest()
    )

    print(
        "=== TECHNICAL SCORE BACKTEST ==="
    )

    print(
        "Strategy:",
        result["strategy"],
    )

    print(
        "Entry threshold:",
        result["entry_score"],
    )

    print(
        "Exit threshold:",
        result["exit_score"],
    )

    print(
        "Ready rows:",
        result["ready_rows"],
    )

    print(
        "BUY signals:",
        result["buy_signals"],
    )

    print(
        "SELL signals:",
        result["sell_signals"],
    )

    print(
        "Final value:",
        result["final_value"],
    )

    print(
        "Return:",
        f"{result['total_return_pct']}%",
    )

    print(
        "Buy & Hold:",
        f"{result['buy_hold_return_pct']}%",
    )

    print(
        "Excess return:",
        f"{result['excess_return_pct']}%",
    )

    print(
        "Closed trades:",
        result["closed_trades"],
    )

    print(
        "Win rate:",
        f"{result['win_rate_pct']}%",
    )

    print(
        "Sharpe:",
        result["sharpe_ratio"],
    )

    print(
        "Max drawdown:",
        f"{result['max_drawdown_pct']}%",
    )

    print(
        "Volatility:",
        f"{result['annualized_volatility_pct']}%",
    )

    print(
        "Fees:",
        result["total_fees_paid"],
    )
