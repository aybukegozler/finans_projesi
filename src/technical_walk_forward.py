from __future__ import annotations

from pathlib import Path
from statistics import mean, median

import pandas as pd

from src.optimizer import (
    DEFAULT_MARKET_DATA_PATH,
    evaluate_strategy,
    load_market_data,
)
from src.technical_indicators import (
    calculate_technical_snapshot,
)


def build_oos_technical_signals(
    market_data: pd.DataFrame,
    train_end: int,
    test_end: int,
    entry_score: int = 2,
    exit_score: int = -2,
    warmup_rows: int = 80,
) -> pd.DataFrame:
    if not 1 <= entry_score <= 4:
        raise ValueError(
            "entry_score must be between 1 and 4."
        )

    if not -4 <= exit_score <= -1:
        raise ValueError(
            "exit_score must be between -4 and -1."
        )

    warmup_start = max(
        0,
        train_end - warmup_rows,
    )

    context = market_data.iloc[
        warmup_start:test_end
    ].reset_index(drop=True)

    closes: list[float] = []
    rows: list[dict] = []

    in_position = False

    for index, row in enumerate(
        context.itertuples(index=False)
    ):
        close = float(
            row.Close
        )

        closes.append(close)

        technical = (
            calculate_technical_snapshot(
                closes
            )
        )

        global_index = (
            warmup_start + index
        )

        # Warm-up / training geçmişi yalnızca
        # indikatörleri hazırlamak için kullanılır.
        # Trading yalnızca test döneminde başlar.
        if global_index < train_end:
            continue

        signal = 0

        if (
            technical["ready"]
            and technical["score"]
            is not None
        ):
            if (
                not in_position
                and technical["score"]
                >= entry_score
            ):
                signal = 1
                in_position = True

            elif (
                in_position
                and technical["score"]
                <= exit_score
            ):
                signal = -1
                in_position = False

        rows.append(
            {
                "Date": row.Date,
                "Close": close,
                "Signal": signal,
                "TechnicalScore":
                    technical["score"],
                "TechnicalRating":
                    technical["rating"],
            }
        )

    return pd.DataFrame(rows)


def run_technical_walk_forward(
    market_data_path: Path | str = DEFAULT_MARKET_DATA_PATH,
    initial_train_size: int = 250,
    test_size: int = 50,
    entry_score: int = 2,
    exit_score: int = -2,
    initial_capital: float = 10_000.0,
    transaction_fee_pct: float = 0.10,
    slippage_pct: float = 0.05,
) -> dict:
    if initial_capital <= 0:
        raise ValueError(
            "initial_capital must be greater than zero."
        )

    if initial_train_size < 100:
        raise ValueError(
            "initial_train_size must be at least 100."
        )

    if test_size <= 0:
        raise ValueError(
            "test_size must be greater than zero."
        )

    market_data = load_market_data(
        market_data_path
    )

    if (
        len(market_data)
        < initial_train_size + test_size
    ):
        raise ValueError(
            "Not enough market data."
        )

    strategy_capital = float(
        initial_capital
    )

    benchmark_capital = float(
        initial_capital
    )

    folds: list[dict] = []

    train_end = initial_train_size
    fold_number = 1

    while (
        train_end + test_size
        <= len(market_data)
    ):
        test_end = (
            train_end + test_size
        )

        test_signals = (
            build_oos_technical_signals(
                market_data=market_data,
                train_end=train_end,
                test_end=test_end,
                entry_score=entry_score,
                exit_score=exit_score,
            )
        )

        if test_signals.empty:
            raise ValueError(
                "Walk-forward produced an empty test fold."
            )

        fold_start_capital = (
            strategy_capital
        )

        metrics = evaluate_strategy(
            test_signals,
            initial_capital=fold_start_capital,
            transaction_fee_pct=transaction_fee_pct,
            slippage_pct=slippage_pct,
            force_close_at_end=True,
        )

        strategy_capital = float(
            metrics["final_value"]
        )

        benchmark_fold_return = (
            metrics[
                "buy_hold_return_pct"
            ]
            / 100.0
        )

        benchmark_capital *= (
            1.0
            + benchmark_fold_return
        )

        folds.append(
            {
                "fold":
                    fold_number,

                "train_rows":
                    train_end,

                "test_rows":
                    len(test_signals),

                "test_start":
                    str(
                        test_signals.iloc[0][
                            "Date"
                        ]
                    ),

                "test_end":
                    str(
                        test_signals.iloc[-1][
                            "Date"
                        ]
                    ),

                "start_capital":
                    round(
                        fold_start_capital,
                        2,
                    ),

                "final_value":
                    metrics[
                        "final_value"
                    ],

                "return_pct":
                    metrics[
                        "total_return_pct"
                    ],

                "benchmark_return_pct":
                    metrics[
                        "buy_hold_return_pct"
                    ],

                "excess_return_pct":
                    metrics[
                        "excess_return_pct"
                    ],

                "sharpe_ratio":
                    metrics[
                        "sharpe_ratio"
                    ],

                "max_drawdown_pct":
                    metrics[
                        "max_drawdown_pct"
                    ],

                "closed_trades":
                    metrics[
                        "closed_trades"
                    ],
            }
        )

        fold_number += 1
        train_end += test_size

    if not folds:
        raise ValueError(
            "No walk-forward folds produced."
        )

    strategy_return = (
        (
            strategy_capital
            / initial_capital
        )
        - 1.0
    ) * 100.0

    benchmark_return = (
        (
            benchmark_capital
            / initial_capital
        )
        - 1.0
    ) * 100.0

    returns = [
        fold["return_pct"]
        for fold in folds
    ]

    sharpes = [
        fold["sharpe_ratio"]
        for fold in folds
    ]

    profitable_folds = sum(
        1
        for value in returns
        if value > 0
    )

    return {
        "strategy":
            "Technical Score Walk-Forward",

        "entry_score":
            entry_score,

        "exit_score":
            exit_score,

        "summary": {
            "folds":
                len(folds),

            "initial_capital":
                round(
                    initial_capital,
                    2,
                ),

            "final_value":
                round(
                    strategy_capital,
                    2,
                ),

            "oos_return_pct":
                round(
                    strategy_return,
                    2,
                ),

            "benchmark_final_value":
                round(
                    benchmark_capital,
                    2,
                ),

            "benchmark_return_pct":
                round(
                    benchmark_return,
                    2,
                ),

            "excess_return_pct":
                round(
                    strategy_return
                    - benchmark_return,
                    2,
                ),

            "profitable_folds":
                profitable_folds,

            "profitable_fold_rate_pct":
                round(
                    profitable_folds
                    / len(folds)
                    * 100.0,
                    2,
                ),

            "average_fold_return_pct":
                round(
                    mean(returns),
                    2,
                ),

            "median_fold_return_pct":
                round(
                    median(returns),
                    2,
                ),

            "average_fold_sharpe":
                round(
                    mean(sharpes),
                    3,
                ),

            "median_fold_sharpe":
                round(
                    median(sharpes),
                    3,
                ),

            "worst_fold_drawdown_pct":
                round(
                    max(
                        fold[
                            "max_drawdown_pct"
                        ]
                        for fold in folds
                    ),
                    2,
                ),

            "total_closed_trades":
                sum(
                    fold[
                        "closed_trades"
                    ]
                    for fold in folds
                ),

            "zero_trade_folds":
                sum(
                    1
                    for fold in folds
                    if fold[
                        "closed_trades"
                    ] == 0
                ),
        },

        "folds":
            folds,
    }


if __name__ == "__main__":
    result = (
        run_technical_walk_forward()
    )

    summary = result["summary"]

    print(
        "=== TECHNICAL SCORE WALK-FORWARD ==="
    )

    print(
        "Entry / Exit:",
        result["entry_score"],
        "/",
        result["exit_score"],
    )

    print(
        "Folds:",
        summary["folds"],
    )

    print(
        "OOS Final Value:",
        summary["final_value"],
    )

    print(
        "OOS Return:",
        f"{summary['oos_return_pct']}%",
    )

    print(
        "Benchmark:",
        f"{summary['benchmark_return_pct']}%",
    )

    print(
        "Excess Return:",
        f"{summary['excess_return_pct']}%",
    )

    print(
        "Profitable Folds:",
        f"{summary['profitable_folds']}"
        f"/{summary['folds']}",
    )

    print(
        "Average Fold Sharpe:",
        summary["average_fold_sharpe"],
    )

    print(
        "Median Fold Sharpe:",
        summary["median_fold_sharpe"],
    )

    print(
        "Worst Fold DD:",
        f"{summary['worst_fold_drawdown_pct']}%",
    )

    print(
        "Closed Trades:",
        summary["total_closed_trades"],
    )

    print("\nFOLDS")

    for fold in result["folds"]:
        print(
            f"Fold {fold['fold']} | "
            f"{fold['test_start']} → "
            f"{fold['test_end']} | "
            f"Return: "
            f"{fold['return_pct']}% | "
            f"Benchmark: "
            f"{fold['benchmark_return_pct']}% | "
            f"Sharpe: "
            f"{fold['sharpe_ratio']} | "
            f"DD: "
            f"{fold['max_drawdown_pct']}% | "
            f"Trades: "
            f"{fold['closed_trades']}"
        )
