from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from src.optimizer import (
    DEFAULT_MARKET_DATA_PATH,
    evaluate_strategy,
    generate_sma_signals,
    load_market_data,
)


DEFAULT_SHORT_WINDOWS = (
    5,
    10,
    15,
    20,
    25,
    30,
)

DEFAULT_LONG_WINDOWS = (
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
)

ALLOWED_OBJECTIVES = {
    "sharpe_ratio",
    "total_return_pct",
    "excess_return_pct",
}


def select_best_parameters(
    train_data,
    short_windows: Iterable[int],
    long_windows: Iterable[int],
    initial_capital: float,
    transaction_fee_pct: float,
    slippage_pct: float,
    objective: str,
) -> dict:
    if objective not in ALLOWED_OBJECTIVES:
        raise ValueError(
            "objective must be one of: "
            + ", ".join(sorted(ALLOWED_OBJECTIVES))
        )

    results: list[dict] = []

    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue

            if len(train_data) <= long_window:
                continue

            signals = generate_sma_signals(
                train_data,
                short_window=short_window,
                long_window=long_window,
            )

            metrics = evaluate_strategy(
                signals,
                initial_capital=initial_capital,
                transaction_fee_pct=transaction_fee_pct,
                slippage_pct=slippage_pct,
            )

            results.append(
                {
                    "short_window": short_window,
                    "long_window": long_window,
                    **metrics,
                }
            )

    if not results:
        raise ValueError(
            "No valid SMA combinations could be tested."
        )

    results.sort(
        key=lambda row: (
            row[objective],
            row["total_return_pct"],
        ),
        reverse=True,
    )

    return results[0]


def build_out_of_sample_signals(
    market_data,
    train_end: int,
    test_end: int,
    short_window: int,
    long_window: int,
):
    # Test başlangıcındaki SMA değerlerini hesaplayabilmek için
    # yalnızca geçmişten gerekli warm-up satırlarını kullanıyoruz.
    warmup_start = max(
        0,
        train_end - long_window - 1,
    )

    context = market_data.iloc[
        warmup_start:test_end
    ].reset_index(drop=True)

    signals = generate_sma_signals(
        context,
        short_window=short_window,
        long_window=long_window,
    )

    test_offset = (
        train_end - warmup_start
    )

    test_signals = signals.iloc[
        test_offset:
    ].reset_index(drop=True)

    return test_signals


def walk_forward_validate(
    market_data_path: Path | str = DEFAULT_MARKET_DATA_PATH,
    short_windows: Iterable[int] = DEFAULT_SHORT_WINDOWS,
    long_windows: Iterable[int] = DEFAULT_LONG_WINDOWS,
    initial_train_size: int = 250,
    test_size: int = 50,
    initial_capital: float = 10_000.0,
    transaction_fee_pct: float = 0.10,
    slippage_pct: float = 0.05,
    objective: str = "sharpe_ratio",
) -> dict:
    if objective not in ALLOWED_OBJECTIVES:
        raise ValueError(
            "Invalid optimization objective."
        )

    if initial_capital <= 0:
        raise ValueError(
            "initial_capital must be greater than zero."
        )

    if test_size <= 0:
        raise ValueError(
            "test_size must be greater than zero."
        )

    short_windows = tuple(short_windows)
    long_windows = tuple(long_windows)

    if not short_windows or not long_windows:
        raise ValueError(
            "SMA window lists cannot be empty."
        )

    max_long_window = max(long_windows)

    if initial_train_size <= max_long_window:
        raise ValueError(
            "initial_train_size must be greater "
            "than the largest long SMA window."
        )

    market_data = load_market_data(
        market_data_path
    )

    if (
        len(market_data)
        < initial_train_size + test_size
    ):
        raise ValueError(
            "Not enough data for walk-forward validation."
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

        train_data = market_data.iloc[
            :train_end
        ].reset_index(drop=True)

        best = select_best_parameters(
            train_data=train_data,
            short_windows=short_windows,
            long_windows=long_windows,
            initial_capital=strategy_capital,
            transaction_fee_pct=transaction_fee_pct,
            slippage_pct=slippage_pct,
            objective=objective,
        )

        short_window = int(
            best["short_window"]
        )

        long_window = int(
            best["long_window"]
        )

        test_signals = build_out_of_sample_signals(
            market_data=market_data,
            train_end=train_end,
            test_end=test_end,
            short_window=short_window,
            long_window=long_window,
        )

        fold_start_capital = (
            strategy_capital
        )

        test_metrics = evaluate_strategy(
            test_signals,
            initial_capital=fold_start_capital,
            transaction_fee_pct=transaction_fee_pct,
            slippage_pct=slippage_pct,
        )

        strategy_capital = float(
            test_metrics["final_value"]
        )

        benchmark_fold_return = (
            test_metrics[
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
                "fold": fold_number,

                "train_start":
                    str(
                        train_data.iloc[0][
                            "Date"
                        ]
                    ),

                "train_end":
                    str(
                        train_data.iloc[-1][
                            "Date"
                        ]
                    ),

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

                "train_rows":
                    len(train_data),

                "test_rows":
                    len(test_signals),

                "selected_short_window":
                    short_window,

                "selected_long_window":
                    long_window,

                "training_objective_score":
                    best[objective],

                "training_return_pct":
                    best[
                        "total_return_pct"
                    ],

                "test_start_capital":
                    round(
                        fold_start_capital,
                        2,
                    ),

                "test_final_value":
                    test_metrics[
                        "final_value"
                    ],

                "test_return_pct":
                    test_metrics[
                        "total_return_pct"
                    ],

                "test_buy_hold_return_pct":
                    test_metrics[
                        "buy_hold_return_pct"
                    ],

                "test_excess_return_pct":
                    test_metrics[
                        "excess_return_pct"
                    ],

                "test_sharpe_ratio":
                    test_metrics[
                        "sharpe_ratio"
                    ],

                "test_max_drawdown_pct":
                    test_metrics[
                        "max_drawdown_pct"
                    ],

                "test_closed_trades":
                    test_metrics[
                        "closed_trades"
                    ],
            }
        )

        fold_number += 1
        train_end += test_size

    if not folds:
        raise ValueError(
            "Walk-forward produced no test folds."
        )

    parameter_pairs = [
        (
            fold[
                "selected_short_window"
            ],
            fold[
                "selected_long_window"
            ],
        )
        for fold in folds
    ]

    most_common_pair, frequency = (
        Counter(
            parameter_pairs
        ).most_common(1)[0]
    )

    profitable_folds = sum(
        1
        for fold in folds
        if fold["test_return_pct"] > 0
    )

    strategy_return_pct = (
        (
            strategy_capital
            / initial_capital
        )
        - 1.0
    ) * 100.0

    benchmark_return_pct = (
        (
            benchmark_capital
            / initial_capital
        )
        - 1.0
    ) * 100.0

    return {
        "method":
            "Expanding Window Walk-Forward",

        "objective":
            objective,

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

            "out_of_sample_return_pct":
                round(
                    strategy_return_pct,
                    2,
                ),

            "benchmark_final_value":
                round(
                    benchmark_capital,
                    2,
                ),

            "benchmark_return_pct":
                round(
                    benchmark_return_pct,
                    2,
                ),

            "excess_return_pct":
                round(
                    strategy_return_pct
                    - benchmark_return_pct,
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

            "average_test_return_pct":
                round(
                    mean(
                        fold[
                            "test_return_pct"
                        ]
                        for fold in folds
                    ),
                    2,
                ),

            "median_test_return_pct":
                round(
                    median(
                        fold[
                            "test_return_pct"
                        ]
                        for fold in folds
                    ),
                    2,
                ),

            "average_test_sharpe":
                round(
                    mean(
                        fold[
                            "test_sharpe_ratio"
                        ]
                        for fold in folds
                    ),
                    3,
                ),

            "median_test_sharpe":
                round(
                    median(
                        fold[
                            "test_sharpe_ratio"
                        ]
                        for fold in folds
                    ),
                    3,
                ),

            "total_closed_trades":
                sum(
                    fold[
                        "test_closed_trades"
                    ]
                    for fold in folds
                ),

            "zero_closed_trade_folds":
                sum(
                    1
                    for fold in folds
                    if fold[
                        "test_closed_trades"
                    ] == 0
                ),

            "worst_fold_drawdown_pct":
                round(
                    max(
                        fold[
                            "test_max_drawdown_pct"
                        ]
                        for fold in folds
                    ),
                    2,
                ),

            "most_selected_pair":
                {
                    "short_window":
                        most_common_pair[0],

                    "long_window":
                        most_common_pair[1],

                    "times_selected":
                        frequency,

                    "selection_rate_pct":
                        round(
                            frequency
                            / len(folds)
                            * 100.0,
                            2,
                        ),
                },
        },

        "folds": folds,
    }


if __name__ == "__main__":
    result = walk_forward_validate()

    summary = result["summary"]

    print(
        "=== WALK-FORWARD VALIDATION ==="
    )

    print(
        "Method:",
        result["method"],
    )

    print(
        "Objective:",
        result["objective"],
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
        f"{summary['out_of_sample_return_pct']}%",
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
        "Average Test Sharpe:",
        summary["average_test_sharpe"],
    )

    print(
        "Worst Fold Drawdown:",
        f"{summary['worst_fold_drawdown_pct']}%",
    )

    pair = summary[
        "most_selected_pair"
    ]

    print(
        "Most Selected Pair:",
        f"SMA{pair['short_window']}/"
        f"SMA{pair['long_window']}",
    )

    print("\nFOLDS")

    for fold in result["folds"]:
        print(
            f"Fold {fold['fold']} | "
            f"SMA"
            f"{fold['selected_short_window']}/"
            f"SMA"
            f"{fold['selected_long_window']} | "
            f"Test Return: "
            f"{fold['test_return_pct']}% | "
            f"Sharpe: "
            f"{fold['test_sharpe_ratio']} | "
            f"DD: "
            f"{fold['test_max_drawdown_pct']}%"
        )
