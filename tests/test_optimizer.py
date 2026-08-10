from pathlib import Path

import pandas as pd
import pytest

from src.optimizer import (
    evaluate_strategy,
    generate_sma_signals,
    optimize_sma_strategy,
)


def create_market_csv(
    tmp_path: Path,
) -> Path:
    path = (
        tmp_path
        / "market_data.csv"
    )

    prices = [
        100
        + index * 0.7
        + (
            8
            if index % 15 < 7
            else -8
        )
        for index in range(140)
    ]

    dataframe = pd.DataFrame(
        {
            "Date": pd.date_range(
                "2025-01-01",
                periods=len(prices),
                freq="D",
            ).strftime("%Y-%m-%d"),
            "Close": prices,
        }
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    return path


def test_signal_generation_respects_windows(
    tmp_path,
):
    path = create_market_csv(
        tmp_path
    )

    market_data = pd.read_csv(
        path
    )

    result = generate_sma_signals(
        market_data,
        short_window=5,
        long_window=20,
    )

    assert len(result) == len(
        market_data
    )

    assert (
        result.iloc[:20]["Signal"]
        == 0
    ).all()

    assert set(
        result["Signal"].unique()
    ).issubset(
        {-1, 0, 1}
    )


def test_costs_reduce_strategy_value(
    tmp_path,
):
    path = create_market_csv(
        tmp_path
    )

    market_data = pd.read_csv(
        path
    )

    signals = generate_sma_signals(
        market_data,
        short_window=5,
        long_window=20,
    )

    no_cost = evaluate_strategy(
        signals,
        transaction_fee_pct=0,
        slippage_pct=0,
    )

    with_cost = evaluate_strategy(
        signals,
        transaction_fee_pct=1,
        slippage_pct=1,
    )

    assert (
        with_cost["final_value"]
        <= no_cost["final_value"]
    )


def test_optimizer_returns_ranked_results(
    tmp_path,
):
    path = create_market_csv(
        tmp_path
    )

    result = optimize_sma_strategy(
        market_data_path=path,
        short_windows=[
            5,
            10,
        ],
        long_windows=[
            20,
            30,
        ],
        objective="sharpe_ratio",
        top_n=3,
    )

    assert (
        result["tested_combinations"]
        == 4
    )

    assert len(
        result["top_results"]
    ) == 3

    scores = [
        row["sharpe_ratio"]
        for row in result[
            "top_results"
        ]
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    assert (
        result["best"]
        == result["top_results"][0]
    )


def test_optimizer_rejects_invalid_objective(
    tmp_path,
):
    path = create_market_csv(
        tmp_path
    )

    with pytest.raises(
        ValueError
    ):
        optimize_sma_strategy(
            market_data_path=path,
            short_windows=[5],
            long_windows=[20],
            objective="magic_metric",
        )


def test_force_close_at_end_closes_open_position(
    tmp_path,
):
    market_data = pd.DataFrame(
        {
            "Date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ],
            "Close": [
                100.0,
                110.0,
                120.0,
            ],
            "SMA_SHORT": [
                0.0,
                0.0,
                0.0,
            ],
            "SMA_LONG": [
                0.0,
                0.0,
                0.0,
            ],
            "Signal": [
                1,
                0,
                0,
            ],
        }
    )

    result = evaluate_strategy(
        market_data,
        initial_capital=10_000,
        transaction_fee_pct=0,
        slippage_pct=0,
        force_close_at_end=True,
    )

    assert (
        result["closed_trades"]
        == 1
    )

    assert (
        result["winning_trades"]
        == 1
    )

    assert (
        result["final_value"]
        == 12_000.0
    )
