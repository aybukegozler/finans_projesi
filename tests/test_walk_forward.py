from pathlib import Path

import pandas as pd
import pytest

from src.walk_forward import (
    walk_forward_validate,
)


def create_walk_forward_market(
    tmp_path: Path,
    rows: int = 180,
) -> Path:
    path = (
        tmp_path
        / "market_data.csv"
    )

    prices = []

    for index in range(rows):
        trend = index * 0.25

        cycle = (
            9
            if index % 24 < 12
            else -7
        )

        prices.append(
            100
            + trend
            + cycle
        )

    dataframe = pd.DataFrame(
        {
            "Date": pd.date_range(
                "2025-01-01",
                periods=rows,
                freq="D",
            ).strftime(
                "%Y-%m-%d"
            ),

            "Close": prices,
        }
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    return path


def test_walk_forward_creates_unseen_folds(
    tmp_path,
):
    path = create_walk_forward_market(
        tmp_path
    )

    result = walk_forward_validate(
        market_data_path=path,
        short_windows=[
            5,
            10,
        ],
        long_windows=[
            20,
            30,
        ],
        initial_train_size=80,
        test_size=20,
    )

    assert (
        result["summary"]["folds"]
        == 5
    )

    assert len(
        result["folds"]
    ) == 5

    for fold in result["folds"]:
        assert (
            fold["selected_short_window"]
            < fold["selected_long_window"]
        )

        assert (
            fold["test_rows"]
            == 20
        )

        assert (
            fold["train_end"]
            < fold["test_start"]
        )


def test_walk_forward_capital_compounds(
    tmp_path,
):
    path = create_walk_forward_market(
        tmp_path
    )

    result = walk_forward_validate(
        market_data_path=path,
        short_windows=[5],
        long_windows=[20],
        initial_train_size=80,
        test_size=20,
        initial_capital=10_000,
    )

    folds = result["folds"]

    for index in range(
        1,
        len(folds),
    ):
        assert (
            folds[index][
                "test_start_capital"
            ]
            == pytest.approx(
                folds[index - 1][
                    "test_final_value"
                ],
                abs=0.01,
            )
        )


def test_walk_forward_reports_stability(
    tmp_path,
):
    path = create_walk_forward_market(
        tmp_path
    )

    result = walk_forward_validate(
        market_data_path=path,
        short_windows=[
            5,
            10,
        ],
        long_windows=[
            20,
            30,
        ],
        initial_train_size=80,
        test_size=20,
    )

    summary = result["summary"]

    assert (
        "most_selected_pair"
        in summary
    )

    assert (
        summary[
            "most_selected_pair"
        ][
            "times_selected"
        ]
        >= 1
    )

    assert (
        0
        <= summary[
            "profitable_fold_rate_pct"
        ]
        <= 100
    )


def test_walk_forward_rejects_small_training_set(
    tmp_path,
):
    path = create_walk_forward_market(
        tmp_path
    )

    with pytest.raises(
        ValueError
    ):
        walk_forward_validate(
            market_data_path=path,
            short_windows=[5],
            long_windows=[30],
            initial_train_size=20,
            test_size=20,
        )
