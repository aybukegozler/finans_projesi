from pathlib import Path

import pandas as pd
import pytest

from src.technical_walk_forward import (
    build_oos_technical_signals,
    run_technical_walk_forward,
)


def create_market(
    tmp_path: Path,
    rows: int = 220,
) -> Path:
    path = (
        tmp_path
        / "market_data.csv"
    )

    prices = []

    for index in range(rows):
        base = (
            100
            + index * 0.15
        )

        cycle = (
            12
            if index % 40 < 20
            else -8
        )

        prices.append(
            base + cycle
        )

    pd.DataFrame(
        {
            "Date":
                pd.date_range(
                    "2025-01-01",
                    periods=rows,
                    freq="D",
                ).strftime(
                    "%Y-%m-%d"
                ),

            "Close":
                prices,
        }
    ).to_csv(
        path,
        index=False,
    )

    return path


def test_oos_signals_only_contain_test_rows(
    tmp_path,
):
    path = create_market(
        tmp_path
    )

    market = pd.read_csv(
        path
    )

    result = (
        build_oos_technical_signals(
            market_data=market,
            train_end=120,
            test_end=140,
        )
    )

    assert len(result) == 20

    assert (
        result.iloc[0]["Date"]
        == market.iloc[120]["Date"]
    )


def test_oos_signals_are_valid(
    tmp_path,
):
    path = create_market(
        tmp_path
    )

    market = pd.read_csv(
        path
    )

    result = (
        build_oos_technical_signals(
            market_data=market,
            train_end=120,
            test_end=160,
            entry_score=1,
            exit_score=-1,
        )
    )

    assert set(
        result["Signal"]
    ).issubset(
        {-1, 0, 1}
    )


def test_technical_walk_forward_runs(
    tmp_path,
):
    path = create_market(
        tmp_path
    )

    result = (
        run_technical_walk_forward(
            market_data_path=path,
            initial_train_size=120,
            test_size=20,
            entry_score=1,
            exit_score=-1,
        )
    )

    assert (
        result["summary"]["folds"]
        == 5
    )

    assert (
        result["summary"]["final_value"]
        > 0
    )

    assert (
        len(result["folds"])
        == 5
    )


def test_invalid_training_size(
    tmp_path,
):
    path = create_market(
        tmp_path
    )

    with pytest.raises(
        ValueError
    ):
        run_technical_walk_forward(
            market_data_path=path,
            initial_train_size=50,
            test_size=20,
        )
