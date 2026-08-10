from pathlib import Path

import pandas as pd
import pytest

from src.technical_backtest import (
    build_technical_signal_data,
    run_technical_backtest,
)


def create_market_data(
    tmp_path: Path,
    rows: int = 120,
) -> Path:
    path = (
        tmp_path
        / "market_data.csv"
    )

    prices = []

    for index in range(rows):
        trend = (
            index * 0.20
        )

        cycle = (
            8.0
            if index % 30 < 15
            else -6.0
        )

        prices.append(
            100.0
            + trend
            + cycle
        )

    dataframe = pd.DataFrame(
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
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    return path


def test_technical_signal_data_structure(
    tmp_path,
):
    path = create_market_data(
        tmp_path
    )

    result = (
        build_technical_signal_data(
            market_data_path=path
        )
    )

    assert len(result) == 120

    assert {
        "Date",
        "Close",
        "TechnicalScore",
        "TechnicalRating",
        "RSI14",
        "MACD",
        "MACD_SIGNAL",
        "MACD_HISTOGRAM",
        "BB_UPPER",
        "BB_MIDDLE",
        "BB_LOWER",
        "Signal",
    }.issubset(
        result.columns
    )

    assert set(
        result["Signal"].unique()
    ).issubset(
        {-1, 0, 1}
    )


def test_signals_respect_position_state(
    tmp_path,
):
    path = create_market_data(
        tmp_path
    )

    result = (
        build_technical_signal_data(
            market_data_path=path,
            entry_score=1,
            exit_score=-1,
        )
    )

    position = False

    for signal in result["Signal"]:
        if signal == 1:
            assert position is False
            position = True

        elif signal == -1:
            assert position is True
            position = False


def test_technical_backtest_returns_metrics(
    tmp_path,
):
    path = create_market_data(
        tmp_path
    )

    result = (
        run_technical_backtest(
            market_data_path=path,
            initial_capital=10_000,
            transaction_fee_pct=0.10,
            slippage_pct=0.05,
            entry_score=1,
            exit_score=-1,
        )
    )

    assert (
        result["strategy"]
        == "Multi-Indicator Technical Score"
    )

    assert (
        result["final_value"]
        > 0
    )

    assert (
        "sharpe_ratio"
        in result
    )

    assert (
        "max_drawdown_pct"
        in result
    )


def test_invalid_thresholds_are_rejected(
    tmp_path,
):
    path = create_market_data(
        tmp_path
    )

    with pytest.raises(
        ValueError
    ):
        build_technical_signal_data(
            market_data_path=path,
            entry_score=0,
        )

    with pytest.raises(
        ValueError
    ):
        build_technical_signal_data(
            market_data_path=path,
            exit_score=0,
        )
