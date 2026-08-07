from pathlib import Path

import pandas as pd
import pytest

from src.backtest import (
    calculate_max_drawdown,
    run_backtest,
)


def create_test_csv(
    tmp_path: Path,
    prices: list[float],
    signals: list[int],
) -> Path:
    path = tmp_path / "signals.csv"

    dataframe = pd.DataFrame(
        {
            "Date": [
                f"2026-01-{index + 1:02d}"
                for index in range(len(prices))
            ],
            "Close": prices,
            "Signal": signals,
        }
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    return path


def test_profitable_trade(tmp_path):
    path = create_test_csv(
        tmp_path,
        prices=[
            100.0,
            110.0,
            120.0,
        ],
        signals=[
            1,
            0,
            -1,
        ],
    )

    result = run_backtest(
        path,
        initial_capital=10_000.0,
    )

    summary = result["summary"]

    assert summary["final_value"] == 12_000.0
    assert summary["total_return_pct"] == 20.0

    assert summary["closed_trades"] == 1
    assert summary["winning_trades"] == 1
    assert summary["losing_trades"] == 0
    assert summary["win_rate_pct"] == 100.0


def test_losing_trade(tmp_path):
    path = create_test_csv(
        tmp_path,
        prices=[
            100.0,
            90.0,
        ],
        signals=[
            1,
            -1,
        ],
    )

    result = run_backtest(path)

    summary = result["summary"]

    assert summary["final_value"] == 9_000.0
    assert summary["total_return_pct"] == -10.0

    assert summary["closed_trades"] == 1
    assert summary["winning_trades"] == 0
    assert summary["losing_trades"] == 1


def test_open_position_is_marked_to_market(
    tmp_path,
):
    path = create_test_csv(
        tmp_path,
        prices=[
            100.0,
            150.0,
        ],
        signals=[
            1,
            0,
        ],
    )

    result = run_backtest(path)

    assert (
        result["summary"]["final_value"]
        == 15_000.0
    )

    assert (
        result["summary"]["closed_trades"]
        == 0
    )


def test_max_drawdown():
    curve = [
        100.0,
        120.0,
        90.0,
        110.0,
    ]

    drawdown = calculate_max_drawdown(curve)

    assert drawdown == pytest.approx(25.0)


def test_invalid_capital_is_rejected(
    tmp_path,
):
    path = create_test_csv(
        tmp_path,
        prices=[100.0],
        signals=[0],
    )

    with pytest.raises(ValueError):
        run_backtest(
            path,
            initial_capital=0,
        )


def test_transaction_fee_reduces_return(
    tmp_path,
):
    path = create_test_csv(
        tmp_path,
        prices=[
            100.0,
            120.0,
        ],
        signals=[
            1,
            -1,
        ],
    )

    without_fee = run_backtest(
        path,
        initial_capital=10_000,
    )

    with_fee = run_backtest(
        path,
        initial_capital=10_000,
        transaction_fee_pct=1.0,
    )

    assert (
        with_fee["summary"]["final_value"]
        < without_fee["summary"]["final_value"]
    )

    assert (
        with_fee["summary"]["total_fees_paid"]
        > 0
    )


def test_slippage_reduces_return(
    tmp_path,
):
    path = create_test_csv(
        tmp_path,
        prices=[
            100.0,
            120.0,
        ],
        signals=[
            1,
            -1,
        ],
    )

    without_slippage = run_backtest(
        path,
        initial_capital=10_000,
    )

    with_slippage = run_backtest(
        path,
        initial_capital=10_000,
        slippage_pct=1.0,
    )

    assert (
        with_slippage["summary"]["final_value"]
        < without_slippage["summary"]["final_value"]
    )


def test_risk_metrics_are_returned(
    tmp_path,
):
    path = create_test_csv(
        tmp_path,
        prices=[
            100.0,
            110.0,
            90.0,
            115.0,
            105.0,
        ],
        signals=[
            1,
            0,
            0,
            0,
            -1,
        ],
    )

    result = run_backtest(
        path
    )

    summary = result["summary"]

    assert "sharpe_ratio" in summary
    assert (
        "annualized_volatility_pct"
        in summary
    )

    assert isinstance(
        summary["sharpe_ratio"],
        float,
    )


def test_invalid_cost_parameters_are_rejected(
    tmp_path,
):
    path = create_test_csv(
        tmp_path,
        prices=[
            100.0,
        ],
        signals=[
            0,
        ],
    )

    with pytest.raises(
        ValueError
    ):
        run_backtest(
            path,
            transaction_fee_pct=-1,
        )

    with pytest.raises(
        ValueError
    ):
        run_backtest(
            path,
            slippage_pct=6,
        )
