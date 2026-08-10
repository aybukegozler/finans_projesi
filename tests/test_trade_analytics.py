from pathlib import Path

import pandas as pd
import pytest

from src.trade_analytics import (
    analyze_trades,
)


def create_signal_csv(
    tmp_path: Path,
    prices: list[float],
    signals: list[int],
) -> Path:
    path = (
        tmp_path
        / "signals.csv"
    )

    dataframe = pd.DataFrame(
        {
            "Date": pd.date_range(
                "2026-01-01",
                periods=len(prices),
                freq="D",
            ).strftime("%Y-%m-%d"),
            "Close": prices,
            "Signal": signals,
        }
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    return path


def test_profitable_trade_details(
    tmp_path,
):
    path = create_signal_csv(
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

    result = analyze_trades(
        path,
        initial_capital=10_000,
        transaction_fee_pct=0,
        slippage_pct=0,
    )

    assert (
        result["summary"][
            "closed_trades"
        ]
        == 1
    )

    trade = result["trades"][0]

    assert (
        trade["entry_date"]
        == "2026-01-01"
    )

    assert (
        trade["exit_date"]
        == "2026-01-03"
    )

    assert (
        trade["holding_days"]
        == 2
    )

    assert (
        trade["gross_pnl"]
        == 2000.0
    )

    assert (
        trade["net_pnl"]
        == 2000.0
    )

    assert (
        trade["return_pct"]
        == 20.0
    )

    assert trade["result"] == "WIN"


def test_costs_reduce_net_pnl(
    tmp_path,
):
    path = create_signal_csv(
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

    no_cost = analyze_trades(
        path,
        transaction_fee_pct=0,
        slippage_pct=0,
    )

    with_cost = analyze_trades(
        path,
        transaction_fee_pct=1,
        slippage_pct=1,
    )

    free_trade = (
        no_cost["trades"][0]
    )

    costly_trade = (
        with_cost["trades"][0]
    )

    assert (
        costly_trade["net_pnl"]
        < free_trade["net_pnl"]
    )

    assert (
        costly_trade["total_fees"]
        > 0
    )

    assert (
        costly_trade["slippage_cost"]
        > 0
    )

    assert (
        costly_trade["total_costs"]
        > 0
    )


def test_open_position_is_reported(
    tmp_path,
):
    path = create_signal_csv(
        tmp_path,
        prices=[
            100.0,
            110.0,
            120.0,
        ],
        signals=[
            1,
            0,
            0,
        ],
    )

    result = analyze_trades(
        path,
        transaction_fee_pct=0,
        slippage_pct=0,
    )

    assert (
        result["summary"][
            "closed_trades"
        ]
        == 0
    )

    assert (
        result["summary"][
            "has_open_position"
        ]
        is True
    )

    assert (
        result["open_position"]
        is not None
    )

    assert (
        result["open_position"][
            "unrealized_pnl"
        ]
        == 2000.0
    )


def test_force_close_converts_open_position_to_trade(
    tmp_path,
):
    path = create_signal_csv(
        tmp_path,
        prices=[
            100.0,
            120.0,
        ],
        signals=[
            1,
            0,
        ],
    )

    result = analyze_trades(
        path,
        transaction_fee_pct=0,
        slippage_pct=0,
        force_close_at_end=True,
    )

    assert (
        result["summary"][
            "closed_trades"
        ]
        == 1
    )

    assert (
        result["summary"][
            "has_open_position"
        ]
        is False
    )

    assert (
        result["trades"][0][
            "forced_exit"
        ]
        is True
    )

    assert (
        result["trades"][0][
            "net_pnl"
        ]
        == 2000.0
    )


def test_invalid_cost_is_rejected(
    tmp_path,
):
    path = create_signal_csv(
        tmp_path,
        prices=[100.0],
        signals=[0],
    )

    with pytest.raises(ValueError):
        analyze_trades(
            path,
            transaction_fee_pct=-1,
        )

    with pytest.raises(ValueError):
        analyze_trades(
            path,
            slippage_pct=10,
        )
