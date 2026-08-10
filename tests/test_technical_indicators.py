import pytest

from src.technical_indicators import (
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    calculate_technical_snapshot,
    simple_moving_average,
)


def test_simple_moving_average():
    result = simple_moving_average(
        [1, 2, 3, 4, 5],
        3,
    )

    assert result == pytest.approx(
        4.0
    )


def test_rsi_strong_uptrend():
    closes = [
        float(value)
        for value in range(
            100,
            120,
        )
    ]

    result = calculate_rsi(
        closes,
        14,
    )

    assert result == 100.0


def test_rsi_flat_market_is_neutral():
    closes = [
        100.0
        for _ in range(30)
    ]

    result = calculate_rsi(
        closes,
        14,
    )

    assert result == 50.0


def test_macd_is_available_with_enough_data():
    closes = [
        100.0
        + index
        for index in range(60)
    ]

    result = calculate_macd(
        closes
    )

    assert result["macd"] is not None
    assert result["signal"] is not None
    assert result["histogram"] is not None

    assert (
        result["macd"]
        > 0
    )


def test_bollinger_bands():
    closes = [
        float(value)
        for value in range(
            100,
            120,
        )
    ]

    result = (
        calculate_bollinger_bands(
            closes,
            period=20,
        )
    )

    assert (
        result["upper"]
        > result["middle"]
    )

    assert (
        result["lower"]
        < result["middle"]
    )

    assert (
        result["bandwidth_pct"]
        > 0
    )


def test_technical_snapshot_warms_up():
    result = (
        calculate_technical_snapshot(
            [100.0] * 20
        )
    )

    assert (
        result["ready"]
        is False
    )

    assert (
        result["rating"]
        == "WARMING_UP"
    )


def test_technical_snapshot_returns_score():
    closes = [
        100.0
        + index * 0.5
        for index in range(80)
    ]

    result = (
        calculate_technical_snapshot(
            closes
        )
    )

    assert (
        result["ready"]
        is True
    )

    assert (
        -4
        <= result["score"]
        <= 4
    )

    assert result["rating"] in {
        "STRONG_BUY",
        "BUY",
        "NEUTRAL",
        "SELL",
        "STRONG_SELL",
    }

    assert set(
        result["components"]
    ) == {
        "sma_trend",
        "rsi",
        "macd",
        "bollinger",
    }


def test_invalid_prices_are_rejected():
    with pytest.raises(
        ValueError
    ):
        calculate_technical_snapshot(
            [
                100.0,
                0.0,
                101.0,
            ]
        )
