import json

import pytest

from src.binance_market import (
    normalize_symbol,
    parse_kline_message,
    validate_interval,
)


def test_normalize_binance_symbol():
    assert (
        normalize_symbol(
            "btc/usdt"
        )
        == "BTCUSDT"
    )

    assert (
        normalize_symbol(
            "ETH-USDT"
        )
        == "ETHUSDT"
    )


def test_invalid_symbol_is_rejected():
    with pytest.raises(
        ValueError
    ):
        normalize_symbol(
            "BTC_USDT!"
        )


def test_supported_interval():
    assert (
        validate_interval("1m")
        == "1m"
    )

    assert (
        validate_interval("1h")
        == "1h"
    )


def test_invalid_interval():
    with pytest.raises(
        ValueError
    ):
        validate_interval(
            "7minutes"
        )


def test_parse_binance_kline():
    message = json.dumps(
        {
            "e": "kline",
            "E": 1770000000000,
            "s": "BTCUSDT",
            "k": {
                "t": 1770000000000,
                "T": 1770000059999,
                "s": "BTCUSDT",
                "i": "1m",
                "o": "100.00",
                "c": "105.00",
                "h": "110.00",
                "l": "95.00",
                "v": "12.5",
                "n": 42,
                "x": False,
                "q": "1280.50",
            },
        }
    )

    result = (
        parse_kline_message(
            message
        )
    )

    assert (
        result["symbol"]
        == "BTCUSDT"
    )

    assert (
        result["interval"]
        == "1m"
    )

    assert (
        result["close"]
        == 105.0
    )

    assert (
        result["high"]
        == 110.0
    )

    assert (
        result["volume"]
        == 12.5
    )

    assert (
        result["trade_count"]
        == 42
    )

    assert (
        result["closed"]
        is False
    )
