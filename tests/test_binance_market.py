import json

import pytest

from src.binance_market import (
    normalize_symbol,
    parse_kline_message,
    parse_24h_ticker,
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


def test_parse_24h_ticker():
    payload = {
        "symbol": "BTCUSDT",
        "priceChange": "1500.50",
        "priceChangePercent": "2.35",
        "weightedAvgPrice": "65000.00",
        "prevClosePrice": "64000.00",
        "lastPrice": "65500.50",
        "lastQty": "0.1",
        "bidPrice": "65500.40",
        "askPrice": "65500.50",
        "openPrice": "64000.00",
        "highPrice": "66000.00",
        "lowPrice": "63500.00",
        "volume": "12345.67",
        "quoteVolume": "800000000.00",
        "openTime": 1786300000000,
        "closeTime": 1786386400000,
        "firstId": 1,
        "lastId": 100,
        "count": 100,
    }

    result = parse_24h_ticker(
        payload
    )

    assert result["symbol"] == "BTCUSDT"

    assert (
        result["last_price"]
        == 65500.50
    )

    assert (
        result["price_change_pct"]
        == 2.35
    )

    assert (
        result["high_price"]
        == 66000.00
    )

    assert (
        result["low_price"]
        == 63500.00
    )

    assert (
        result["trade_count"]
        == 100
    )
