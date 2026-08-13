import asyncio
import os

import pytest
from fastapi import HTTPException


os.environ.setdefault(
    "SECRET_KEY",
    (
        "test-only-market-change-"
        "endpoint-secret-"
        "0123456789abcdef"
    ),
)


import src.api as api


def previous_snapshot():
    return {
        "ready": True,
        "state": "NEUTRAL",
        "confidence": 42,

        "trend": {
            "direction": "BEARISH",
        },

        "momentum": {
            "direction": "BEARISH",
        },

        "rsi": {
            "state": "NEUTRAL",
        },

        "volatility": {
            "state": "LOW",
        },

        "bollinger": {
            "position": "MID_RANGE",
        },

        "technical_score": -1,
        "live_signal": "SELL",

        "important": [
            {
                "name": "SMA Trend",
            },
            {
                "name": "MACD Momentum",
            },
        ],

        "low_relevance": [
            {
                "name": "RSI",
            },
        ],

        "conflicts": [],
    }


def current_snapshot():
    return {
        "ready": True,
        "state": "NEUTRAL",
        "confidence": 58,

        "trend": {
            "direction": "BEARISH",
        },

        "momentum": {
            "direction": "BULLISH",
        },

        "rsi": {
            "state": "NEUTRAL",
        },

        "volatility": {
            "state": "LOW",
        },

        "bollinger": {
            "position": "MID_RANGE",
        },

        "technical_score": 1,
        "live_signal": "SELL",

        "important": [
            {
                "name": "SMA Trend",
            },
            {
                "name": "MACD Momentum",
            },
        ],

        "low_relevance": [
            {
                "name": "RSI",
            },
        ],

        "conflicts": [
            "Trend and momentum disagree."
        ],
    }


def test_explain_market_change(
    monkeypatch,
):
    candles = [
        {
            "close": float(
                100 + i
            )
        }
        for i in range(60)
    ]

    monkeypatch.setattr(
        api,
        "get_klines",
        lambda *args, **kwargs: candles,
    )

    class FakeLiveEngine:
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            pass

        def seed(
            self,
            data,
        ):
            return {
                "ready": True,
            }

    monkeypatch.setattr(
        api,
        "LiveSMAEngine",
        FakeLiveEngine,
    )

    monkeypatch.setattr(
        api,
        "calculate_technical_snapshot",
        lambda closes: {
            "ready": True,
        },
    )

    class FakeInterpreter:
        def interpret(
            self,
            technical,
            live,
        ):
            return current_snapshot()

    monkeypatch.setattr(
        api,
        "MarketInterpreter",
        FakeInterpreter,
    )

    class FakeChangeAnalyst:
        def analyze(
            self,
            change,
            mode="simple",
        ):
            return {
                "available": True,
                "source": "ollama",
                "model": "qwen3:4b",
                "reason": None,
                "summary":
                    "Momentum yön değiştirdi.",
                "explanation":
                    (
                        "MACD aşağı yönlüden "
                        "yukarı yönlüye geçti."
                    ),
                "educational_note":
                    (
                        "Tek gösterge trend "
                        "dönüşünü doğrulamaz."
                    ),
            }

    monkeypatch.setattr(
        api,
        "LocalChangeAnalyst",
        FakeChangeAnalyst,
    )

    result = asyncio.run(
        api.explain_market_change(
            previous=previous_snapshot(),
            symbol="BTCUSDT",
            interval="1m",
            mode="technical",
        )
    )

    assert result["symbol"] == "BTCUSDT"
    assert result["interval"] == "1m"
    assert result["mode"] == "technical"

    assert (
        result["change"]["meaningful"]
        is True
    )

    assert (
        result[
            "change"
        ]["confidence"]["delta"]
        == 16
    )

    labels = {
        item["label"]
        for item
        in result["change"]["changes"]
    }

    assert "MACD Momentum" in labels

    assert (
        result["analysis"]["source"]
        == "ollama"
    )

    assert (
        result["analysis"]["summary"]
        == "Momentum yön değiştirdi."
    )


def test_change_requires_previous_ready():
    with pytest.raises(
        HTTPException
    ) as exc:
        asyncio.run(
            api.explain_market_change(
                previous={
                    "ready": False,
                },
                symbol="BTCUSDT",
                interval="1m",
            )
        )

    assert (
        exc.value.status_code
        == 400
    )
