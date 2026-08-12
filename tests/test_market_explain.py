import asyncio
import os

import pytest
from fastapi import HTTPException


os.environ.setdefault(
    "SECRET_KEY",
    (
        "test-only-secret-key-"
        "for-market-explain-tests-"
        "0123456789abcdef"
    ),
)


import src.api as api


def test_market_explain_pipeline(
    monkeypatch,
):
    candles = [
        {
            "close": 100.0,
        }
        for _ in range(150)
    ]

    monkeypatch.setattr(
        api,
        "get_klines",
        lambda symbol, interval, limit:
            candles,
    )

    class FakeLiveEngine:
        def __init__(
            self,
            short_window,
            long_window,
        ):
            pass

        def seed(
            self,
            seed_candles,
        ):
            return {
                "signal": "HOLD",
                "crossover": "HOLD",
            }

    monkeypatch.setattr(
        api,
        "LiveSMAEngine",
        FakeLiveEngine,
    )

    technical = {
        "ready": True,
        "price": 100.0,
    }

    monkeypatch.setattr(
        api,
        "calculate_technical_snapshot",
        lambda closes:
            technical,
    )

    interpretation = {
        "ready": True,
        "state": "NEUTRAL",
    }

    class FakeInterpreter:
        def interpret(
            self,
            technical_snapshot,
            live_snapshot,
        ):
            return interpretation

    monkeypatch.setattr(
        api,
        "MarketInterpreter",
        FakeInterpreter,
    )

    analysis = {
        "available": True,
        "source": "ollama",
        "model": "qwen3:4b",
        "summary": "Test summary",
        "explanation": "Test explanation",
        "educational_note": "Test note",
        "important": [],
        "low_relevance": [],
    }

    class FakeAnalyst:
        def analyze(
            self,
            market_interpretation,
        ):
            return analysis

    monkeypatch.setattr(
        api,
        "LocalLLMAnalyst",
        FakeAnalyst,
    )

    result = asyncio.run(
        api.explain_market(
            symbol="BTCUSDT",
            interval="1m",
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
        result["interpretation"]
        == interpretation
    )

    assert (
        result["analysis"]["source"]
        == "ollama"
    )


def test_market_explain_requires_enough_data(
    monkeypatch,
):
    monkeypatch.setattr(
        api,
        "get_klines",
        lambda symbol, interval, limit:
            [],
    )

    with pytest.raises(
        HTTPException
    ) as exc:
        asyncio.run(
            api.explain_market(
                symbol="BTCUSDT",
                interval="1m",
            )
        )

    assert (
        exc.value.status_code
        == 503
    )
