import io
import json

from src.llm_change_analyst import (
    LocalChangeAnalyst,
)


def meaningful_change():
    return {
        "ready": True,
        "meaningful": True,

        "headline":
            "Bazı teknik bileşenler "
            "yön değiştirdi.",

        "previous_state":
            "NEUTRAL",

        "current_state":
            "NEUTRAL",

        "changes": [
            {
                "category":
                    "momentum",

                "label":
                    "MACD Momentum",

                "before":
                    "BEARISH",

                "after":
                    "BULLISH",
            }
        ],

        "confidence": {
            "previous": 42,
            "current": 58,
            "delta": 16,
        },

        "technical_score": {
            "previous": -1,
            "current": 1,
            "delta": 2,
        },

        "importance_changes": {
            "became_important": [],
            "became_low_relevance": [],
        },

        "conflicts": {
            "added": [
                (
                    "Trend and momentum "
                    "disagree."
                )
            ],
            "resolved": [],
        },
    }


class FakeResponse:
    def __init__(
        self,
        payload,
    ):
        self.payload = payload

    def __enter__(
        self,
    ):
        return self

    def __exit__(
        self,
        *args,
    ):
        return False

    def read(
        self,
    ):
        return json.dumps(
            self.payload
        ).encode(
            "utf-8"
        )


def test_change_analyst_success(
    monkeypatch,
):
    result_payload = {
        "summary":
            "Momentum yön değiştirdi.",

        "explanation":
            (
                "MACD önceki analize göre "
                "aşağı yönlüden yukarı "
                "yönlüye geçti."
            ),

        "educational_note":
            (
                "Tek bir gösterge trend "
                "dönüşünü doğrulamaz."
            ),
    }

    monkeypatch.setattr(
        "src.llm_change_analyst.request.urlopen",
        lambda *args, **kwargs:
            FakeResponse(
                {
                    "message": {
                        "content":
                            json.dumps(
                                result_payload
                            )
                    }
                }
            ),
    )

    analyst = LocalChangeAnalyst()

    result = analyst.analyze(
        meaningful_change()
    )

    assert result["available"] is True
    assert result["source"] == "ollama"
    assert result["model"] == "qwen3:4b"

    assert (
        result["summary"]
        == "Momentum yön değiştirdi."
    )


def test_no_change_skips_ollama(
    monkeypatch,
):
    called = False

    def fake_urlopen(
        *args,
        **kwargs,
    ):
        nonlocal called
        called = True

        return io.BytesIO()

    monkeypatch.setattr(
        "src.llm_change_analyst.request.urlopen",
        fake_urlopen,
    )

    analyst = LocalChangeAnalyst()

    result = analyst.analyze(
        {
            "ready": True,
            "meaningful": False,
            "headline":
                (
                    "Son analizden beri "
                    "belirgin bir yapısal "
                    "değişim yok."
                ),
        }
    )

    assert called is False

    assert (
        result["source"]
        == "deterministic"
    )


def test_change_advice_is_blocked(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.llm_change_analyst.request.urlopen",
        lambda *args, **kwargs:
            FakeResponse(
                {
                    "message": {
                        "content":
                            json.dumps(
                                {
                                    "summary":
                                        (
                                            "Şimdi alım "
                                            "fırsatı var."
                                        ),

                                    "explanation":
                                        "Momentum değişti.",

                                    "educational_note":
                                        "Teknik analiz.",
                                }
                            )
                    }
                }
            ),
    )

    analyst = LocalChangeAnalyst()

    result = analyst.analyze(
        meaningful_change()
    )

    assert (
        result["source"]
        == "deterministic"
    )

    assert (
        result["reason"]
        == "financial_advice_blocked"
    )


def test_invalid_change_uses_fallback():
    analyst = LocalChangeAnalyst()

    result = analyst.analyze(
        {
            "ready": False,
            "meaningful": False,
        }
    )

    assert result["available"] is False

    assert (
        result["reason"]
        == "change_not_ready"
    )
