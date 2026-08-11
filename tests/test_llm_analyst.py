import json
from unittest.mock import patch

from src.llm_analyst import (
    LocalLLMAnalyst,
)


def make_interpretation():
    return {
        "ready": True,
        "state": "NEUTRAL",
        "confidence": 42,

        "headline":
            "Signals are mixed.",

        "trend": {
            "direction": "BEARISH",
            "importance": "LOW",
        },

        "momentum": {
            "direction": "BULLISH",
            "importance": "HIGH",
        },

        "rsi": {
            "value": 47.0,
            "state": "NEUTRAL",
            "importance": "LOW",
        },

        "volatility": {
            "state": "NORMAL",
        },

        "important": [
            {
                "name":
                    "MACD Momentum",

                "direction":
                    "BULLISH",

                "importance":
                    "HIGH",
            }
        ],

        "low_relevance": [
            {
                "name":
                    "SMA Trend",

                "direction":
                    "BEARISH",

                "importance":
                    "LOW",
            },

            {
                "name":
                    "RSI",

                "direction":
                    "NEUTRAL",

                "importance":
                    "LOW",
            },
        ],

        "conflicts": [
            (
                "Bearish SMA trend conflicts "
                "with bullish MACD momentum."
            )
        ],

        "technical_score": 0,

        "explanation":
            (
                "The SMA trend is bearish. "
                "MACD momentum is bullish."
            ),
    }


class FakeResponse:
    def __init__(
        self,
        payload,
    ):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False

    def read(self):
        return json.dumps(
            self.payload
        ).encode(
            "utf-8"
        )


def make_llm_content():
    return {
        "summary":
            (
                "Teknik görünüm şu anda "
                "karışık."
            ),

        "explanation":
            (
                "MACD momentumu yukarı yönlü "
                "olsa da SMA trendi aşağı yönlü. "
                "Bu iki gösterge aynı yönü "
                "teyit etmiyor."
            ),

        "educational_note":
            (
                "Bu açıklama yalnızca mevcut "
                "teknik göstergeleri özetler."
            ),
    }


def test_valid_structured_response():
    analyst = (
        LocalLLMAnalyst()
    )

    response = {
        "message": {
            "role": "assistant",

            "content":
                json.dumps(
                    make_llm_content(),
                    ensure_ascii=False,
                ),
        }
    }

    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            response
        ),
    ):
        result = analyst.analyze(
            make_interpretation()
        )

    assert result["available"] is True
    assert result["source"] == "ollama"

    assert (
        result["important"][0]["name"]
        == "MACD Momentum"
    )

    assert {
        item["name"]
        for item
        in result["low_relevance"]
    } == {
        "SMA Trend",
        "RSI",
    }


def test_financial_advice_is_blocked():
    analyst = (
        LocalLLMAnalyst()
    )

    content = make_llm_content()

    content["educational_note"] = (
        "Bu durumda dikkatli hareket etmelisin."
    )

    response = {
        "message": {
            "role": "assistant",

            "content":
                json.dumps(
                    content,
                    ensure_ascii=False,
                ),
        }
    }

    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            response
        ),
    ):
        result = analyst.analyze(
            make_interpretation()
        )

    assert result["available"] is False

    assert (
        result["reason"]
        == "financial_advice_blocked"
    )


def test_warming_up_uses_fallback():
    analyst = (
        LocalLLMAnalyst()
    )

    result = analyst.analyze(
        {
            "ready": False,
            "headline":
                "Not enough data.",
            "explanation":
                "Collecting candles.",
        }
    )

    assert result["available"] is False

    assert (
        result["reason"]
        == "market_not_ready"
    )


def test_markdown_wrapped_json_is_recovered():
    analyst = (
        LocalLLMAnalyst()
    )

    wrapped = (
        "```json\\n"
        + json.dumps(
            make_llm_content(),
            ensure_ascii=False,
        )
        + "\\n```"
    )

    response = {
        "message": {
            "role": "assistant",
            "content": wrapped,
        }
    }

    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            response
        ),
    ):
        result = analyst.analyze(
            make_interpretation()
        )

    assert result["available"] is True


def test_json_surrounded_by_text_is_recovered():
    analyst = (
        LocalLLMAnalyst()
    )

    wrapped = (
        "İstenen çıktı:\\n"
        + json.dumps(
            make_llm_content(),
            ensure_ascii=False,
        )
    )

    response = {
        "message": {
            "role": "assistant",
            "content": wrapped,
        }
    }

    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            response
        ),
    ):
        result = analyst.analyze(
            make_interpretation()
        )

    assert result["available"] is True


def test_buy_opportunity_language_is_blocked():
    analyst = (
        LocalLLMAnalyst()
    )

    content = make_llm_content()

    content["explanation"] = (
        "Bu durumda kısa vadeli "
        "alım fırsatı olabilir."
    )

    response = {
        "message": {
            "role": "assistant",
            "content":
                json.dumps(
                    content,
                    ensure_ascii=False,
                ),
        }
    }

    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            response
        ),
    ):
        result = analyst.analyze(
            make_interpretation()
        )

    assert result["available"] is False

    assert (
        result["reason"]
        == "financial_advice_blocked"
    )


def test_position_language_is_blocked():
    analyst = (
        LocalLLMAnalyst()
    )

    content = make_llm_content()

    content["summary"] = (
        "Pozisyon açmak için uygun olabilir."
    )

    response = {
        "message": {
            "role": "assistant",
            "content":
                json.dumps(
                    content,
                    ensure_ascii=False,
                ),
        }
    }

    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            response
        ),
    ):
        result = analyst.analyze(
            make_interpretation()
        )

    assert result["available"] is False

    assert (
        result["reason"]
        == "financial_advice_blocked"
    )


def test_fallback_factor_shape_matches_success_shape():
    analyst = (
        LocalLLMAnalyst()
    )

    result = analyst._fallback(
        make_interpretation(),
        "test_fallback",
    )

    assert (
        isinstance(
            result["important"],
            list,
        )
    )

    assert (
        isinstance(
            result["low_relevance"],
            list,
        )
    )

    assert (
        isinstance(
            result["important"][0],
            dict,
        )
    )

    assert {
        "name",
        "direction",
        "importance",
    }.issubset(
        result["important"][0]
    )


def test_investment_disclaimer_is_not_blocked():
    analyst = (
        LocalLLMAnalyst()
    )

    content = make_llm_content()

    content["educational_note"] = (
        "Bu açıklama yatırım tavsiyesi değildir "
        "ve yalnızca mevcut teknik verileri açıklar."
    )

    response = {
        "message": {
            "role": "assistant",
            "content":
                json.dumps(
                    content,
                    ensure_ascii=False,
                ),
        }
    }

    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            response
        ),
    ):
        result = analyst.analyze(
            make_interpretation()
        )

    assert result["available"] is True

    assert (
        result["source"]
        == "ollama"
    )


def test_explicit_investment_instruction_is_blocked():
    analyst = (
        LocalLLMAnalyst()
    )

    content = make_llm_content()

    content["explanation"] = (
        "Bu koşullarda yatırım yapmanız uygun olabilir."
    )

    response = {
        "message": {
            "role": "assistant",
            "content":
                json.dumps(
                    content,
                    ensure_ascii=False,
                ),
        }
    }

    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            response
        ),
    ):
        result = analyst.analyze(
            make_interpretation()
        )

    assert result["available"] is False

    assert (
        result["reason"]
        == "financial_advice_blocked"
    )
