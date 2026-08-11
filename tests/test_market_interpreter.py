from src.market_interpreter import (
    MarketInterpreter,
)


def make_technical(
    *,
    score=0,
    rsi=48.0,
    macd_histogram=-1.5,
    sma20=101.0,
    sma50=100.0,
):
    return {
        "ready": True,
        "price": 100.5,
        "sma20": sma20,
        "sma50": sma50,
        "rsi14": rsi,

        "macd": {
            "macd":
                macd_histogram,

            "signal":
                0.0,

            "histogram":
                macd_histogram,
        },

        "bollinger": {
            "upper": 103.0,
            "middle": 100.0,
            "lower": 97.0,
            "bandwidth_pct": 0.8,
        },

        "score": score,

        "components": {
            "sma_trend":
                1
                if sma20 > sma50
                else -1,

            "rsi":
                -1
                if rsi >= 70
                else 1
                if rsi <= 30
                else 0,

            "macd":
                1
                if macd_histogram > 0
                else -1
                if macd_histogram < 0
                else 0,

            "bollinger":
                0,
        },
    }


def test_mixed_market_detects_conflict():
    interpreter = (
        MarketInterpreter()
    )

    result = interpreter.interpret(
        make_technical(
            score=0,
            sma20=101.0,
            sma50=100.0,
            macd_histogram=-1.5,
        )
    )

    assert (
        result["state"]
        == "NEUTRAL"
    )

    assert (
        result["trend"][
            "direction"
        ]
        == "BULLISH"
    )

    assert (
        result["momentum"][
            "direction"
        ]
        == "BEARISH"
    )

    assert result["conflicts"]


def test_neutral_rsi_is_low_relevance():
    interpreter = (
        MarketInterpreter()
    )

    result = interpreter.interpret(
        make_technical(
            rsi=50.0
        )
    )

    assert (
        result["rsi"][
            "importance"
        ]
        == "LOW"
    )

    low_names = {
        item["name"]
        for item in result[
            "low_relevance"
        ]
    }

    assert "RSI" in low_names


def test_extreme_rsi_becomes_important():
    interpreter = (
        MarketInterpreter()
    )

    result = interpreter.interpret(
        make_technical(
            score=-1,
            rsi=78.0,
        )
    )

    assert (
        result["rsi"]["state"]
        == "OVERBOUGHT"
    )

    assert (
        result["rsi"][
            "importance"
        ]
        == "HIGH"
    )


def test_warming_up_state():
    interpreter = (
        MarketInterpreter()
    )

    result = interpreter.interpret(
        {
            "ready": False
        }
    )

    assert (
        result["ready"]
        is False
    )

    assert (
        result["state"]
        == "WARMING_UP"
    )

    assert (
        result["confidence"]
        == 0
    )
