from src.market_change import (
    MarketChangeDetector,
)


def make_snapshot(
    *,
    state="NEUTRAL",
    trend="BEARISH",
    momentum="BULLISH",
    rsi="NEUTRAL",
    volatility="LOW",
    bollinger="MID_RANGE",
    confidence=42,
    technical_score=0,
    important=None,
    low=None,
    conflicts=None,
):
    return {
        "ready": True,
        "state": state,

        "confidence":
            confidence,

        "trend": {
            "direction": trend,
        },

        "momentum": {
            "direction": momentum,
        },

        "rsi": {
            "state": rsi,
        },

        "volatility": {
            "state": volatility,
        },

        "bollinger": {
            "position": bollinger,
        },

        "technical_score":
            technical_score,

        "live_signal":
            "SELL",

        "important":
            important
            or [
                {
                    "name":
                        "SMA Trend",
                },
                {
                    "name":
                        "MACD Momentum",
                },
            ],

        "low_relevance":
            low
            or [
                {
                    "name":
                        "RSI",
                },
                {
                    "name":
                        "Bollinger Position",
                },
            ],

        "conflicts":
            conflicts
            or [],
    }


def test_no_meaningful_change():
    detector = (
        MarketChangeDetector()
    )

    snapshot = make_snapshot()

    result = detector.compare(
        snapshot,
        snapshot,
    )

    assert result["ready"] is True
    assert result["meaningful"] is False
    assert result["changes"] == []

    assert (
        result["headline"]
        == (
            "Son analizden beri belirgin "
            "bir yapısal değişim yok."
        )
    )


def test_detects_direction_changes():
    detector = (
        MarketChangeDetector()
    )

    previous = make_snapshot()

    current = make_snapshot(
        trend="BULLISH",
        momentum="BEARISH",
    )

    result = detector.compare(
        previous,
        current,
    )

    assert result["meaningful"] is True

    labels = {
        item["label"]
        for item in result["changes"]
    }

    assert "SMA Trend" in labels
    assert "MACD Momentum" in labels


def test_detects_market_state_change():
    detector = (
        MarketChangeDetector()
    )

    previous = make_snapshot(
        state="NEUTRAL"
    )

    current = make_snapshot(
        state="BUY"
    )

    result = detector.compare(
        previous,
        current,
    )

    assert result["meaningful"] is True

    assert (
        result["headline"]
        == "Genel piyasa durumu değişti."
    )


def test_detects_score_changes():
    detector = (
        MarketChangeDetector()
    )

    previous = make_snapshot(
        confidence=42,
        technical_score=0,
    )

    current = make_snapshot(
        confidence=58,
        technical_score=2,
    )

    result = detector.compare(
        previous,
        current,
    )

    assert (
        result["confidence"]["delta"]
        == 16
    )

    assert (
        result[
            "technical_score"
        ]["delta"]
        == 2
    )

    assert result["meaningful"] is True


def test_detects_importance_changes():
    detector = (
        MarketChangeDetector()
    )

    previous = make_snapshot()

    current = make_snapshot(
        important=[
            {
                "name":
                    "SMA Trend",
            },
            {
                "name":
                    "RSI",
            },
        ],
        low=[
            {
                "name":
                    "MACD Momentum",
            },
            {
                "name":
                    "Bollinger Position",
            },
        ],
    )

    result = detector.compare(
        previous,
        current,
    )

    importance = (
        result[
            "importance_changes"
        ]
    )

    assert (
        "RSI"
        in importance[
            "became_important"
        ]
    )

    assert (
        "MACD Momentum"
        in importance[
            "became_low_relevance"
        ]
    )


def test_detects_conflict_changes():
    detector = (
        MarketChangeDetector()
    )

    previous = make_snapshot(
        conflicts=[
            "Trend and momentum disagree."
        ]
    )

    current = make_snapshot(
        conflicts=[
            "RSI conflicts with trend."
        ]
    )

    result = detector.compare(
        previous,
        current,
    )

    assert (
        "RSI conflicts with trend."
        in result["conflicts"]["added"]
    )

    assert (
        "Trend and momentum disagree."
        in result["conflicts"]["resolved"]
    )


def test_requires_two_ready_snapshots():
    detector = (
        MarketChangeDetector()
    )

    result = detector.compare(
        {
            "ready": False,
        },
        make_snapshot(),
    )

    assert result["ready"] is False
    assert result["meaningful"] is False
