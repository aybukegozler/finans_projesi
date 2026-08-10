import pytest

from src.live_signal import (
    LiveSMAEngine,
)


def candle(
    number: int,
    close: float,
    closed: bool = True,
) -> dict:
    return {
        "open_time_ms":
            1_000_000
            + number * 60_000,

        "close":
            close,

        "closed":
            closed,
    }


def test_engine_warms_up():
    engine = LiveSMAEngine(
        short_window=2,
        long_window=3,
        max_points=10,
    )

    result = engine.seed(
        [
            candle(1, 100),
            candle(2, 101),
        ]
    )

    assert (
        result["ready"]
        is False
    )

    assert (
        result["signal"]
        == "HOLD"
    )

    assert (
        result["trend"]
        == "WARMING_UP"
    )


def test_engine_calculates_sma():
    engine = LiveSMAEngine(
        short_window=2,
        long_window=3,
        max_points=10,
    )

    result = engine.seed(
        [
            candle(1, 100),
            candle(2, 100),
            candle(3, 110),
        ]
    )

    assert (
        result["ready"]
        is True
    )

    assert (
        result["sma_short"]
        == pytest.approx(
            105.0
        )
    )

    assert (
        result["sma_long"]
        == pytest.approx(
            103.33333333
        )
    )

    assert (
        result["trend"]
        == "BULLISH"
    )

    assert (
        result["signal"]
        == "BUY"
    )


def test_bullish_crossover_is_detected():
    engine = LiveSMAEngine(
        short_window=2,
        long_window=3,
        max_points=10,
    )

    engine.seed(
        [
            candle(1, 100),
            candle(2, 100),
            candle(3, 100),
        ]
    )

    result = engine.update(
        candle(4, 110)
    )

    assert (
        result["crossover"]
        == "BUY"
    )

    assert (
        result["trend"]
        == "BULLISH"
    )

    assert (
        result["last_crossover"][
            "signal"
        ]
        == "BUY"
    )


def test_bearish_crossover_is_detected():
    engine = LiveSMAEngine(
        short_window=2,
        long_window=3,
        max_points=10,
    )

    engine.seed(
        [
            candle(1, 100),
            candle(2, 100),
            candle(3, 100),
        ]
    )

    result = engine.update(
        candle(4, 90)
    )

    assert (
        result["crossover"]
        == "SELL"
    )

    assert (
        result["trend"]
        == "BEARISH"
    )


def test_same_live_candle_is_replaced():
    engine = LiveSMAEngine(
        short_window=2,
        long_window=3,
        max_points=10,
    )

    engine.seed(
        [
            candle(1, 100),
            candle(2, 100),
            candle(
                3,
                101,
                closed=False,
            ),
        ]
    )

    first_count = (
        len(engine.candles)
    )

    result = engine.update(
        candle(
            3,
            105,
            closed=False,
        )
    )

    assert (
        len(engine.candles)
        == first_count
    )

    assert (
        result["candle_count"]
        == 3
    )

    assert (
        engine.candles[-1][
            "close"
        ]
        == 105.0
    )


def test_new_candle_is_appended():
    engine = LiveSMAEngine(
        short_window=2,
        long_window=3,
        max_points=10,
    )

    engine.seed(
        [
            candle(1, 100),
            candle(2, 101),
            candle(3, 102),
        ]
    )

    engine.update(
        candle(4, 103)
    )

    assert (
        len(engine.candles)
        == 4
    )


def test_invalid_windows_are_rejected():
    with pytest.raises(
        ValueError
    ):
        LiveSMAEngine(
            short_window=1,
            long_window=50,
        )

    with pytest.raises(
        ValueError
    ):
        LiveSMAEngine(
            short_window=50,
            long_window=20,
        )


def test_unclosed_crossover_is_not_confirmed():
    engine = LiveSMAEngine(
        short_window=2,
        long_window=3,
        max_points=10,
    )

    engine.seed(
        [
            candle(1, 100),
            candle(2, 100),
            candle(3, 100),
        ]
    )

    result = engine.update(
        candle(
            4,
            110,
            closed=False,
        )
    )

    assert result["crossover"] == "BUY"

    assert (
        result["last_crossover"]
        is None
    )
