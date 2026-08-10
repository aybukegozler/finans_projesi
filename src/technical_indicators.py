from __future__ import annotations

from math import sqrt


def simple_moving_average(
    values: list[float],
    window: int,
) -> float | None:
    if window <= 0:
        raise ValueError(
            "window must be greater than zero."
        )

    if len(values) < window:
        return None

    return (
        sum(values[-window:])
        / window
    )


def exponential_moving_average_series(
    values: list[float],
    period: int,
) -> list[float | None]:
    if period <= 0:
        raise ValueError(
            "period must be greater than zero."
        )

    if not values:
        return []

    result: list[float | None] = [
        None
    ] * len(values)

    if len(values) < period:
        return result

    initial_sma = (
        sum(values[:period])
        / period
    )

    result[
        period - 1
    ] = initial_sma

    multiplier = (
        2.0
        / (period + 1.0)
    )

    previous_ema = initial_sma

    for index in range(
        period,
        len(values),
    ):
        current_ema = (
            (
                values[index]
                - previous_ema
            )
            * multiplier
            + previous_ema
        )

        result[index] = (
            current_ema
        )

        previous_ema = (
            current_ema
        )

    return result


def calculate_rsi(
    closes: list[float],
    period: int = 14,
) -> float | None:
    if period <= 0:
        raise ValueError(
            "period must be greater than zero."
        )

    if len(closes) < period + 1:
        return None

    gains: list[float] = []
    losses: list[float] = []

    for index in range(
        1,
        period + 1,
    ):
        change = (
            closes[index]
            - closes[index - 1]
        )

        gains.append(
            max(change, 0.0)
        )

        losses.append(
            max(-change, 0.0)
        )

    average_gain = (
        sum(gains)
        / period
    )

    average_loss = (
        sum(losses)
        / period
    )

    for index in range(
        period + 1,
        len(closes),
    ):
        change = (
            closes[index]
            - closes[index - 1]
        )

        gain = max(
            change,
            0.0,
        )

        loss = max(
            -change,
            0.0,
        )

        average_gain = (
            (
                average_gain
                * (period - 1)
            )
            + gain
        ) / period

        average_loss = (
            (
                average_loss
                * (period - 1)
            )
            + loss
        ) / period

    if average_loss == 0:
        if average_gain == 0:
            return 50.0

        return 100.0

    relative_strength = (
        average_gain
        / average_loss
    )

    rsi = (
        100.0
        - (
            100.0
            / (
                1.0
                + relative_strength
            )
        )
    )

    return round(
        rsi,
        4,
    )


def calculate_macd(
    closes: list[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict:
    if (
        fast_period <= 0
        or slow_period <= 0
        or signal_period <= 0
    ):
        raise ValueError(
            "MACD periods must be positive."
        )

    if fast_period >= slow_period:
        raise ValueError(
            "fast_period must be less than slow_period."
        )

    fast_ema = (
        exponential_moving_average_series(
            closes,
            fast_period,
        )
    )

    slow_ema = (
        exponential_moving_average_series(
            closes,
            slow_period,
        )
    )

    macd_values: list[
        float | None
    ] = []

    valid_macd_values: list[
        float
    ] = []

    for fast, slow in zip(
        fast_ema,
        slow_ema,
    ):
        if (
            fast is None
            or slow is None
        ):
            macd_values.append(
                None
            )

        else:
            value = (
                fast - slow
            )

            macd_values.append(
                value
            )

            valid_macd_values.append(
                value
            )

    if (
        len(valid_macd_values)
        < signal_period
    ):
        return {
            "macd":
                None,

            "signal":
                None,

            "histogram":
                None,
        }

    signal_series = (
        exponential_moving_average_series(
            valid_macd_values,
            signal_period,
        )
    )

    signal_value = (
        signal_series[-1]
    )

    macd_value = (
        valid_macd_values[-1]
    )

    if signal_value is None:
        return {
            "macd": None,
            "signal": None,
            "histogram": None,
        }

    histogram = (
        macd_value
        - signal_value
    )

    return {
        "macd":
            round(
                macd_value,
                8,
            ),

        "signal":
            round(
                signal_value,
                8,
            ),

        "histogram":
            round(
                histogram,
                8,
            ),
    }


def calculate_bollinger_bands(
    closes: list[float],
    period: int = 20,
    standard_deviations: float = 2.0,
) -> dict:
    if period <= 1:
        raise ValueError(
            "period must be greater than one."
        )

    if standard_deviations <= 0:
        raise ValueError(
            "standard_deviations must be positive."
        )

    if len(closes) < period:
        return {
            "middle": None,
            "upper": None,
            "lower": None,
            "bandwidth_pct": None,
        }

    values = (
        closes[-period:]
    )

    middle = (
        sum(values)
        / period
    )

    variance = (
        sum(
            (
                value
                - middle
            ) ** 2
            for value in values
        )
        / period
    )

    standard_deviation = (
        sqrt(variance)
    )

    upper = (
        middle
        + standard_deviations
        * standard_deviation
    )

    lower = (
        middle
        - standard_deviations
        * standard_deviation
    )

    bandwidth_pct = (
        (
            upper - lower
        )
        / middle
        * 100.0
        if middle != 0
        else 0.0
    )

    return {
        "middle":
            round(
                middle,
                8,
            ),

        "upper":
            round(
                upper,
                8,
            ),

        "lower":
            round(
                lower,
                8,
            ),

        "bandwidth_pct":
            round(
                bandwidth_pct,
                4,
            ),
    }


def calculate_technical_snapshot(
    closes: list[float],
) -> dict:
    if not closes:
        raise ValueError(
            "closes cannot be empty."
        )

    closes = [
        float(value)
        for value in closes
    ]

    if any(
        value <= 0
        for value in closes
    ):
        raise ValueError(
            "all closes must be positive."
        )

    latest_price = (
        closes[-1]
    )

    sma20 = (
        simple_moving_average(
            closes,
            20,
        )
    )

    sma50 = (
        simple_moving_average(
            closes,
            50,
        )
    )

    rsi = (
        calculate_rsi(
            closes,
            14,
        )
    )

    macd = (
        calculate_macd(
            closes
        )
    )

    bollinger = (
        calculate_bollinger_bands(
            closes
        )
    )

    ready = all(
        value is not None
        for value in [
            sma20,
            sma50,
            rsi,
            macd["macd"],
            macd["signal"],
            bollinger["upper"],
            bollinger["lower"],
        ]
    )

    if not ready:
        return {
            "ready": False,
            "price": latest_price,
            "sma20": sma20,
            "sma50": sma50,
            "rsi14": rsi,
            "macd": macd,
            "bollinger": bollinger,
            "score": None,
            "rating": "WARMING_UP",
            "components": {},
        }

    score = 0

    components: dict[
        str,
        int,
    ] = {}

    # Trend component.
    sma_score = (
        1
        if sma20 > sma50
        else -1
        if sma20 < sma50
        else 0
    )

    score += sma_score

    components[
        "sma_trend"
    ] = sma_score


    # RSI is treated as a mean-reversion component.
    if rsi <= 30:
        rsi_score = 1

    elif rsi >= 70:
        rsi_score = -1

    else:
        rsi_score = 0

    score += rsi_score

    components[
        "rsi"
    ] = rsi_score


    # MACD momentum.
    if (
        macd["macd"]
        > macd["signal"]
    ):
        macd_score = 1

    elif (
        macd["macd"]
        < macd["signal"]
    ):
        macd_score = -1

    else:
        macd_score = 0

    score += macd_score

    components[
        "macd"
    ] = macd_score


    # Bollinger position is also treated
    # as a mean-reversion component.
    if (
        latest_price
        < bollinger["lower"]
    ):
        bollinger_score = 1

    elif (
        latest_price
        > bollinger["upper"]
    ):
        bollinger_score = -1

    else:
        bollinger_score = 0

    score += bollinger_score

    components[
        "bollinger"
    ] = bollinger_score


    if score >= 3:
        rating = "STRONG_BUY"

    elif score >= 1:
        rating = "BUY"

    elif score == 0:
        rating = "NEUTRAL"

    elif score <= -3:
        rating = "STRONG_SELL"

    else:
        rating = "SELL"

    return {
        "ready": True,

        "price":
            round(
                latest_price,
                8,
            ),

        "sma20":
            round(
                sma20,
                8,
            ),

        "sma50":
            round(
                sma50,
                8,
            ),

        "rsi14":
            rsi,

        "macd":
            macd,

        "bollinger":
            bollinger,

        "score":
            score,

        "max_score":
            4,

        "min_score":
            -4,

        "rating":
            rating,

        "components":
            components,
    }
