from __future__ import annotations

from typing import Any


def _importance(
    score: float,
) -> str:
    if score >= 0.70:
        return "HIGH"

    if score >= 0.35:
        return "MEDIUM"

    return "LOW"


class MarketInterpreter:
    """
    Converts raw technical indicators into a smaller,
    human-readable market interpretation.

    This first version is deterministic and heuristic.
    It does NOT predict future prices and its confidence
    value is not a statistical probability.
    """

    VERSION = "heuristic-v1"

    def interpret(
        self,
        technical: dict[str, Any],
        live_indicators: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        live_indicators = (
            live_indicators or {}
        )

        if (
            not technical
            or not technical.get(
                "ready"
            )
        ):
            return {
                "version":
                    self.VERSION,

                "ready":
                    False,

                "state":
                    "WARMING_UP",

                "confidence":
                    0,

                "headline":
                    (
                        "Not enough market data "
                        "is available yet."
                    ),

                "important":
                    [],

                "low_relevance":
                    [],

                "conflicts":
                    [],

                "explanation":
                    (
                        "The indicator engine is "
                        "still collecting enough "
                        "candles for analysis."
                    ),
            }

        price = float(
            technical["price"]
        )

        sma20 = float(
            technical["sma20"]
        )

        sma50 = float(
            technical["sma50"]
        )

        rsi = float(
            technical["rsi14"]
        )

        macd = technical[
            "macd"
        ]

        bollinger = technical[
            "bollinger"
        ]

        score = int(
            technical["score"]
        )

        # ---------------------------------
        # TREND
        # ---------------------------------

        spread_pct = (
            (
                sma20 - sma50
            )
            / sma50
            * 100.0
            if sma50
            else 0.0
        )

        if spread_pct > 0:
            trend_direction = (
                "BULLISH"
            )

        elif spread_pct < 0:
            trend_direction = (
                "BEARISH"
            )

        else:
            trend_direction = (
                "NEUTRAL"
            )

        trend_strength = min(
            abs(spread_pct)
            / 0.10,
            1.0,
        )

        trend_importance = (
            _importance(
                trend_strength
            )
        )

        trend_detail = (
            f"SMA20 is "
            f"{abs(spread_pct):.4f}% "
            f"{'above' if spread_pct >= 0 else 'below'} "
            f"SMA50."
        )

        # ---------------------------------
        # MOMENTUM / MACD
        # ---------------------------------

        macd_value = float(
            macd["macd"]
        )

        macd_signal = float(
            macd["signal"]
        )

        macd_histogram = float(
            macd["histogram"]
        )

        if macd_histogram > 0:
            momentum_direction = (
                "BULLISH"
            )

        elif macd_histogram < 0:
            momentum_direction = (
                "BEARISH"
            )

        else:
            momentum_direction = (
                "NEUTRAL"
            )

        histogram_pct = (
            abs(
                macd_histogram
            )
            / price
            * 100.0
            if price
            else 0.0
        )

        momentum_strength = min(
            histogram_pct
            / 0.01,
            1.0,
        )

        momentum_importance = (
            _importance(
                momentum_strength
            )
        )

        momentum_detail = (
            "MACD is "
            + (
                "above"
                if macd_value
                > macd_signal
                else "below"
                if macd_value
                < macd_signal
                else "equal to"
            )
            + " its signal line."
        )

        # ---------------------------------
        # RSI
        # ---------------------------------

        if rsi >= 70:
            rsi_state = (
                "OVERBOUGHT"
            )

            rsi_importance_score = (
                min(
                    0.70
                    + (
                        rsi - 70
                    )
                    / 30,
                    1.0,
                )
            )

        elif rsi <= 30:
            rsi_state = (
                "OVERSOLD"
            )

            rsi_importance_score = (
                min(
                    0.70
                    + (
                        30 - rsi
                    )
                    / 30,
                    1.0,
                )
            )

        elif (
            45 <= rsi <= 55
        ):
            rsi_state = "NEUTRAL"

            rsi_importance_score = (
                0.10
            )

        elif rsi > 55:
            rsi_state = (
                "BULLISH_BIAS"
            )

            rsi_importance_score = (
                0.40
            )

        else:
            rsi_state = (
                "BEARISH_BIAS"
            )

            rsi_importance_score = (
                0.40
            )

        rsi_importance = (
            _importance(
                rsi_importance_score
            )
        )

        rsi_detail = (
            f"RSI14 is {rsi:.2f}."
        )

        # ---------------------------------
        # BOLLINGER / VOLATILITY
        # ---------------------------------

        bb_upper = float(
            bollinger["upper"]
        )

        bb_middle = float(
            bollinger["middle"]
        )

        bb_lower = float(
            bollinger["lower"]
        )

        bandwidth_pct = float(
            bollinger[
                "bandwidth_pct"
            ]
        )

        band_range = (
            bb_upper - bb_lower
        )

        if band_range > 0:
            band_position = (
                (
                    price
                    - bb_lower
                )
                / band_range
            )
        else:
            band_position = 0.5

        band_position = max(
            0.0,
            min(
                1.0,
                band_position,
            ),
        )

        if band_position >= 0.90:
            bb_state = (
                "NEAR_UPPER_BAND"
            )

            bb_importance_score = (
                0.75
            )

        elif band_position <= 0.10:
            bb_state = (
                "NEAR_LOWER_BAND"
            )

            bb_importance_score = (
                0.75
            )

        elif (
            band_position >= 0.75
            or band_position <= 0.25
        ):
            bb_state = (
                "OUTER_REGION"
            )

            bb_importance_score = (
                0.40
            )

        else:
            bb_state = (
                "MID_RANGE"
            )

            bb_importance_score = (
                0.15
            )

        bb_importance = (
            _importance(
                bb_importance_score
            )
        )

        if bandwidth_pct < 0.20:
            volatility_state = (
                "LOW"
            )

        elif bandwidth_pct > 2.0:
            volatility_state = (
                "HIGH"
            )

        else:
            volatility_state = (
                "NORMAL"
            )

        bb_detail = (
            f"Price is at "
            f"{band_position * 100:.1f}% "
            f"of the Bollinger range; "
            f"bandwidth is "
            f"{bandwidth_pct:.4f}%."
        )

        # ---------------------------------
        # CONFLICT DETECTION
        # ---------------------------------

        conflicts: list[str] = []

        if (
            trend_direction
            == "BULLISH"
            and momentum_direction
            == "BEARISH"
        ):
            conflicts.append(
                (
                    "Bullish SMA trend conflicts "
                    "with bearish MACD momentum."
                )
            )

        elif (
            trend_direction
            == "BEARISH"
            and momentum_direction
            == "BULLISH"
        ):
            conflicts.append(
                (
                    "Bearish SMA trend conflicts "
                    "with bullish MACD momentum."
                )
            )

        # ---------------------------------
        # FACTOR RANKING
        # ---------------------------------

        factors = [
            {
                "name":
                    "SMA Trend",

                "importance":
                    trend_importance,

                "importance_score":
                    round(
                        trend_strength,
                        4,
                    ),

                "direction":
                    trend_direction,

                "detail":
                    trend_detail,
            },

            {
                "name":
                    "MACD Momentum",

                "importance":
                    momentum_importance,

                "importance_score":
                    round(
                        momentum_strength,
                        4,
                    ),

                "direction":
                    momentum_direction,

                "detail":
                    momentum_detail,
            },

            {
                "name":
                    "RSI",

                "importance":
                    rsi_importance,

                "importance_score":
                    round(
                        rsi_importance_score,
                        4,
                    ),

                "direction":
                    rsi_state,

                "detail":
                    rsi_detail,
            },

            {
                "name":
                    "Bollinger Position",

                "importance":
                    bb_importance,

                "importance_score":
                    round(
                        bb_importance_score,
                        4,
                    ),

                "direction":
                    bb_state,

                "detail":
                    bb_detail,
            },
        ]

        factors.sort(
            key=lambda item:
                item[
                    "importance_score"
                ],
            reverse=True,
        )

        important = [
            factor
            for factor in factors
            if factor[
                "importance"
            ] != "LOW"
        ]

        low_relevance = [
            factor
            for factor in factors
            if factor[
                "importance"
            ] == "LOW"
        ]

        # ---------------------------------
        # MARKET STATE
        # ---------------------------------

        if score >= 3:
            state = "STRONG_BUY"

        elif score >= 1:
            state = "BUY"

        elif score <= -3:
            state = "STRONG_SELL"

        elif score <= -1:
            state = "SELL"

        else:
            state = "NEUTRAL"

        # This is a heuristic explanation score,
        # NOT a probability of future returns.

        active_votes = [
            value
            for value in technical[
                "components"
            ].values()
            if value != 0
        ]

        if not active_votes:
            agreement = 0.0

        elif score > 0:
            agreement = (
                sum(
                    1
                    for vote
                    in active_votes
                    if vote > 0
                )
                / len(active_votes)
            )

        elif score < 0:
            agreement = (
                sum(
                    1
                    for vote
                    in active_votes
                    if vote < 0
                )
                / len(active_votes)
            )

        else:
            agreement = 0.5

        confidence = (
            45
            + abs(score) * 8
            + agreement * 15
            - len(conflicts) * 10
        )

        confidence = int(
            round(
                max(
                    25,
                    min(
                        90,
                        confidence,
                    ),
                )
            )
        )

        # ---------------------------------
        # HUMAN SUMMARY
        # ---------------------------------

        if (
            state == "NEUTRAL"
            and conflicts
        ):
            headline = (
                "Signals are mixed."
            )

        elif state in {
            "BUY",
            "STRONG_BUY",
        }:
            headline = (
                "The technical picture "
                "has a bullish bias."
            )

        elif state in {
            "SELL",
            "STRONG_SELL",
        }:
            headline = (
                "The technical picture "
                "has a bearish bias."
            )

        else:
            headline = (
                "No clear directional "
                "bias is present."
            )

        explanation_parts = []

        explanation_parts.append(
            (
                f"The SMA trend is "
                f"{trend_direction.lower()}."
            )
        )

        explanation_parts.append(
            (
                f"MACD momentum is "
                f"{momentum_direction.lower()}."
            )
        )

        if rsi_importance == "LOW":
            explanation_parts.append(
                (
                    f"RSI is {rsi:.1f} "
                    "and is not especially "
                    "informative right now."
                )
            )

        elif rsi_state == "OVERBOUGHT":
            explanation_parts.append(
                (
                    f"RSI is elevated at "
                    f"{rsi:.1f}."
                )
            )

        elif rsi_state == "OVERSOLD":
            explanation_parts.append(
                (
                    f"RSI is depressed at "
                    f"{rsi:.1f}."
                )
            )

        if conflicts:
            explanation_parts.append(
                conflicts[0]
            )

        return {
            "version":
                self.VERSION,

            "ready":
                True,

            "state":
                state,

            "confidence":
                confidence,

            "confidence_type":
                "heuristic",

            "headline":
                headline,

            "trend": {
                "direction":
                    trend_direction,

                "importance":
                    trend_importance,

                "spread_pct":
                    round(
                        spread_pct,
                        4,
                    ),
            },

            "momentum": {
                "direction":
                    momentum_direction,

                "importance":
                    momentum_importance,

                "macd_histogram":
                    macd_histogram,
            },

            "rsi": {
                "value":
                    rsi,

                "state":
                    rsi_state,

                "importance":
                    rsi_importance,
            },

            "volatility": {
                "state":
                    volatility_state,

                "bollinger_bandwidth_pct":
                    bandwidth_pct,
            },

            "bollinger": {
                "state":
                    bb_state,

                "importance":
                    bb_importance,

                "position":
                    round(
                        band_position,
                        4,
                    ),

                "middle":
                    bb_middle,
            },

            "important":
                important,

            "low_relevance":
                low_relevance,

            "conflicts":
                conflicts,

            "explanation":
                " ".join(
                    explanation_parts
                ),

            "technical_score":
                score,

            "live_signal":
                live_indicators.get(
                    "signal"
                ),

            "live_crossover":
                live_indicators.get(
                    "crossover"
                ),
        }
