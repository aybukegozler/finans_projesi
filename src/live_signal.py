from __future__ import annotations

from collections import deque
from datetime import datetime, timezone


class LiveSMAEngine:
    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        max_points: int = 250,
    ):
        if short_window < 2:
            raise ValueError(
                "short_window must be at least 2."
            )

        if long_window <= short_window:
            raise ValueError(
                "long_window must be greater than short_window."
            )

        if max_points < long_window + 1:
            raise ValueError(
                "max_points must be greater than long_window."
            )

        self.short_window = short_window
        self.long_window = long_window

        self.candles = deque(
            maxlen=max_points
        )

        self.last_crossover = None

    @staticmethod
    def _normalize_candle(
        candle: dict,
    ) -> dict:
        open_time_ms = int(
            candle["open_time_ms"]
        )

        close = float(
            candle["close"]
        )

        if close <= 0:
            raise ValueError(
                "Candle close must be greater than zero."
            )

        return {
            "open_time_ms": open_time_ms,
            "close": close,
            "closed": bool(
                candle.get(
                    "closed",
                    False,
                )
            ),
        }

    @staticmethod
    def _timestamp_to_iso(
        timestamp_ms: int,
    ) -> str:
        return (
            datetime.fromtimestamp(
                timestamp_ms / 1000,
                tz=timezone.utc,
            )
            .isoformat()
        )

    def seed(
        self,
        candles: list[dict],
    ) -> dict:
        self.candles.clear()
        self.last_crossover = None

        snapshot = self.snapshot()

        for candle in candles:
            snapshot = self.update(
                candle
            )

        return snapshot

    def update(
        self,
        candle: dict,
    ) -> dict:
        normalized = (
            self._normalize_candle(
                candle
            )
        )

        if (
            self.candles
            and self.candles[-1][
                "open_time_ms"
            ]
            == normalized[
                "open_time_ms"
            ]
        ):
            # Binance aynı açık mumu sürekli
            # günceller. Yeni mum sanıp append
            # etmek yerine son mumu değiştiriyoruz.
            self.candles[-1] = normalized

        else:
            self.candles.append(
                normalized
            )

        result = self.snapshot()

        if (
            result["crossover"] != "HOLD"
            and normalized["closed"]
        ):
            self.last_crossover = {
                "signal":
                    result["crossover"],

                "time_ms":
                    normalized[
                        "open_time_ms"
                    ],

                "time":
                    self._timestamp_to_iso(
                        normalized[
                            "open_time_ms"
                        ]
                    ),

                "confirmed":
                    normalized["closed"],
            }

        result["last_crossover"] = (
            self.last_crossover
        )

        return result

    def snapshot(
        self,
    ) -> dict:
        count = len(
            self.candles
        )

        if count == 0:
            return {
                "ready": False,
                "candle_count": 0,
                "short_window":
                    self.short_window,
                "long_window":
                    self.long_window,
                "sma_short": None,
                "sma_long": None,
                "spread_pct": None,
                "trend": "WARMING_UP",
                "signal": "HOLD",
                "crossover": "HOLD",
                "candle_closed": False,
                "last_crossover":
                    self.last_crossover,
            }

        latest = self.candles[-1]

        if count < self.long_window:
            return {
                "ready": False,
                "candle_count": count,
                "short_window":
                    self.short_window,
                "long_window":
                    self.long_window,
                "sma_short": None,
                "sma_long": None,
                "spread_pct": None,
                "trend": "WARMING_UP",
                "signal": "HOLD",
                "crossover": "HOLD",
                "candle_closed":
                    latest["closed"],
                "last_crossover":
                    self.last_crossover,
            }

        closes = [
            candle["close"]
            for candle in self.candles
        ]

        sma_short = (
            sum(
                closes[
                    -self.short_window:
                ]
            )
            / self.short_window
        )

        sma_long = (
            sum(
                closes[
                    -self.long_window:
                ]
            )
            / self.long_window
        )

        if sma_short > sma_long:
            trend = "BULLISH"
            signal = "BUY"

        elif sma_short < sma_long:
            trend = "BEARISH"
            signal = "SELL"

        else:
            trend = "NEUTRAL"
            signal = "HOLD"

        crossover = "HOLD"

        if count >= self.long_window + 1:
            previous_closes = (
                closes[:-1]
            )

            previous_short = (
                sum(
                    previous_closes[
                        -self.short_window:
                    ]
                )
                / self.short_window
            )

            previous_long = (
                sum(
                    previous_closes[
                        -self.long_window:
                    ]
                )
                / self.long_window
            )

            if (
                previous_short
                <= previous_long
                and sma_short
                > sma_long
            ):
                crossover = "BUY"

            elif (
                previous_short
                >= previous_long
                and sma_short
                < sma_long
            ):
                crossover = "SELL"

        spread_pct = (
            (
                sma_short
                - sma_long
            )
            / sma_long
            * 100.0
        )

        return {
            "ready": True,

            "candle_count":
                count,

            "short_window":
                self.short_window,

            "long_window":
                self.long_window,

            "sma_short":
                round(
                    sma_short,
                    8,
                ),

            "sma_long":
                round(
                    sma_long,
                    8,
                ),

            "spread_pct":
                round(
                    spread_pct,
                    4,
                ),

            "trend":
                trend,

            "signal":
                signal,

            "crossover":
                crossover,

            "candle_closed":
                latest["closed"],

            "last_crossover":
                self.last_crossover,
        }
