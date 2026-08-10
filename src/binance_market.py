from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import AsyncIterator
from urllib.parse import urlencode
from urllib.request import urlopen

from websockets.asyncio.client import connect


BINANCE_REST_BASE = (
    "https://api.binance.com"
)

BINANCE_STREAM_BASE = (
    "wss://stream.binance.com:9443/ws"
)

ALLOWED_INTERVALS = {
    "1s",
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}


def normalize_symbol(
    symbol: str,
) -> str:
    normalized = (
        symbol
        .strip()
        .upper()
        .replace("/", "")
        .replace("-", "")
    )

    if not normalized:
        raise ValueError(
            "Symbol cannot be empty."
        )

    if not normalized.isalnum():
        raise ValueError(
            "Invalid Binance symbol."
        )

    return normalized


def validate_interval(
    interval: str,
) -> str:
    if interval not in ALLOWED_INTERVALS:
        raise ValueError(
            "Unsupported Binance interval."
        )

    return interval


def timestamp_to_iso(
    timestamp_ms: int,
) -> str:
    return (
        datetime.fromtimestamp(
            timestamp_ms / 1000,
            tz=timezone.utc,
        )
        .isoformat()
    )


def get_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    limit: int = 500,
) -> list[dict]:
    symbol = normalize_symbol(
        symbol
    )

    interval = validate_interval(
        interval
    )

    if not 1 <= limit <= 1000:
        raise ValueError(
            "limit must be between 1 and 1000."
        )

    query = urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
    )

    url = (
        f"{BINANCE_REST_BASE}"
        f"/api/v3/klines?{query}"
    )

    with urlopen(
        url,
        timeout=10,
    ) as response:
        raw_data = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    result: list[dict] = []

    for item in raw_data:
        result.append(
            {
                "open_time":
                    timestamp_to_iso(
                        int(item[0])
                    ),

                "open_time_ms":
                    int(item[0]),

                "open":
                    float(item[1]),

                "high":
                    float(item[2]),

                "low":
                    float(item[3]),

                "close":
                    float(item[4]),

                "volume":
                    float(item[5]),

                "close_time_ms":
                    int(item[6]),

                "quote_volume":
                    float(item[7]),

                "trade_count":
                    int(item[8]),
            }
        )

    return result


def parse_kline_message(
    raw_message: str,
) -> dict:
    payload = json.loads(
        raw_message
    )

    if payload.get("e") != "kline":
        raise ValueError(
            "Unexpected Binance event."
        )

    kline = payload["k"]

    return {
        "event":
            payload["e"],

        "symbol":
            payload["s"],

        "event_time_ms":
            int(payload["E"]),

        "event_time":
            timestamp_to_iso(
                int(payload["E"])
            ),

        "interval":
            kline["i"],

        "open_time_ms":
            int(kline["t"]),

        "close_time_ms":
            int(kline["T"]),

        "open":
            float(kline["o"]),

        "high":
            float(kline["h"]),

        "low":
            float(kline["l"]),

        "close":
            float(kline["c"]),

        "volume":
            float(kline["v"]),

        "quote_volume":
            float(kline["q"]),

        "trade_count":
            int(kline["n"]),

        "closed":
            bool(kline["x"]),
    }


async def stream_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
) -> AsyncIterator[dict]:
    symbol = normalize_symbol(
        symbol
    )

    interval = validate_interval(
        interval
    )

    stream_name = (
        f"{symbol.lower()}"
        f"@kline_{interval}"
    )

    url = (
        f"{BINANCE_STREAM_BASE}/"
        f"{stream_name}"
    )

    # websockets.connect async iterator
    # bağlantı kopmalarında yeniden bağlanabilir.
    async for websocket in connect(
        url,
        open_timeout=10,
        close_timeout=5,
    ):
        try:
            async for message in websocket:
                yield parse_kline_message(
                    message
                )

        except Exception as error:
            print(
                "Binance stream yeniden "
                "bağlanıyor:",
                type(error).__name__,
            )

            continue


if __name__ == "__main__":
    import asyncio


    async def demo():
        print(
            "=== BINANCE LIVE MARKET ==="
        )

        count = 0

        async for kline in stream_klines(
            "BTCUSDT",
            "1m",
        ):
            print(
                kline["symbol"],
                "|",
                kline["interval"],
                "| Price:",
                kline["close"],
                "| Closed:",
                kline["closed"],
            )

            count += 1

            if count >= 5:
                break


    asyncio.run(demo())
