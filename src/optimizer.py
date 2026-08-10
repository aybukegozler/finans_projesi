from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.backtest import (
    calculate_max_drawdown,
    calculate_risk_metrics,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MARKET_DATA_PATH = ROOT_DIR / "data" / "market_data.csv"


def load_market_data(
    market_data_path: Path | str = DEFAULT_MARKET_DATA_PATH,
) -> pd.DataFrame:
    path = Path(market_data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Market data file not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "Date",
        "Close",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Date", "Close"]
    )

    df = df[
        df["Close"] > 0
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid market data is available."
        )

    return df


def generate_sma_signals(
    market_data: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    if short_window < 2:
        raise ValueError(
            "short_window must be at least 2."
        )

    if long_window <= short_window:
        raise ValueError(
            "long_window must be greater than short_window."
        )

    if len(market_data) <= long_window:
        raise ValueError(
            "Not enough market data for selected windows."
        )

    df = market_data.copy()

    df["SMA_SHORT"] = (
        df["Close"]
        .rolling(
            window=short_window,
            min_periods=short_window,
        )
        .mean()
    )

    df["SMA_LONG"] = (
        df["Close"]
        .rolling(
            window=long_window,
            min_periods=long_window,
        )
        .mean()
    )

    current_bullish = (
        df["SMA_SHORT"]
        > df["SMA_LONG"]
    )

    previous_bullish = (
        current_bullish
        .shift(
            1,
            fill_value=False,
        )
        .astype(bool)
    )

    current_valid = (
        df["SMA_SHORT"].notna()
        & df["SMA_LONG"].notna()
    )

    previous_valid = (
        df["SMA_SHORT"].shift(1).notna()
        & df["SMA_LONG"].shift(1).notna()
    )

    buy_signal = (
        current_valid
        & previous_valid
        & (~previous_bullish)
        & current_bullish
    )

    sell_signal = (
        current_valid
        & previous_valid
        & previous_bullish
        & (~current_bullish)
    )

    df["Signal"] = 0

    df.loc[
        buy_signal,
        "Signal",
    ] = 1

    df.loc[
        sell_signal,
        "Signal",
    ] = -1

    return df[
        [
            "Date",
            "Close",
            "SMA_SHORT",
            "SMA_LONG",
            "Signal",
        ]
    ]


def evaluate_strategy(
    signal_data: pd.DataFrame,
    initial_capital: float = 10_000.0,
    transaction_fee_pct: float = 0.10,
    slippage_pct: float = 0.05,
    force_close_at_end: bool = False,
) -> dict:
    if initial_capital <= 0:
        raise ValueError(
            "initial_capital must be greater than zero."
        )

    if not 0 <= transaction_fee_pct <= 5:
        raise ValueError(
            "transaction_fee_pct must be between 0 and 5."
        )

    if not 0 <= slippage_pct <= 5:
        raise ValueError(
            "slippage_pct must be between 0 and 5."
        )

    fee_rate = (
        transaction_fee_pct / 100.0
    )

    slippage_rate = (
        slippage_pct / 100.0
    )

    cash = float(initial_capital)
    shares = 0.0
    entry_value = None

    closed_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_fees_paid = 0.0

    equity_curve: list[float] = []

    for row in signal_data.itertuples(
        index=False
    ):
        market_price = float(
            row.Close
        )

        signal = int(
            row.Signal
        )

        if signal == 1 and shares == 0:
            effective_buy_price = (
                market_price
                * (1.0 + slippage_rate)
            )

            entry_value = cash

            shares = (
                cash
                / (
                    effective_buy_price
                    * (1.0 + fee_rate)
                )
            )

            buy_notional = (
                shares
                * effective_buy_price
            )

            buy_fee = (
                buy_notional
                * fee_rate
            )

            total_fees_paid += buy_fee
            cash = 0.0

        elif signal == -1 and shares > 0:
            effective_sell_price = (
                market_price
                * (1.0 - slippage_rate)
            )

            gross_exit = (
                shares
                * effective_sell_price
            )

            sell_fee = (
                gross_exit
                * fee_rate
            )

            total_fees_paid += sell_fee

            exit_value = (
                gross_exit
                - sell_fee
            )

            if (
                entry_value is not None
                and exit_value > entry_value
            ):
                winning_trades += 1
            else:
                losing_trades += 1

            closed_trades += 1

            cash = exit_value
            shares = 0.0
            entry_value = None

        equity = (
            cash
            + shares * market_price
        )

        equity_curve.append(
            equity
        )

    if not equity_curve:
        raise ValueError(
            "Strategy produced no equity data."
        )

    last_price = float(
        signal_data.iloc[-1]["Close"]
    )

    first_price = float(
        signal_data.iloc[0]["Close"]
    )

    # Walk-forward test penceresinin sonunda açık pozisyon
    # varsa gerçekçi biçimde kapatılır. Böylece sonraki fold'a
    # hayali mark-to-market nakit taşınmaz.
    if force_close_at_end and shares > 0:
        effective_sell_price = (
            last_price
            * (1.0 - slippage_rate)
        )

        gross_exit = (
            shares
            * effective_sell_price
        )

        sell_fee = (
            gross_exit
            * fee_rate
        )

        total_fees_paid += sell_fee

        exit_value = (
            gross_exit
            - sell_fee
        )

        if (
            entry_value is not None
            and exit_value > entry_value
        ):
            winning_trades += 1
        else:
            losing_trades += 1

        closed_trades += 1

        cash = exit_value
        shares = 0.0
        entry_value = None

        if equity_curve:
            equity_curve[-1] = cash

    final_value = (
        cash
        + shares * last_price
    )

    total_return_pct = (
        (
            final_value
            / initial_capital
        )
        - 1
    ) * 100.0

    buy_hold_return_pct = (
        (
            last_price
            / first_price
        )
        - 1
    ) * 100.0

    excess_return_pct = (
        total_return_pct
        - buy_hold_return_pct
    )

    win_rate_pct = (
        winning_trades
        / closed_trades
        * 100.0
        if closed_trades
        else 0.0
    )

    max_drawdown_pct = (
        calculate_max_drawdown(
            equity_curve
        )
    )

    (
        sharpe_ratio,
        annualized_volatility_pct,
    ) = calculate_risk_metrics(
        equity_curve
    )

    return {
        "final_value": round(
            final_value,
            2,
        ),
        "total_return_pct": round(
            total_return_pct,
            2,
        ),
        "buy_hold_return_pct": round(
            buy_hold_return_pct,
            2,
        ),
        "excess_return_pct": round(
            excess_return_pct,
            2,
        ),
        "closed_trades": closed_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": round(
            win_rate_pct,
            2,
        ),
        "max_drawdown_pct": round(
            max_drawdown_pct,
            2,
        ),
        "sharpe_ratio": round(
            sharpe_ratio,
            3,
        ),
        "annualized_volatility_pct": round(
            annualized_volatility_pct,
            2,
        ),
        "total_fees_paid": round(
            total_fees_paid,
            2,
        ),
    }


def optimize_sma_strategy(
    market_data_path: Path | str = DEFAULT_MARKET_DATA_PATH,
    short_windows: Iterable[int] = (
        5,
        10,
        15,
        20,
        25,
        30,
    ),
    long_windows: Iterable[int] = (
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    ),
    initial_capital: float = 10_000.0,
    transaction_fee_pct: float = 0.10,
    slippage_pct: float = 0.05,
    objective: str = "sharpe_ratio",
    top_n: int = 10,
) -> dict:
    allowed_objectives = {
        "sharpe_ratio",
        "total_return_pct",
        "excess_return_pct",
    }

    if objective not in allowed_objectives:
        raise ValueError(
            "objective must be one of: "
            + ", ".join(
                sorted(allowed_objectives)
            )
        )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    market_data = load_market_data(
        market_data_path
    )

    results: list[dict] = []

    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue

            signal_data = (
                generate_sma_signals(
                    market_data,
                    short_window,
                    long_window,
                )
            )

            metrics = evaluate_strategy(
                signal_data,
                initial_capital=initial_capital,
                transaction_fee_pct=transaction_fee_pct,
                slippage_pct=slippage_pct,
            )

            results.append(
                {
                    "short_window": short_window,
                    "long_window": long_window,
                    **metrics,
                }
            )

    if not results:
        raise ValueError(
            "No valid SMA combinations were tested."
        )

    results.sort(
        key=lambda item: (
            item[objective],
            item["total_return_pct"],
        ),
        reverse=True,
    )

    best = results[0]

    return {
        "objective": objective,
        "tested_combinations": len(
            results
        ),
        "best": best,
        "top_results": results[
            :top_n
        ],
    }


if __name__ == "__main__":
    result = optimize_sma_strategy()

    print(
        "=== SMA STRATEGY OPTIMIZER ==="
    )

    print(
        "Objective:",
        result["objective"],
    )

    print(
        "Tested combinations:",
        result["tested_combinations"],
    )

    print("\nBEST CONFIGURATION")

    for key, value in (
        result["best"].items()
    ):
        print(
            f"{key}: {value}"
        )

    print("\nTOP 5")

    for index, row in enumerate(
        result["top_results"][:5],
        start=1,
    ):
        print(
            f"{index}. "
            f"SMA{row['short_window']}/"
            f"SMA{row['long_window']} | "
            f"Return: "
            f"{row['total_return_pct']}% | "
            f"Sharpe: "
            f"{row['sharpe_ratio']} | "
            f"DD: "
            f"{row['max_drawdown_pct']}%"
        )
