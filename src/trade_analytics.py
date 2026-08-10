from __future__ import annotations

from pathlib import Path
from statistics import mean

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SIGNALS_PATH = ROOT_DIR / "data" / "signals.csv"


def analyze_trades(
    signals_path: Path | str = DEFAULT_SIGNALS_PATH,
    initial_capital: float = 10_000.0,
    transaction_fee_pct: float = 0.10,
    slippage_pct: float = 0.05,
    force_close_at_end: bool = False,
) -> dict:
    path = Path(signals_path)

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

    if not path.exists():
        raise FileNotFoundError(
            f"Signal file not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "Date",
        "Close",
        "Signal",
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

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    df["Signal"] = pd.to_numeric(
        df["Signal"],
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=[
                "Date",
                "Close",
                "Signal",
            ]
        )
        .query("Close > 0")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "No valid market data is available."
        )

    fee_rate = (
        transaction_fee_pct / 100.0
    )

    slippage_rate = (
        slippage_pct / 100.0
    )

    cash = float(initial_capital)
    shares = 0.0
    entry = None

    trades: list[dict] = []

    def close_trade(
        exit_date,
        exit_market_price: float,
        forced_exit: bool,
    ) -> None:
        nonlocal cash
        nonlocal shares
        nonlocal entry

        if entry is None or shares <= 0:
            return

        effective_exit_price = (
            exit_market_price
            * (1.0 - slippage_rate)
        )

        gross_exit_value = (
            shares
            * effective_exit_price
        )

        sell_fee = (
            gross_exit_value
            * fee_rate
        )

        net_exit_value = (
            gross_exit_value
            - sell_fee
        )

        gross_pnl = (
            shares
            * (
                exit_market_price
                - entry["market_entry_price"]
            )
        )

        exit_slippage_cost = (
            shares
            * (
                exit_market_price
                - effective_exit_price
            )
        )

        slippage_cost = (
            entry["entry_slippage_cost"]
            + exit_slippage_cost
        )

        total_fees = (
            entry["buy_fee"]
            + sell_fee
        )

        total_costs = (
            total_fees
            + slippage_cost
        )

        net_pnl = (
            net_exit_value
            - entry["entry_capital"]
        )

        return_pct = (
            net_pnl
            / entry["entry_capital"]
            * 100.0
        )

        holding_days = (
            exit_date
            - entry["entry_date"]
        ).days

        trades.append(
            {
                "trade_number":
                    len(trades) + 1,

                "entry_date":
                    entry[
                        "entry_date"
                    ].strftime("%Y-%m-%d"),

                "exit_date":
                    exit_date.strftime(
                        "%Y-%m-%d"
                    ),

                "holding_days":
                    holding_days,

                "market_entry_price":
                    round(
                        entry[
                            "market_entry_price"
                        ],
                        4,
                    ),

                "effective_entry_price":
                    round(
                        entry[
                            "effective_entry_price"
                        ],
                        4,
                    ),

                "market_exit_price":
                    round(
                        exit_market_price,
                        4,
                    ),

                "effective_exit_price":
                    round(
                        effective_exit_price,
                        4,
                    ),

                "shares":
                    round(
                        shares,
                        6,
                    ),

                "gross_pnl":
                    round(
                        gross_pnl,
                        2,
                    ),

                "net_pnl":
                    round(
                        net_pnl,
                        2,
                    ),

                "return_pct":
                    round(
                        return_pct,
                        2,
                    ),

                "buy_fee":
                    round(
                        entry["buy_fee"],
                        2,
                    ),

                "sell_fee":
                    round(
                        sell_fee,
                        2,
                    ),

                "total_fees":
                    round(
                        total_fees,
                        2,
                    ),

                "slippage_cost":
                    round(
                        slippage_cost,
                        2,
                    ),

                "total_costs":
                    round(
                        total_costs,
                        2,
                    ),

                "result":
                    (
                        "WIN"
                        if net_pnl > 0
                        else "LOSS"
                    ),

                "forced_exit":
                    forced_exit,
            }
        )

        cash = net_exit_value
        shares = 0.0
        entry = None

    for row in df.itertuples(
        index=False
    ):
        date = row.Date
        market_price = float(
            row.Close
        )
        signal = int(
            row.Signal
        )

        if signal == 1 and shares == 0:
            effective_entry_price = (
                market_price
                * (1.0 + slippage_rate)
            )

            entry_capital = cash

            shares = (
                cash
                / (
                    effective_entry_price
                    * (1.0 + fee_rate)
                )
            )

            buy_notional = (
                shares
                * effective_entry_price
            )

            buy_fee = (
                buy_notional
                * fee_rate
            )

            entry_slippage_cost = (
                shares
                * (
                    effective_entry_price
                    - market_price
                )
            )

            entry = {
                "entry_date":
                    date,

                "entry_capital":
                    entry_capital,

                "market_entry_price":
                    market_price,

                "effective_entry_price":
                    effective_entry_price,

                "buy_fee":
                    buy_fee,

                "entry_slippage_cost":
                    entry_slippage_cost,
            }

            cash = 0.0

        elif signal == -1 and shares > 0:
            close_trade(
                exit_date=date,
                exit_market_price=market_price,
                forced_exit=False,
            )

    if (
        force_close_at_end
        and shares > 0
        and entry is not None
    ):
        final_row = df.iloc[-1]

        close_trade(
            exit_date=final_row["Date"],
            exit_market_price=float(
                final_row["Close"]
            ),
            forced_exit=True,
        )

    open_position = None

    if shares > 0 and entry is not None:
        final_row = df.iloc[-1]

        current_price = float(
            final_row["Close"]
        )

        effective_liquidation_price = (
            current_price
            * (1.0 - slippage_rate)
        )

        gross_liquidation_value = (
            shares
            * effective_liquidation_price
        )

        estimated_sell_fee = (
            gross_liquidation_value
            * fee_rate
        )

        estimated_liquidation_value = (
            gross_liquidation_value
            - estimated_sell_fee
        )

        unrealized_pnl = (
            estimated_liquidation_value
            - entry["entry_capital"]
        )

        open_position = {
            "entry_date":
                entry[
                    "entry_date"
                ].strftime("%Y-%m-%d"),

            "current_date":
                final_row[
                    "Date"
                ].strftime("%Y-%m-%d"),

            "holding_days":
                (
                    final_row["Date"]
                    - entry["entry_date"]
                ).days,

            "market_entry_price":
                round(
                    entry[
                        "market_entry_price"
                    ],
                    4,
                ),

            "current_market_price":
                round(
                    current_price,
                    4,
                ),

            "shares":
                round(
                    shares,
                    6,
                ),

            "estimated_liquidation_value":
                round(
                    estimated_liquidation_value,
                    2,
                ),

            "unrealized_pnl":
                round(
                    unrealized_pnl,
                    2,
                ),

            "unrealized_return_pct":
                round(
                    unrealized_pnl
                    / entry["entry_capital"]
                    * 100.0,
                    2,
                ),
        }

    winning_trades = [
        trade
        for trade in trades
        if trade["net_pnl"] > 0
    ]

    losing_trades = [
        trade
        for trade in trades
        if trade["net_pnl"] <= 0
    ]

    total_net_pnl = sum(
        trade["net_pnl"]
        for trade in trades
    )

    total_fees = sum(
        trade["total_fees"]
        for trade in trades
    )

    total_slippage_cost = sum(
        trade["slippage_cost"]
        for trade in trades
    )

    gross_profit = sum(
        trade["net_pnl"]
        for trade in winning_trades
    )

    gross_loss = abs(
        sum(
            trade["net_pnl"]
            for trade in losing_trades
        )
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else None
    )

    returns = [
        trade["return_pct"]
        for trade in trades
    ]

    holding_periods = [
        trade["holding_days"]
        for trade in trades
    ]

    summary = {
        "closed_trades":
            len(trades),

        "winning_trades":
            len(winning_trades),

        "losing_trades":
            len(losing_trades),

        "win_rate_pct":
            round(
                (
                    len(winning_trades)
                    / len(trades)
                    * 100.0
                )
                if trades
                else 0.0,
                2,
            ),

        "total_net_pnl":
            round(
                total_net_pnl,
                2,
            ),

        "total_fees":
            round(
                total_fees,
                2,
            ),

        "total_slippage_cost":
            round(
                total_slippage_cost,
                2,
            ),

        "profit_factor":
            (
                round(
                    profit_factor,
                    3,
                )
                if profit_factor is not None
                else None
            ),

        "average_trade_return_pct":
            round(
                mean(returns),
                2,
            )
            if returns
            else 0.0,

        "average_holding_days":
            round(
                mean(holding_periods),
                2,
            )
            if holding_periods
            else 0.0,

        "best_trade_return_pct":
            round(
                max(returns),
                2,
            )
            if returns
            else 0.0,

        "worst_trade_return_pct":
            round(
                min(returns),
                2,
            )
            if returns
            else 0.0,

        "has_open_position":
            open_position is not None,

        "first_date":
            df.iloc[0][
                "Date"
            ].strftime("%Y-%m-%d"),

        "last_date":
            df.iloc[-1][
                "Date"
            ].strftime("%Y-%m-%d"),
    }

    return {
        "summary": summary,
        "trades": trades,
        "open_position": open_position,
    }


if __name__ == "__main__":
    result = analyze_trades()

    print("=== TRADE ANALYTICS ===")

    for key, value in (
        result["summary"].items()
    ):
        print(
            f"{key}: {value}"
        )

    print("\nTRADES")

    for trade in result["trades"]:
        print(
            f"#{trade['trade_number']} | "
            f"{trade['entry_date']} → "
            f"{trade['exit_date']} | "
            f"{trade['result']} | "
            f"P&L: ${trade['net_pnl']} | "
            f"Return: {trade['return_pct']}% | "
            f"Days: {trade['holding_days']}"
        )

    if result["open_position"]:
        print("\nOPEN POSITION")

        for key, value in (
            result[
                "open_position"
            ].items()
        ):
            print(
                f"{key}: {value}"
            )
