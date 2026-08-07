from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SIGNALS_PATH = ROOT_DIR / "data" / "signals.csv"

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    initial_capital: float
    final_value: float

    total_return_pct: float
    buy_hold_return_pct: float
    excess_return_pct: float

    closed_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float

    max_drawdown_pct: float
    sharpe_ratio: float
    annualized_volatility_pct: float

    transaction_fee_pct: float
    slippage_pct: float
    total_fees_paid: float

    first_date: str
    last_date: str


def calculate_max_drawdown(
    equity_curve: list[float],
) -> float:
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_drawdown = 0.0

    for value in equity_curve:
        peak = max(peak, value)

        if peak <= 0:
            continue

        drawdown = (value - peak) / peak
        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

    return abs(max_drawdown) * 100.0


def calculate_risk_metrics(
    equity_curve: list[float],
) -> tuple[float, float]:
    """
    Risk-free rate 0 varsayımıyla yıllıklaştırılmış
    Sharpe Ratio ve volatilite hesaplar.
    """

    if len(equity_curve) < 2:
        return 0.0, 0.0

    series = pd.Series(
        equity_curve,
        dtype="float64",
    )

    returns = (
        series
        .pct_change()
        .dropna()
    )

    if returns.empty:
        return 0.0, 0.0

    daily_std = float(
        returns.std(ddof=1)
    )

    if pd.isna(daily_std) or daily_std == 0:
        return 0.0, 0.0

    daily_mean = float(
        returns.mean()
    )

    annualized_volatility = (
        daily_std
        * sqrt(TRADING_DAYS_PER_YEAR)
        * 100.0
    )

    sharpe_ratio = (
        daily_mean
        / daily_std
        * sqrt(TRADING_DAYS_PER_YEAR)
    )

    return (
        sharpe_ratio,
        annualized_volatility,
    )


def run_backtest(
    signals_path: Path | str = DEFAULT_SIGNALS_PATH,
    initial_capital: float = 10_000.0,
    transaction_fee_pct: float = 0.0,
    slippage_pct: float = 0.0,
) -> dict:
    signals_path = Path(signals_path)

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

    if not signals_path.exists():
        raise FileNotFoundError(
            f"Signal file not found: {signals_path}"
        )

    df = pd.read_csv(
        signals_path
    )

    required_columns = {
        "Date",
        "Close",
        "Signal",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    df = df.copy()

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    df["Signal"] = pd.to_numeric(
        df["Signal"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Date",
            "Close",
            "Signal",
        ]
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

    cash = float(
        initial_capital
    )

    shares = 0.0

    position_entry_value: float | None = None

    closed_trades = 0
    winning_trades = 0
    losing_trades = 0

    total_fees_paid = 0.0

    equity_curve: list[float] = []
    equity_points: list[dict] = []

    for row in df.itertuples(
        index=False
    ):
        market_price = float(
            row.Close
        )

        signal = int(
            row.Signal
        )

        if market_price <= 0:
            continue

        # BUY
        if (
            signal == 1
            and shares == 0
        ):
            effective_buy_price = (
                market_price
                * (1.0 + slippage_rate)
            )

            available_before_trade = cash

            shares = (
                cash
                / (
                    effective_buy_price
                    * (1.0 + fee_rate)
                )
            )

            trade_notional = (
                shares
                * effective_buy_price
            )

            buy_fee = (
                trade_notional
                * fee_rate
            )

            total_fees_paid += buy_fee

            position_entry_value = (
                available_before_trade
            )

            cash = 0.0

        # SELL
        elif (
            signal == -1
            and shares > 0
        ):
            effective_sell_price = (
                market_price
                * (1.0 - slippage_rate)
            )

            gross_exit_value = (
                shares
                * effective_sell_price
            )

            sell_fee = (
                gross_exit_value
                * fee_rate
            )

            total_fees_paid += sell_fee

            exit_value = (
                gross_exit_value
                - sell_fee
            )

            if (
                position_entry_value is not None
                and exit_value
                > position_entry_value
            ):
                winning_trades += 1
            else:
                losing_trades += 1

            closed_trades += 1

            cash = exit_value
            shares = 0.0
            position_entry_value = None

        equity = (
            cash
            + shares * market_price
        )

        equity_curve.append(
            equity
        )

        equity_points.append(
            {
                "date": str(
                    row.Date
                ),
                "equity": round(
                    equity,
                    2,
                ),
            }
        )

    last_price = float(
        df.iloc[-1]["Close"]
    )

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

    first_price = float(
        df.iloc[0]["Close"]
    )

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

    result = BacktestResult(
        initial_capital=round(
            initial_capital,
            2,
        ),
        final_value=round(
            final_value,
            2,
        ),
        total_return_pct=round(
            total_return_pct,
            2,
        ),
        buy_hold_return_pct=round(
            buy_hold_return_pct,
            2,
        ),
        excess_return_pct=round(
            excess_return_pct,
            2,
        ),
        closed_trades=closed_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate_pct=round(
            win_rate_pct,
            2,
        ),
        max_drawdown_pct=round(
            max_drawdown_pct,
            2,
        ),
        sharpe_ratio=round(
            sharpe_ratio,
            3,
        ),
        annualized_volatility_pct=round(
            annualized_volatility_pct,
            2,
        ),
        transaction_fee_pct=round(
            transaction_fee_pct,
            4,
        ),
        slippage_pct=round(
            slippage_pct,
            4,
        ),
        total_fees_paid=round(
            total_fees_paid,
            2,
        ),
        first_date=str(
            df.iloc[0]["Date"]
        ),
        last_date=str(
            df.iloc[-1]["Date"]
        ),
    )

    return {
        "summary": asdict(
            result
        ),
        "equity_curve": equity_points,
    }


if __name__ == "__main__":
    result = run_backtest(
        initial_capital=10_000,
        transaction_fee_pct=0.10,
        slippage_pct=0.05,
    )

    print(
        "=== SMA20 / SMA50 BACKTEST ==="
    )

    for key, value in (
        result["summary"].items()
    ):
        print(
            f"{key}: {value}"
        )
