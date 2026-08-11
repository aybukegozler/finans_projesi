import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DATABASE_PATH = ROOT_DIR / "data" / "test_quant_app.db"
SIGNALS_PATH = ROOT_DIR / "data" / "signals.csv"

TEST_DATABASE_PATH.unlink(missing_ok=True)


# Uygulama import edilmeden ÖNCE test environment değişkenleri tanımlanır.
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DATABASE_PATH}"
os.environ["SECRET_KEY"] = (
    "automated-test-secret-key-not-used-in-production-2026"
)
os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "10"

os.environ["ADMIN_USERNAME"] = "test_admin"
os.environ["ADMIN_PASSWORD"] = "TestAdmin2026!"

os.environ["USER_USERNAME"] = "test_user"
os.environ["USER_PASSWORD"] = "TestUser2026!"


from src.api import app  # noqa: E402
from src.database import engine  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

    engine.dispose()
    TEST_DATABASE_PATH.unlink(missing_ok=True)


def login(client, username, password):
    response = client.post(
        "/token",
        data={
            "username": username,
            "password": password,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "access_token" in body
    assert body["token_type"] == "bearer"

    return body


def auth_header(token):
    return {
        "Authorization": f"Bearer {token}",
    }


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["engine_exists"] is True
    assert body["signals_exists"] is True
    assert body["database_connected"] is True
    assert body["database_backend"] == "sqlite"


def test_api_requires_authentication(client):
    response = client.get("/api/data")

    assert response.status_code == 401


def test_wrong_password_is_rejected(client):
    response = client.post(
        "/token",
        data={
            "username": os.environ["ADMIN_USERNAME"],
            "password": "wrong-password",
        },
    )

    assert response.status_code == 401


def test_normal_user_can_read_data_without_running_engine(client):
    login_response = login(
        client,
        os.environ["USER_USERNAME"],
        os.environ["USER_PASSWORD"],
    )

    assert login_response["role"] == "user"
    assert SIGNALS_PATH.exists()

    before = SIGNALS_PATH.stat().st_mtime_ns

    response = client.get(
        "/api/data",
        headers=auth_header(login_response["access_token"]),
    )

    after = SIGNALS_PATH.stat().st_mtime_ns

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0

    # USER yalnızca mevcut çıktıyı okumalı.
    assert after == before


def test_admin_runs_cpp_engine(client):
    login_response = login(
        client,
        os.environ["ADMIN_USERNAME"],
        os.environ["ADMIN_PASSWORD"],
    )

    assert login_response["role"] == "admin"
    assert SIGNALS_PATH.exists()

    before = SIGNALS_PATH.stat().st_mtime_ns

    time.sleep(1.1)

    response = client.get(
        "/api/data",
        headers=auth_header(login_response["access_token"]),
    )

    after = SIGNALS_PATH.stat().st_mtime_ns

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0

    # ADMIN isteği C++ motorunu yeniden çalıştırmalı.
    assert after > before



def test_backtest_requires_authentication(client):
    response = client.get("/api/backtest")

    assert response.status_code == 401


def test_normal_user_can_run_backtest(client):
    login_response = login(
        client,
        os.environ["USER_USERNAME"],
        os.environ["USER_PASSWORD"],
    )

    response = client.get(
        "/api/backtest?initial_capital=10000",
        headers=auth_header(
            login_response["access_token"]
        ),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["strategy"] == "SMA20/SMA50 Crossover"
    assert body["requested_by_role"] == "user"

    assert "summary" in body
    assert "equity_curve" in body

    summary = body["summary"]

    assert summary["initial_capital"] == 10000.0
    assert summary["final_value"] > 0
    assert summary["closed_trades"] >= 0

    assert isinstance(
        body["equity_curve"],
        list,
    )

    assert len(body["equity_curve"]) > 0


def test_backtest_rejects_invalid_capital(client):
    login_response = login(
        client,
        os.environ["USER_USERNAME"],
        os.environ["USER_PASSWORD"],
    )

    response = client.get(
        "/api/backtest?initial_capital=0",
        headers=auth_header(
            login_response["access_token"]
        ),
    )

    assert response.status_code == 400



def test_frontend_contains_backtest_dashboard(
    client,
):
    response = client.get("/")

    assert response.status_code == 200

    html = response.text

    assert 'id="metric-strategy-return"' in html
    assert 'id="metric-buy-hold"' in html
    assert 'id="metric-drawdown"' in html
    assert 'id="metric-win-rate"' in html
    assert 'id="metric-final-value"' in html
    assert 'id="metric-trades"' in html
    assert 'id="metric-excess-return"' in html
    assert 'id="metric-sharpe"' in html
    assert 'id="metric-volatility"' in html
    assert 'id="metric-total-fees"' in html
    assert 'id="transaction-fee"' in html
    assert 'id="slippage"' in html
    assert 'id="equity-chart-container"' in html
    assert 'id="optimizer-objective"' in html
    assert 'id="optimizer-best-pair"' in html
    assert 'id="optimizer-return"' in html
    assert 'id="optimizer-sharpe"' in html
    assert 'id="optimizer-drawdown"' in html
    assert 'id="optimizer-results-body"' in html
    assert "loadOptimizerData()" in html
    assert "/api/optimize" in html
    assert 'id="walk-forward-objective"' in html
    assert 'id="wf-oos-return"' in html
    assert 'id="wf-benchmark"' in html
    assert 'id="wf-excess"' in html
    assert 'id="wf-median-sharpe"' in html
    assert 'id="walk-forward-results-body"' in html
    assert "loadWalkForwardData()" in html
    assert "/api/walk-forward" in html
    assert 'id="trade-closed-count"' in html
    assert 'id="trade-net-pnl"' in html
    assert 'id="trade-profit-factor"' in html
    assert 'id="position-unrealized-pnl"' in html
    assert 'id="trade-history-body"' in html
    assert "loadTradeAnalytics()" in html
    assert "/api/trades" in html
    assert 'id="live-market-symbol"' in html
    assert 'id="live-market-interval"' in html
    assert 'id="live-market-status"' in html
    assert 'id="live-price"' in html
    assert 'id="live-market-chart"' in html
    assert 'id="live-sma20"' in html
    assert 'id="live-sma50"' in html
    assert 'id="live-signal"' in html
    assert 'id="live-trend"' in html
    assert 'id="live-sma-spread"' in html
    assert 'id="live-crossover"' in html
    assert 'id="live-last-crossover"' in html
    assert 'id="live-24h-change"' in html
    assert 'id="live-24h-high"' in html
    assert 'id="live-24h-low"' in html
    assert 'id="live-24h-volume"' in html
    assert "load24hMarketStats" in html
    assert "scheduleLiveMarketReconnect" in html
    assert "liveVolumeSeries" in html
    assert "addHistogramSeries" in html
    assert "liveSignalMarkers" in html
    assert "calculateCrossoverMarkers" in html
    assert "updateLiveCrossoverMarker" in html
    assert "setMarkers" in html
    assert "arrowUp" in html
    assert "arrowDown" in html
    assert 'id="live-signal-history"' in html
    assert 'id="live-signal-toast"' in html
    assert 'id="live-rsi14"' in html
    assert 'id="live-macd"' in html
    assert 'id="live-macd-histogram"' in html
    assert 'id="live-bb-upper"' in html
    assert 'id="live-bb-middle"' in html
    assert 'id="live-bb-lower"' in html
    assert 'id="live-technical-score"' in html
    assert 'id="live-technical-rating"' in html
    assert "calculateBollingerSeries" in html
    assert "updateLiveTechnicalAnalysis" in html
    assert "renderLiveSignalHistory" in html
    assert "recordConfirmedCrossover" in html
    assert "showLiveSignalToast" in html
    assert "clearLiveSignalHistory" in html
    assert "priceScaleId: 'volume'" in html
    assert "/api/market/ticker/24h" in html
    assert "startLiveMarket()" in html
    assert "/api/market/klines" in html
    assert "/ws/market/" in html
    assert "loadBacktestData()" in html
    assert "/api/backtest" in html


def test_backtest_accepts_market_costs(
    client,
):
    login_response = login(
        client,
        os.environ["USER_USERNAME"],
        os.environ["USER_PASSWORD"],
    )

    response = client.get(
        (
            "/api/backtest"
            "?initial_capital=10000"
            "&transaction_fee_pct=0.10"
            "&slippage_pct=0.05"
        ),
        headers=auth_header(
            login_response["access_token"]
        ),
    )

    assert response.status_code == 200

    summary = (
        response.json()["summary"]
    )

    assert (
        summary["transaction_fee_pct"]
        == 0.10
    )

    assert (
        summary["slippage_pct"]
        == 0.05
    )

    assert "sharpe_ratio" in summary
    assert "excess_return_pct" in summary
    assert "total_fees_paid" in summary


def test_optimizer_requires_authentication(client):
    response = client.get("/api/optimize")

    assert response.status_code == 401


def test_authenticated_user_can_optimize_strategy(client):
    login_response = login(
        client,
        os.environ["USER_USERNAME"],
        os.environ["USER_PASSWORD"],
    )

    response = client.get(
        (
            "/api/optimize"
            "?objective=sharpe_ratio"
            "&top_n=5"
            "&initial_capital=10000"
            "&transaction_fee_pct=0.10"
            "&slippage_pct=0.05"
        ),
        headers=auth_header(
            login_response["access_token"]
        ),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["strategy_family"] == "SMA Crossover"
    assert body["requested_by_role"] == "user"
    assert body["objective"] == "sharpe_ratio"

    assert body["tested_combinations"] > 0
    assert len(body["top_results"]) == 5

    assert (
        body["best"]
        == body["top_results"][0]
    )

    assert (
        body["best"]["short_window"]
        < body["best"]["long_window"]
    )


def test_optimizer_rejects_invalid_objective(client):
    login_response = login(
        client,
        os.environ["USER_USERNAME"],
        os.environ["USER_PASSWORD"],
    )

    response = client.get(
        "/api/optimize?objective=magic_metric",
        headers=auth_header(
            login_response["access_token"]
        ),
    )

    assert response.status_code == 400


def test_walk_forward_requires_authentication(client):
    response = client.get("/api/walk-forward")

    assert response.status_code == 401


def test_authenticated_user_can_run_walk_forward(
    client,
    monkeypatch,
):
    def fake_walk_forward_validate(**kwargs):
        return {
            "method": "Expanding Window Walk-Forward",
            "objective": kwargs["objective"],
            "summary": {
                "folds": 5,
                "initial_capital": 10000.0,
                "final_value": 10935.42,
                "out_of_sample_return_pct": 9.35,
                "benchmark_final_value": 14808.0,
                "benchmark_return_pct": 48.08,
                "excess_return_pct": -38.72,
                "profitable_folds": 2,
                "profitable_fold_rate_pct": 40.0,
                "average_test_return_pct": 1.87,
                "median_test_return_pct": 0.0,
                "average_test_sharpe": 1.236,
                "median_test_sharpe": 0.0,
                "total_closed_trades": 2,
                "zero_closed_trade_folds": 3,
                "worst_fold_drawdown_pct": 0.8,
                "most_selected_pair": {
                    "short_window": 30,
                    "long_window": 60,
                    "times_selected": 4,
                    "selection_rate_pct": 80.0,
                },
            },
            "folds": [],
        }

    monkeypatch.setattr(
        "src.api.walk_forward_validate",
        fake_walk_forward_validate,
    )

    login_response = login(
        client,
        os.environ["USER_USERNAME"],
        os.environ["USER_PASSWORD"],
    )

    response = client.get(
        (
            "/api/walk-forward"
            "?objective=sharpe_ratio"
            "&initial_train_size=250"
            "&test_size=50"
        ),
        headers=auth_header(
            login_response["access_token"]
        ),
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["method"]
        == "Expanding Window Walk-Forward"
    )

    assert (
        body["summary"]["out_of_sample_return_pct"]
        == 9.35
    )

    assert (
        body["summary"]["median_test_sharpe"]
        == 0.0
    )


def test_walk_forward_rejects_invalid_objective(
    client,
):
    login_response = login(
        client,
        os.environ["USER_USERNAME"],
        os.environ["USER_PASSWORD"],
    )

    response = client.get(
        "/api/walk-forward?objective=magic_metric",
        headers=auth_header(
            login_response["access_token"]
        ),
    )

    assert response.status_code == 400


def test_trade_analytics_requires_authentication(
    client,
):
    response = client.get("/api/trades")

    assert response.status_code == 401


def test_authenticated_user_can_read_trade_analytics(
    client,
    monkeypatch,
):
    def fake_analyze_trades(**kwargs):
        return {
            "summary": {
                "closed_trades": 6,
                "winning_trades": 2,
                "losing_trades": 4,
                "win_rate_pct": 33.33,
                "total_net_pnl": 979.68,
                "total_fees": 113.35,
                "total_slippage_cost": 56.67,
                "profit_factor": 1.421,
                "average_trade_return_pct": 2.25,
                "average_holding_days": 59.5,
                "best_trade_return_pct": 23.18,
                "worst_trade_return_pct": -9.34,
                "has_open_position": True,
                "first_date": "2024-08-05",
                "last_date": "2026-08-04",
            },
            "trades": [
                {
                    "trade_number": 1,
                    "entry_date": "2024-12-03",
                    "exit_date": "2025-01-27",
                    "holding_days": 55,
                    "market_entry_price": 100.0,
                    "effective_entry_price": 100.05,
                    "market_exit_price": 95.0,
                    "effective_exit_price": 94.95,
                    "shares": 99.0,
                    "gross_pnl": -500.0,
                    "net_pnl": -555.47,
                    "return_pct": -5.55,
                    "buy_fee": 10.0,
                    "sell_fee": 9.4,
                    "total_fees": 19.4,
                    "slippage_cost": 9.8,
                    "total_costs": 29.2,
                    "result": "LOSS",
                    "forced_exit": False,
                }
            ],
            "open_position": {
                "entry_date": "2026-07-09",
                "current_date": "2026-08-04",
                "holding_days": 26,
                "market_entry_price": 316.22,
                "current_market_price": 309.38,
                "shares": 34.6696,
                "estimated_liquidation_value": 10710.0,
                "unrealized_pnl": -269.67,
                "unrealized_return_pct": -2.46,
            },
        }

    monkeypatch.setattr(
        "src.api.analyze_trades",
        fake_analyze_trades,
    )

    login_response = login(
        client,
        os.environ["USER_USERNAME"],
        os.environ["USER_PASSWORD"],
    )

    response = client.get(
        (
            "/api/trades"
            "?initial_capital=10000"
            "&transaction_fee_pct=0.10"
            "&slippage_pct=0.05"
        ),
        headers=auth_header(
            login_response["access_token"]
        ),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["strategy"] == "SMA20/SMA50"
    assert body["summary"]["closed_trades"] == 6
    assert body["summary"]["win_rate_pct"] == 33.33
    assert body["summary"]["total_net_pnl"] == 979.68
    assert len(body["trades"]) == 1
    assert body["open_position"] is not None


def test_trade_analytics_rejects_invalid_cost(
    client,
):
    login_response = login(
        client,
        os.environ["USER_USERNAME"],
        os.environ["USER_PASSWORD"],
    )

    response = client.get(
        "/api/trades?transaction_fee_pct=-1",
        headers=auth_header(
            login_response["access_token"]
        ),
    )

    assert response.status_code == 400


def test_market_klines_requires_authentication(
    client,
):
    response = client.get(
        "/api/market/klines"
    )

    assert response.status_code == 401


def test_authenticated_user_can_get_market_klines(
    client,
    monkeypatch,
):
    def fake_get_klines(
        symbol,
        interval,
        limit,
    ):
        assert symbol == "BTCUSDT"
        assert interval == "1m"
        assert limit == 2

        return [
            {
                "open_time":
                    "2026-08-10T06:00:00+00:00",

                "open_time_ms":
                    1786341600000,

                "open":
                    100.0,

                "high":
                    110.0,

                "low":
                    95.0,

                "close":
                    105.0,

                "volume":
                    10.0,

                "close_time_ms":
                    1786341659999,

                "quote_volume":
                    1030.0,

                "trade_count":
                    25,
            },
            {
                "open_time":
                    "2026-08-10T06:01:00+00:00",

                "open_time_ms":
                    1786341660000,

                "open":
                    105.0,

                "high":
                    112.0,

                "low":
                    103.0,

                "close":
                    108.0,

                "volume":
                    12.0,

                "close_time_ms":
                    1786341719999,

                "quote_volume":
                    1290.0,

                "trade_count":
                    30,
            },
        ]

    monkeypatch.setattr(
        "src.api.get_klines",
        fake_get_klines,
    )

    login_response = login(
        client,
        os.environ["USER_USERNAME"],
        os.environ["USER_PASSWORD"],
    )

    response = client.get(
        (
            "/api/market/klines"
            "?symbol=btc/usdt"
            "&interval=1m"
            "&limit=2"
        ),
        headers=auth_header(
            login_response["access_token"]
        ),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["source"] == "Binance Spot"
    assert body["symbol"] == "BTCUSDT"
    assert body["interval"] == "1m"
    assert body["count"] == 2

    assert (
        body["candles"][0]["close"]
        == 105.0
    )


def test_market_klines_rejects_invalid_interval(
    client,
):
    login_response = login(
        client,
        os.environ["USER_USERNAME"],
        os.environ["USER_PASSWORD"],
    )

    response = client.get(
        (
            "/api/market/klines"
            "?symbol=BTCUSDT"
            "&interval=7minutes"
        ),
        headers=auth_header(
            login_response["access_token"]
        ),
    )

    assert response.status_code == 400


def test_live_market_websocket(
    client,
    monkeypatch,
):
    def fake_get_klines(
        symbol,
        interval,
        limit,
    ):
        assert symbol == "BTCUSDT"
        assert interval == "1m"
        assert limit == 100

        base_time = 1786338000000

        return [
            {
                "open_time":
                    "2026-08-10T00:00:00+00:00",

                "open_time_ms":
                    base_time
                    + index * 60_000,

                "open":
                    100.0,

                "high":
                    100.0,

                "low":
                    100.0,

                "close":
                    100.0,

                "volume":
                    1.0,

                "close_time_ms":
                    base_time
                    + index * 60_000
                    + 59_999,

                "quote_volume":
                    100.0,

                "trade_count":
                    1,
            }
            for index in range(60)
        ]

    async def fake_stream_klines(
        symbol,
        interval,
    ):
        assert symbol == "BTCUSDT"
        assert interval == "1m"

        yield {
            "event": "kline",
            "symbol": "BTCUSDT",
            "event_time_ms":
                1786341601000,

            "event_time":
                "2026-08-10T06:00:01+00:00",

            "interval": "1m",

            "open_time_ms":
                1786341600000,

            "close_time_ms":
                1786341659999,

            "open": 100.0,
            "high": 110.0,
            "low": 95.0,
            "close": 105.0,
            "volume": 10.5,
            "quote_volume": 1075.0,
            "trade_count": 42,
            "closed": False,
        }

    monkeypatch.setattr(
        "src.api.get_klines",
        fake_get_klines,
    )

    monkeypatch.setattr(
        "src.api.stream_klines",
        fake_stream_klines,
    )

    with client.websocket_connect(
        "/ws/market/BTCUSDT?interval=1m"
    ) as websocket:
        connected = (
            websocket.receive_json()
        )

        assert (
            connected["type"]
            == "connected"
        )

        assert (
            connected["symbol"]
            == "BTCUSDT"
        )

        assert (
            connected["indicators"][
                "ready"
            ]
            is True
        )

        message = (
            websocket.receive_json()
        )

        assert (
            message["type"]
            == "kline"
        )

        assert (
            message["data"]["close"]
            == 105.0
        )

        assert (
            message["data"][
                "trade_count"
            ]
            == 42
        )

        assert (
            message["indicators"][
                "signal"
            ]
            == "BUY"
        )

        assert (
            message["indicators"][
                "crossover"
            ]
            == "BUY"
        )

        assert (
            message["indicators"][
                "last_crossover"
            ]
            is None
        )

        assert (
            message["technical"][
                "ready"
            ]
            is True
        )

        assert (
            "rsi14"
            in message["technical"]
        )

        assert (
            "macd"
            in message["technical"]
        )

        assert (
            "bollinger"
            in message["technical"]
        )

        assert (
            "rating"
            in message["technical"]
        )

        assert (
            "interpretation"
            in message
        )

        assert (
            message["interpretation"][
                "ready"
            ]
            is True
        )

        assert (
            "state"
            in message["interpretation"]
        )

        assert (
            "confidence"
            in message["interpretation"]
        )

        assert (
            "important"
            in message["interpretation"]
        )

        assert (
            "low_relevance"
            in message["interpretation"]
        )

        assert (
            "explanation"
            in message["interpretation"]
        )


def test_market_24h_ticker_requires_authentication(
    client,
):
    response = client.get(
        "/api/market/ticker/24h"
    )

    assert response.status_code == 401


def test_authenticated_user_can_get_24h_ticker(
    client,
    monkeypatch,
):
    def fake_get_24h_ticker(
        symbol,
    ):
        assert symbol == "BTCUSDT"

        return {
            "symbol": "BTCUSDT",
            "price_change": 1500.0,
            "price_change_pct": 2.35,
            "weighted_average_price": 65000.0,
            "previous_close": 64000.0,
            "last_price": 65500.0,
            "open_price": 64000.0,
            "high_price": 66000.0,
            "low_price": 63500.0,
            "volume": 12345.0,
            "quote_volume": 800000000.0,
            "trade_count": 100000,
            "open_time_ms": 1786300000000,
            "close_time_ms": 1786386400000,
        }

    monkeypatch.setattr(
        "src.api.get_24h_ticker",
        fake_get_24h_ticker,
    )

    login_response = login(
        client,
        os.environ["USER_USERNAME"],
        os.environ["USER_PASSWORD"],
    )

    response = client.get(
        "/api/market/ticker/24h?symbol=BTCUSDT",
        headers=auth_header(
            login_response["access_token"]
        ),
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["source"]
        == "Binance Spot"
    )

    assert (
        body["ticker"]["symbol"]
        == "BTCUSDT"
    )

    assert (
        body["ticker"][
            "price_change_pct"
        ]
        == 2.35
    )
