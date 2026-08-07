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
