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
