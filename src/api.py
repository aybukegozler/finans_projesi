from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
import pandas as pd
import subprocess
import os
from contextlib import asynccontextmanager
from pathlib import Path
import bcrypt

# Oluşturduğumuz veritabanı modülünü içe aktarıyoruz
from src.database import (
    SessionLocal,
    User,
    check_database_connection,
    create_initial_users,
    get_database_backend,
)

from src.backtest import run_backtest
from src.optimizer import optimize_sma_strategy
from src.walk_forward import walk_forward_validate
from fastapi import WebSocket as FastAPIWebSocket, WebSocketDisconnect as FastAPIWebSocketDisconnect
from src.trade_analytics import analyze_trades
from src.live_signal import LiveSMAEngine
from src.technical_indicators import calculate_technical_snapshot
from src.binance_market import (
    get_klines,
    get_24h_ticker,
    normalize_symbol,
    stream_klines,
    validate_interval,
)

# --- UYGULAMA VE GÜVENLİK AYARLARI ---
ROOT_DIR = Path(__file__).resolve().parent.parent
ENGINE_PATH = ROOT_DIR / "src" / "engine"
SIGNALS_PATH = ROOT_DIR / "data" / "signals.csv"

SECRET_KEY = os.getenv("SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY environment variable is required."
    )

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    prepared_users = create_initial_users()

    print(
        f"Hazırlanan environment kullanıcısı: "
        f"{prepared_users}"
    )

    print(
        f"Database backend: {get_database_backend()}"
    )

    yield


app = FastAPI(
    title="Secure Quant Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- VERİTABANI BAĞLANTISI ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- YETKİLENDİRME (AUTH) FONKSİYONLARI ---
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Geçersiz kimlik doğrulama bilgileri",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# --- API UÇ NOKTALARI ---

@app.get("/health")
def health_check():
    engine_exists = ENGINE_PATH.exists()
    signals_exists = SIGNALS_PATH.exists()
    database_connected = check_database_connection()

    checks = {
        "engine_exists": engine_exists,
        "signals_exists": signals_exists,
        "database_connected": database_connected,
        "database_backend": get_database_backend(),
    }

    if not (
        engine_exists
        and signals_exists
        and database_connected
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "degraded",
                **checks,
            },
        )

    return {
        "status": "ok",
        **checks,
    }


# 1. Kullanıcı Giriş (Login) Rotası
@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı adı veya şifre hatalı",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Başarılı girişte Token üretiyoruz (İçine rol bilgisini de ekliyoruz)
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

# 2. Korunan Veri Rotası (RBAC Mimarisi Burada Çalışıyor)
@app.get("/api/data")
def get_market_data(current_user: User = Depends(get_current_user)):
    
    # Sadece admin C++ motorunu yeniden çalıştırabilir.
    if current_user.role == "admin":
        try:
            result = subprocess.run(
                [str(ENGINE_PATH)],
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )

            print(
                f"[{current_user.username}] "
                f"C++ motoru çalıştırıldı. "
                f"{result.stdout.strip()}"
            )

        except FileNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="C++ hesaplama motoru bulunamadı.",
            ) from error

        except subprocess.TimeoutExpired as error:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="C++ motoru zaman aşımına uğradı.",
            ) from error

        except subprocess.CalledProcessError as error:
            print(
                f"C++ motor hatası: {error.stderr}"
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Finansal hesaplama motoru "
                    "çalıştırılamadı."
                ),
            ) from error

    else:
        print(
            f"[{current_user.username}] "
            "Mevcut sonuçları okuyor."
        )

    if not SIGNALS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sinyal dosyası henüz oluşturulmadı.",
        )

    df = pd.read_csv(SIGNALS_PATH)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['SMA20'] = pd.to_numeric(df['SMA20'], errors='coerce')
    df['SMA50'] = pd.to_numeric(df['SMA50'], errors='coerce')
    df['Signal'] = pd.to_numeric(df['Signal'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce').dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=['Date', 'Close'])
    df = df.drop_duplicates(subset=['Date']).sort_values('Date')
    df = df.fillna(0) 
    
    data_list = df.to_dict(orient="records")
    if len(data_list) == 0:
         return {"error": "Veri var ama tablo boş!"}
    return data_list

# 3. Backtest Analiz Rotası
@app.get("/api/backtest")
def get_backtest_results(
    initial_capital: float = 10_000.0,
    transaction_fee_pct: float = 0.10,
    slippage_pct: float = 0.05,
    current_user: User = Depends(get_current_user),
):
    if initial_capital <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Başlangıç sermayesi sıfırdan büyük olmalıdır.",
        )

    if initial_capital > 100_000_000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Başlangıç sermayesi çok yüksek.",
        )

    if not SIGNALS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backtest için sinyal dosyası bulunamadı.",
        )

    try:
        result = run_backtest(
            signals_path=SIGNALS_PATH,
            initial_capital=initial_capital,
            transaction_fee_pct=transaction_fee_pct,
            slippage_pct=slippage_pct,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backtest veri dosyası bulunamadı.",
        ) from error

    except Exception as error:
        print(
            "Backtest hatası: "
            f"{type(error).__name__}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Backtest hesaplaması tamamlanamadı.",
        ) from error

    return {
        "strategy": "SMA20/SMA50 Crossover",
        "requested_by_role": current_user.role,
        **result,
    }


# Strategy Optimizer
@app.get("/api/optimize")
def optimize_strategy(
    objective: str = "sharpe_ratio",
    top_n: int = 5,
    initial_capital: float = 10_000.0,
    transaction_fee_pct: float = 0.10,
    slippage_pct: float = 0.05,
    current_user: User = Depends(get_current_user),
):
    allowed_objectives = {
        "sharpe_ratio",
        "total_return_pct",
        "excess_return_pct",
    }

    if objective not in allowed_objectives:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Geçersiz optimizasyon metriği.",
        )

    if not 1 <= top_n <= 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_n 1 ile 20 arasında olmalıdır.",
        )

    if initial_capital <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Başlangıç sermayesi sıfırdan büyük olmalıdır.",
        )

    try:
        result = optimize_sma_strategy(
            initial_capital=initial_capital,
            transaction_fee_pct=transaction_fee_pct,
            slippage_pct=slippage_pct,
            objective=objective,
            top_n=top_n,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market verisi bulunamadı.",
        ) from error

    except Exception as error:
        print(
            "Optimizer hatası: "
            f"{type(error).__name__}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Strateji optimizasyonu tamamlanamadı.",
        ) from error

    return {
        "strategy_family": "SMA Crossover",
        "requested_by_role": current_user.role,
        **result,
    }


# Walk-Forward Validation
@app.get("/api/walk-forward")
def walk_forward_analysis(
    objective: str = "sharpe_ratio",
    initial_train_size: int = 250,
    test_size: int = 50,
    initial_capital: float = 10_000.0,
    transaction_fee_pct: float = 0.10,
    slippage_pct: float = 0.05,
    current_user: User = Depends(get_current_user),
):
    allowed_objectives = {
        "sharpe_ratio",
        "total_return_pct",
        "excess_return_pct",
    }

    if objective not in allowed_objectives:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Geçersiz validation metriği.",
        )

    if initial_capital <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Başlangıç sermayesi sıfırdan büyük olmalıdır.",
        )

    if not 101 <= initial_train_size <= 450:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Training size 101 ile 450 arasında olmalıdır.",
        )

    if not 10 <= test_size <= 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Test size 10 ile 100 arasında olmalıdır.",
        )

    try:
        result = walk_forward_validate(
            initial_train_size=initial_train_size,
            test_size=test_size,
            initial_capital=initial_capital,
            transaction_fee_pct=transaction_fee_pct,
            slippage_pct=slippage_pct,
            objective=objective,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market verisi bulunamadı.",
        ) from error

    except Exception as error:
        print(
            "Walk-forward hatası: "
            f"{type(error).__name__}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Walk-forward validation tamamlanamadı.",
        ) from error

    return {
        "requested_by_role": current_user.role,
        **result,
    }


# Trade Analytics
@app.get("/api/trades")
def trade_analytics(
    initial_capital: float = 10_000.0,
    transaction_fee_pct: float = 0.10,
    slippage_pct: float = 0.05,
    force_close_at_end: bool = False,
    current_user: User = Depends(get_current_user),
):
    if initial_capital <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Başlangıç sermayesi sıfırdan büyük olmalıdır.",
        )

    if not 0 <= transaction_fee_pct <= 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="İşlem ücreti yüzde 0 ile 5 arasında olmalıdır.",
        )

    if not 0 <= slippage_pct <= 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Slippage yüzde 0 ile 5 arasında olmalıdır.",
        )

    try:
        result = analyze_trades(
            initial_capital=initial_capital,
            transaction_fee_pct=transaction_fee_pct,
            slippage_pct=slippage_pct,
            force_close_at_end=force_close_at_end,
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Signal verisi bulunamadı.",
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except Exception as error:
        print(
            "Trade analytics hatası: "
            f"{type(error).__name__}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trade analytics tamamlanamadı.",
        ) from error

    return {
        "strategy": "SMA20/SMA50",
        "requested_by_role": current_user.role,
        "initial_capital": initial_capital,
        "transaction_fee_pct": transaction_fee_pct,
        "slippage_pct": slippage_pct,
        **result,
    }


# Binance Historical Market Data
@app.get("/api/market/klines")
def market_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    limit: int = 500,
    current_user = Depends(get_current_user),
):
    try:
        normalized_symbol = normalize_symbol(
            symbol
        )

        validated_interval = validate_interval(
            interval
        )

        candles = get_klines(
            symbol=normalized_symbol,
            interval=validated_interval,
            limit=limit,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except Exception as error:
        print(
            "Binance REST hatası:",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Binance market verisine ulaşılamadı.",
        ) from error

    return {
        "source": "Binance Spot",
        "symbol": normalized_symbol,
        "interval": validated_interval,
        "count": len(candles),
        "candles": candles,
    }


# Binance 24h Market Statistics
@app.get("/api/market/ticker/24h")
def market_24h_ticker(
    symbol: str = "BTCUSDT",
    current_user = Depends(get_current_user),
):
    try:
        normalized_symbol = normalize_symbol(
            symbol
        )

        ticker = get_24h_ticker(
            normalized_symbol
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except Exception as error:
        print(
            "Binance 24h ticker hatası:",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Binance 24h market verisine "
                "ulaşılamadı."
            ),
        ) from error

    return {
        "source": "Binance Spot",
        "ticker": ticker,
    }


# Binance Live Market WebSocket
@app.websocket("/ws/market/{symbol}")
async def market_websocket(
    websocket: FastAPIWebSocket,
    symbol: str,
    interval: str = "1m",
):
    import asyncio
    import time

    await websocket.accept()

    try:
        normalized_symbol = normalize_symbol(
            symbol
        )

        validated_interval = validate_interval(
            interval
        )

    except ValueError as error:
        await websocket.send_json(
            {
                "type": "error",
                "detail": str(error),
            }
        )

        await websocket.close(
            code=1008
        )

        return

    engine = LiveSMAEngine(
        short_window=20,
        long_window=50,
        max_points=250,
    )

    try:
        historical_candles = (
            await asyncio.to_thread(
                get_klines,
                normalized_symbol,
                validated_interval,
                100,
            )
        )

        now_ms = int(
            time.time() * 1000
        )

        seed_candles = [
            {
                "open_time_ms":
                    candle["open_time_ms"],

                "close":
                    candle["close"],

                "closed":
                    candle[
                        "close_time_ms"
                    ] <= now_ms,
            }
            for candle
            in historical_candles
        ]

        indicator_snapshot = (
            engine.seed(
                seed_candles
            )
        )

        technical_snapshot = (
            calculate_technical_snapshot(
                [
                    candle["close"]
                    for candle
                    in engine.candles
                ]
            )
        )

    except Exception as error:
        print(
            "Live SMA seed hatası:",
            type(error).__name__,
        )

        await websocket.send_json(
            {
                "type": "error",
                "detail": (
                    "Canlı SMA başlangıç "
                    "verisi hazırlanamadı."
                ),
            }
        )

        await websocket.close(
            code=1011
        )

        return

    await websocket.send_json(
        {
            "type": "connected",
            "source": "Binance Spot",
            "symbol": normalized_symbol,
            "interval": validated_interval,
            "indicators":
                indicator_snapshot,

            "technical":
                technical_snapshot,
        }
    )

    try:
        async for kline in stream_klines(
            symbol=normalized_symbol,
            interval=validated_interval,
        ):
            indicators = (
                engine.update(
                    kline
                )
            )

            technical_snapshot = (
                calculate_technical_snapshot(
                    [
                        candle["close"]
                        for candle
                        in engine.candles
                    ]
                )
            )

            await websocket.send_json(
                {
                    "type": "kline",
                    "data": kline,
                    "indicators":
                        indicators,

                    "technical":
                        technical_snapshot,
                }
            )

    except FastAPIWebSocketDisconnect:
        return

    except Exception as error:
        print(
            "Market WebSocket hatası:",
            type(error).__name__,
        )

        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "detail": (
                        "Binance canlı bağlantısı "
                        "geçici olarak kullanılamıyor."
                    ),
                }
            )

            await websocket.close(
                code=1011
            )

        except Exception:
            pass


# 3. Ön Yüz (Frontend)
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>Quant Dashboard - Secure Access</title>
        <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body { font-family: 'Segoe UI', Tahoma, sans-serif; background-color: #131722; color: #d1d4dc; margin: 0; padding: 20px; }
            /* Form Stilleri */
            .auth-container { max-width: 350px; margin: 100px auto; background: #1E222D; padding: 30px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
            .auth-container h2 { text-align: center; margin-bottom: 25px; color: #fff; }
            .input-group { margin-bottom: 15px; }
            .input-group input { width: 93%; padding: 12px; background: #2B2B43; color: white; border: 1px solid #434651; border-radius: 6px; outline: none; }
            .btn-login { width: 100%; padding: 12px; background: #2962FF; color: white; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.3s; }
            .btn-login:hover { background: #1E4BD8; }
            #login-error { color: #FF5252; text-align: center; margin-top: 15px; display: none; font-size: 14px; }
            
            /* Dashboard Stilleri */
            .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            .header-text h1 { color: #fff; margin: 0 0 5px 0; }
            .header-text p { color: #8a919e; margin: 0; }
            .role-badge { background: #00E676; color: #131722; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 10px; vertical-align: middle; }
            .btn-logout { padding: 8px 20px; background: #FF5252; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }
            #chart-container {
                width: 100%;
                height: 600px;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            }

            #error-box {
                color: #FF5252;
                text-align: center;
                margin-top: 20px;
                font-weight: bold;
            }

            .section-title {
                color: #fff;
                margin: 32px 0 14px;
                font-size: 20px;
            }

            .backtest-toolbar {
                display: flex;
                gap: 12px;
                align-items: end;
                flex-wrap: wrap;
                margin-bottom: 16px;
            }

            .backtest-control {
                display: flex;
                flex-direction: column;
                gap: 6px;
            }

            .backtest-control label {
                color: #8a919e;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.6px;
            }

            .backtest-control input {
                width: 180px;
                padding: 10px 12px;
                background: #1E222D;
                color: #fff;
                border: 1px solid #434651;
                border-radius: 6px;
                outline: none;
            }

            .btn-backtest {
                padding: 10px 18px;
                background: #2962FF;
                color: #fff;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: bold;
            }

            .btn-backtest:hover {
                background: #1E4BD8;
            }

            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(
                    auto-fit,
                    minmax(170px, 1fr)
                );
                gap: 14px;
                margin-bottom: 18px;
            }

            .metric-card {
                background: #1E222D;
                border: 1px solid #2B2B43;
                border-radius: 10px;
                padding: 18px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            }

            .metric-label {
                color: #8a919e;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.7px;
                text-transform: uppercase;
                margin-bottom: 8px;
            }

            .metric-value {
                color: #fff;
                font-size: 25px;
                font-weight: 700;
            }

            .metric-positive {
                color: #00E676;
            }

            .metric-negative {
                color: #FF5252;
            }

            .metric-neutral {
                color: #d1d4dc;
            }

            .backtest-meta {
                color: #8a919e;
                font-size: 13px;
                margin: 8px 0 14px;
            }

            #equity-chart-container {
                width: 100%;
                height: 340px;
                background: #1E222D;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.35);
            }

            #backtest-error {
                color: #FF5252;
                margin-top: 10px;
                font-weight: bold;
            }

            .optimizer-toolbar {
                display: flex;
                gap: 12px;
                align-items: end;
                flex-wrap: wrap;
                margin-bottom: 16px;
            }

            .optimizer-toolbar select {
                min-width: 190px;
                padding: 10px 12px;
                background: #1E222D;
                color: #fff;
                border: 1px solid #434651;
                border-radius: 6px;
                outline: none;
            }

            .optimizer-table-wrapper {
                overflow-x: auto;
                margin-top: 18px;
                border: 1px solid #2B2B43;
                border-radius: 10px;
            }

            .optimizer-table {
                width: 100%;
                border-collapse: collapse;
                background: #1E222D;
            }

            .optimizer-table th,
            .optimizer-table td {
                padding: 13px 15px;
                text-align: right;
                border-bottom: 1px solid #2B2B43;
                color: #d1d4dc;
            }

            .optimizer-table th {
                color: #8a919e;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .optimizer-table th:nth-child(1),
            .optimizer-table th:nth-child(2),
            .optimizer-table td:nth-child(1),
            .optimizer-table td:nth-child(2) {
                text-align: left;
            }

            .optimizer-table tbody tr:last-child td {
                border-bottom: none;
            }

            .live-market-toolbar {
                display: flex;
                align-items: end;
                gap: 14px;
                flex-wrap: wrap;
                margin-bottom: 18px;
            }

            .live-market-toolbar select {
                min-width: 150px;
                padding: 10px 12px;
                background: #1E222D;
                color: #ffffff;
                border: 1px solid #434651;
                border-radius: 6px;
                outline: none;
            }

            .live-market-status {
                padding: 10px 14px;
                border-radius: 7px;
                font-weight: 800;
                letter-spacing: 0.5px;
            }

            .live-market-status.connected {
                color: #00e676;
                background: rgba(0, 230, 118, 0.08);
                border: 1px solid rgba(0, 230, 118, 0.25);
            }

            .live-market-status.connecting {
                color: #ffca28;
                background: rgba(255, 202, 40, 0.08);
                border: 1px solid rgba(255, 202, 40, 0.25);
            }

            .live-market-status.disconnected {
                color: #ff5252;
                background: rgba(255, 82, 82, 0.08);
                border: 1px solid rgba(255, 82, 82, 0.25);
            }

            .live-chart-card {
                background: #1E222D;
                border: 1px solid #2B2B43;
                border-radius: 12px;
                overflow: hidden;
                margin-top: 18px;
                margin-bottom: 24px;
            }

            .live-market-chart {
                width: 100%;
                height: 480px;
            }

            #live-market-error {
                color: #FF5252;
                margin-top: 10px;
                font-weight: bold;
            }

            .signal-alert-section {
                margin-top: 24px;
                margin-bottom: 28px;
                padding: 22px;
                background: #1E222D;
                border: 1px solid #2B2B43;
                border-radius: 12px;
            }

            .signal-alert-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 16px;
                flex-wrap: wrap;
                margin-bottom: 16px;
            }

            .signal-alert-header .section-title {
                margin-top: 0;
                margin-bottom: 5px;
            }

            .signal-alert-description {
                color: #8a919e;
                font-size: 13px;
            }

            .btn-secondary {
                padding: 9px 14px;
                border-radius: 7px;
                border: 1px solid #434651;
                background: #252936;
                color: #d1d4dc;
                cursor: pointer;
                font-weight: 700;
            }

            .btn-secondary:hover {
                background: #303543;
            }

            .signal-history-list {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }

            .signal-history-empty {
                color: #8a919e;
                padding: 16px 0;
            }

            .signal-history-item {
                display: grid;
                grid-template-columns:
                    90px
                    minmax(100px, 1fr)
                    70px
                    90px
                    minmax(140px, 1fr);

                align-items: center;
                gap: 14px;

                padding: 13px 15px;

                background: #171B26;

                border:
                    1px solid #2B2B43;

                border-radius: 8px;
            }

            .signal-history-time,
            .signal-history-symbol,
            .signal-history-interval,
            .signal-history-price {
                color: #d1d4dc;
            }

            .signal-history-buy {
                color: #00e676;
                font-weight: 900;
            }

            .signal-history-sell {
                color: #ff5252;
                font-weight: 900;
            }

            .signal-toast {
                position: fixed;

                top: 24px;
                right: 24px;

                z-index: 9999;

                min-width: 280px;

                padding: 16px 20px;

                border-radius: 10px;

                background: #1E222D;

                border: 1px solid #434651;

                box-shadow:
                    0 12px 30px
                    rgba(0, 0, 0, 0.35);

                opacity: 0;

                transform:
                    translateY(-15px);

                pointer-events: none;

                transition:
                    opacity 0.2s ease,
                    transform 0.2s ease;
            }

            .signal-toast.visible {
                opacity: 1;

                transform:
                    translateY(0);
            }

            .signal-toast.buy {
                border-color:
                    rgba(
                        0,
                        230,
                        118,
                        0.55
                    );
            }

            .signal-toast.sell {
                border-color:
                    rgba(
                        255,
                        82,
                        82,
                        0.55
                    );
            }

            @media (
                max-width: 900px
            ) {
                .signal-history-item {
                    grid-template-columns:
                        1fr
                        1fr;
                }
            }


            #optimizer-error,
            #walk-forward-error,
            #trade-analytics-error {
                color: #FF5252;
                margin-top: 10px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        
        <!-- 1. GİRİŞ EKRANI -->
        <div id="login-section" class="auth-container">
            <h2>Sisteme Giriş</h2>
            <div class="input-group">
                <input type="text" id="username" placeholder="Kullanıcı Adı">
            </div>
            <div class="input-group">
                <input type="password" id="password" placeholder="Şifre">
            </div>
            <button class="btn-login" onclick="login()">Giriş Yap</button>
            <p id="login-error">Kullanıcı adı veya şifre hatalı!</p>
        </div>

        <!-- 2. DASHBOARD EKRANI (Başlangıçta Gizli) -->
        <div id="dashboard-section" style="display: none;">
            <div class="header">
                <div class="header-text">
                    <h1>Algoritmik Ticaret Paneli <span id="role-badge" class="role-badge"></span></h1>
                    <p>Sistem Durumu: Çevrimiçi | Motor Entegrasyonu: Aktif</p>
                </div>
                <button class="btn-logout" onclick="logout()">Çıkış Yap</button>
            </div>
            <div id="chart-container"></div>
            <div id="error-box"></div>

            <h2 class="section-title">
                Live Market
            </h2>

            <div class="live-market-toolbar">
                <div class="backtest-control">
                    <label for="live-market-symbol">
                        Symbol
                    </label>

                    <select id="live-market-symbol">
                        <option value="BTCUSDT">
                            BTC / USDT
                        </option>
                        <option value="ETHUSDT">
                            ETH / USDT
                        </option>
                        <option value="BNBUSDT">
                            BNB / USDT
                        </option>
                        <option value="SOLUSDT">
                            SOL / USDT
                        </option>
                    </select>
                </div>

                <div class="backtest-control">
                    <label for="live-market-interval">
                        Interval
                    </label>

                    <select id="live-market-interval">
                        <option value="1m">1m</option>
                        <option value="5m">5m</option>
                        <option value="15m">15m</option>
                        <option value="1h">1h</option>
                    </select>
                </div>

                <button
                    class="btn-backtest"
                    onclick="startLiveMarket()"
                >
                    Connect Market
                </button>

                <div
                    id="live-market-status"
                    class="live-market-status disconnected"
                >
                    ● OFFLINE
                </div>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">
                        Symbol
                    </div>
                    <div
                        id="live-symbol"
                        class="metric-value metric-neutral"
                    >
                        BTCUSDT
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Live Price
                    </div>
                    <div
                        id="live-price"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Open
                    </div>
                    <div
                        id="live-open"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        High
                    </div>
                    <div
                        id="live-high"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Low
                    </div>
                    <div
                        id="live-low"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Volume
                    </div>
                    <div
                        id="live-volume"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Trades
                    </div>
                    <div
                        id="live-trade-count"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Candle
                    </div>
                    <div
                        id="live-candle-status"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">
                        24H Change
                    </div>
                    <div
                        id="live-24h-change"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        24H High
                    </div>
                    <div
                        id="live-24h-high"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        24H Low
                    </div>
                    <div
                        id="live-24h-low"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        24H Quote Volume
                    </div>
                    <div
                        id="live-24h-volume"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>
            </div>


            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">
                        SMA20
                    </div>
                    <div
                        id="live-sma20"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        SMA50
                    </div>
                    <div
                        id="live-sma50"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Live Signal
                    </div>
                    <div
                        id="live-signal"
                        class="metric-value metric-neutral"
                    >
                        HOLD
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Trend
                    </div>
                    <div
                        id="live-trend"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        SMA Spread
                    </div>
                    <div
                        id="live-sma-spread"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Current Crossover
                    </div>
                    <div
                        id="live-crossover"
                        class="metric-value metric-neutral"
                    >
                        NONE
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Last Confirmed Crossover
                    </div>
                    <div
                        id="live-last-crossover"
                        class="metric-value metric-neutral"
                    >
                        NONE
                    </div>
                </div>
            </div>

            <h3 class="section-title">
                Live Technical Analysis
            </h3>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">
                        RSI 14
                    </div>
                    <div
                        id="live-rsi14"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        MACD
                    </div>
                    <div
                        id="live-macd"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        MACD Signal
                    </div>
                    <div
                        id="live-macd-signal"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        MACD Histogram
                    </div>
                    <div
                        id="live-macd-histogram"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        BB Upper
                    </div>
                    <div
                        id="live-bb-upper"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        BB Middle
                    </div>
                    <div
                        id="live-bb-middle"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        BB Lower
                    </div>
                    <div
                        id="live-bb-lower"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        BB Width
                    </div>
                    <div
                        id="live-bb-width"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Technical Score
                    </div>
                    <div
                        id="live-technical-score"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Technical Rating
                    </div>
                    <div
                        id="live-technical-rating"
                        class="metric-value metric-neutral"
                    >
                        NEUTRAL
                    </div>
                </div>
            </div>

            <div class="live-chart-card">
                <div
                    id="live-market-chart"
                    class="live-market-chart"
                ></div>
            </div>

            <div class="signal-alert-section">
                <div class="signal-alert-header">
                    <div>
                        <h3 class="section-title">
                            Live Signal Alerts
                        </h3>

                        <div class="signal-alert-description">
                            Only confirmed crossovers from closed candles
                            are added to this history.
                        </div>
                    </div>

                    <button
                        class="btn-secondary"
                        onclick="clearLiveSignalHistory()"
                    >
                        Clear History
                    </button>
                </div>

                <div
                    id="live-signal-history-empty"
                    class="signal-history-empty"
                >
                    No confirmed signals in this session.
                </div>

                <div
                    id="live-signal-history"
                    class="signal-history-list"
                ></div>
            </div>

            <div
                id="live-signal-toast"
                class="signal-toast"
            ></div>

            <div id="live-market-error"></div>


            <h2 class="section-title">
                Strategy Backtest
            </h2>

            <div class="backtest-toolbar">
                <div class="backtest-control">
                    <label for="initial-capital">
                        Initial Capital
                    </label>

                    <input
                        type="number"
                        id="initial-capital"
                        value="10000"
                        min="1"
                        max="100000000"
                        step="1000"
                    >
                </div>

                <div class="backtest-control">
                    <label for="transaction-fee">
                        Transaction Fee (%)
                    </label>

                    <input
                        type="number"
                        id="transaction-fee"
                        value="0.10"
                        min="0"
                        max="5"
                        step="0.01"
                    >
                </div>

                <div class="backtest-control">
                    <label for="slippage">
                        Slippage (%)
                    </label>

                    <input
                        type="number"
                        id="slippage"
                        value="0.05"
                        min="0"
                        max="5"
                        step="0.01"
                    >
                </div>

                <button
                    class="btn-backtest"
                    onclick="loadBacktestData()"
                >
                    Backtesti Yenile
                </button>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">
                        Strategy Return
                    </div>
                    <div
                        id="metric-strategy-return"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Buy & Hold
                    </div>
                    <div
                        id="metric-buy-hold"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Max Drawdown
                    </div>
                    <div
                        id="metric-drawdown"
                        class="metric-value metric-negative"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Win Rate
                    </div>
                    <div
                        id="metric-win-rate"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Final Portfolio
                    </div>
                    <div
                        id="metric-final-value"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Closed Trades
                    </div>
                    <div
                        id="metric-trades"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Excess Return
                    </div>
                    <div
                        id="metric-excess-return"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Sharpe Ratio
                    </div>
                    <div
                        id="metric-sharpe"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Annual Volatility
                    </div>
                    <div
                        id="metric-volatility"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Total Fees
                    </div>
                    <div
                        id="metric-total-fees"
                        class="metric-value metric-negative"
                    >
                        --
                    </div>
                </div>
            </div>

            <div id="backtest-meta" class="backtest-meta"></div>

            <div id="equity-chart-container"></div>

            <div id="backtest-error"></div>

            <h2 class="section-title">
                Strategy Optimizer
            </h2>

            <div class="optimizer-toolbar">
                <div class="backtest-control">
                    <label for="optimizer-objective">
                        Optimization Objective
                    </label>

                    <select id="optimizer-objective">
                        <option value="sharpe_ratio">
                            Sharpe Ratio
                        </option>

                        <option value="total_return_pct">
                            Total Return
                        </option>

                        <option value="excess_return_pct">
                            Excess Return
                        </option>
                    </select>
                </div>

                <button
                    class="btn-backtest"
                    onclick="loadOptimizerData()"
                >
                    Optimize Strategy
                </button>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">
                        Best SMA Pair
                    </div>
                    <div
                        id="optimizer-best-pair"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Optimized Return
                    </div>
                    <div
                        id="optimizer-return"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Optimized Sharpe
                    </div>
                    <div
                        id="optimizer-sharpe"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Optimized Drawdown
                    </div>
                    <div
                        id="optimizer-drawdown"
                        class="metric-value metric-negative"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Excess Return
                    </div>
                    <div
                        id="optimizer-excess"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Combinations Tested
                    </div>
                    <div
                        id="optimizer-count"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>
            </div>

            <div class="optimizer-table-wrapper">
                <table class="optimizer-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>SMA Pair</th>
                            <th>Return</th>
                            <th>Sharpe</th>
                            <th>Drawdown</th>
                            <th>Trades</th>
                        </tr>
                    </thead>

                    <tbody id="optimizer-results-body">
                    </tbody>
                </table>
            </div>

            <div id="optimizer-error"></div>

            <h2 class="section-title">
                Walk-Forward Validation
            </h2>

            <div class="optimizer-toolbar">
                <div class="backtest-control">
                    <label for="walk-forward-objective">
                        Training Objective
                    </label>

                    <select id="walk-forward-objective">
                        <option value="sharpe_ratio">
                            Sharpe Ratio
                        </option>
                        <option value="total_return_pct">
                            Total Return
                        </option>
                        <option value="excess_return_pct">
                            Excess Return
                        </option>
                    </select>
                </div>

                <div class="backtest-control">
                    <label for="walk-forward-train-size">
                        Initial Train Rows
                    </label>

                    <input
                        id="walk-forward-train-size"
                        type="number"
                        value="250"
                        min="101"
                        max="450"
                    >
                </div>

                <div class="backtest-control">
                    <label for="walk-forward-test-size">
                        Test Rows
                    </label>

                    <input
                        id="walk-forward-test-size"
                        type="number"
                        value="50"
                        min="10"
                        max="100"
                    >
                </div>

                <button
                    class="btn-backtest"
                    onclick="loadWalkForwardData()"
                >
                    Validate Out-of-Sample
                </button>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">OOS Return</div>
                    <div id="wf-oos-return" class="metric-value">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Benchmark</div>
                    <div id="wf-benchmark" class="metric-value">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">OOS Excess Return</div>
                    <div id="wf-excess" class="metric-value">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Average Test Sharpe</div>
                    <div id="wf-average-sharpe" class="metric-value">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Median Test Sharpe</div>
                    <div id="wf-median-sharpe" class="metric-value">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Worst Fold Drawdown</div>
                    <div id="wf-drawdown" class="metric-value metric-negative">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Profitable Folds</div>
                    <div id="wf-profitable-folds" class="metric-value">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Most Selected Pair</div>
                    <div id="wf-pair" class="metric-value">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Closed Trades</div>
                    <div id="wf-trades" class="metric-value metric-neutral">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Zero-Trade Folds</div>
                    <div id="wf-zero-folds" class="metric-value metric-neutral">--</div>
                </div>
            </div>

            <div class="optimizer-table-wrapper">
                <table class="optimizer-table">
                    <thead>
                        <tr>
                            <th>Fold</th>
                            <th>Selected SMA</th>
                            <th>Test Period</th>
                            <th>Return</th>
                            <th>Benchmark</th>
                            <th>Sharpe</th>
                            <th>Drawdown</th>
                            <th>Trades</th>
                        </tr>
                    </thead>

                    <tbody id="walk-forward-results-body"></tbody>
                </table>
            </div>

            <div id="walk-forward-error"></div>

            <h2 class="section-title">
                Trade Analytics
            </h2>

            <div class="optimizer-toolbar">
                <button
                    class="btn-backtest"
                    onclick="loadTradeAnalytics()"
                >
                    Load Trade History
                </button>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">
                        Closed Trades
                    </div>
                    <div
                        id="trade-closed-count"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Win Rate
                    </div>
                    <div
                        id="trade-win-rate"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Realized Net P&amp;L
                    </div>
                    <div
                        id="trade-net-pnl"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Profit Factor
                    </div>
                    <div
                        id="trade-profit-factor"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Avg Trade Return
                    </div>
                    <div
                        id="trade-average-return"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Avg Holding Period
                    </div>
                    <div
                        id="trade-average-holding"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Total Fees
                    </div>
                    <div
                        id="trade-total-fees"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Slippage Cost
                    </div>
                    <div
                        id="trade-slippage-cost"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>
            </div>

            <h3 class="section-title">
                Open Position
            </h3>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">
                        Entry Date
                    </div>
                    <div
                        id="position-entry-date"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Entry Price
                    </div>
                    <div
                        id="position-entry-price"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Current Price
                    </div>
                    <div
                        id="position-current-price"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Unrealized P&amp;L
                    </div>
                    <div
                        id="position-unrealized-pnl"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Unrealized Return
                    </div>
                    <div
                        id="position-unrealized-return"
                        class="metric-value"
                    >
                        --
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">
                        Holding Days
                    </div>
                    <div
                        id="position-holding-days"
                        class="metric-value metric-neutral"
                    >
                        --
                    </div>
                </div>
            </div>

            <h3 class="section-title">
                Trade History
            </h3>

            <div class="optimizer-table-wrapper">
                <table class="optimizer-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Entry</th>
                            <th>Exit</th>
                            <th>Days</th>
                            <th>Entry Price</th>
                            <th>Exit Price</th>
                            <th>Return</th>
                            <th>Net P&amp;L</th>
                            <th>Fees</th>
                            <th>Result</th>
                        </tr>
                    </thead>

                    <tbody id="trade-history-body">
                    </tbody>
                </table>
            </div>

            <div id="trade-analytics-error"></div>
        </div>

        <script>
            // --- KİMLİK DOĞRULAMA MANTIĞI ---
            function login() {
                const params = new URLSearchParams();
                params.append('username', document.getElementById('username').value);
                params.append('password', document.getElementById('password').value);
                
                fetch('/token', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: params
                })
                .then(response => {
                    if (!response.ok) throw new Error("Giriş Başarısız");
                    return response.json();
                })
                .then(data => {
                    // Token'ı tarayıcıya (localStorage) kaydet
                    localStorage.setItem('quant_token', data.access_token);
                    localStorage.setItem('quant_role', data.role);
                    document.getElementById('login-error').style.display = 'none';
                    showDashboard();
                })
                .catch(err => {
                    document.getElementById('login-error').style.display = 'block';
                });
            }

            function logout() {
                localStorage.removeItem('quant_token');
                localStorage.removeItem('quant_role');
                location.reload(); // Sayfayı yenileyerek giriş ekranına dön
            }

            function showDashboard() {
                document.getElementById('login-section').style.display = 'none';
                document.getElementById('dashboard-section').style.display = 'block';
                
                // Rol etiketini güncelle (ADMIN veya USER)
                const userRole = localStorage.getItem('quant_role').toUpperCase();
                const badge = document.getElementById('role-badge');
                badge.innerText = userRole;
                if(userRole !== 'ADMIN') badge.style.background = '#2962FF'; // Standart kullanıcıya mavi etiket
                
                loadChartData();
                loadBacktestData();
            }

            // Sayfa yüklendiğinde oturum açık mı kontrol et
            if (localStorage.getItem('quant_token')) {
                showDashboard();
            }

            // --- BACKTEST DASHBOARD MANTIĞI ---

            function formatPercent(value) {
                const number = Number(value);

                return (
                    (number >= 0 ? '+' : '')
                    + number.toFixed(2)
                    + '%'
                );
            }

            function formatCurrency(value) {
                return new Intl.NumberFormat(
                    'en-US',
                    {
                        style: 'currency',
                        currency: 'USD',
                        maximumFractionDigits: 2
                    }
                ).format(Number(value));
            }

            function setMetricTrend(elementId, value) {
                const element =
                    document.getElementById(elementId);

                element.classList.remove(
                    'metric-positive',
                    'metric-negative',
                    'metric-neutral'
                );

                if (Number(value) > 0) {
                    element.classList.add(
                        'metric-positive'
                    );
                } else if (Number(value) < 0) {
                    element.classList.add(
                        'metric-negative'
                    );
                } else {
                    element.classList.add(
                        'metric-neutral'
                    );
                }
            }

            function loadBacktestData() {
                const token =
                    localStorage.getItem('quant_token');

                const capitalInput =
                    document.getElementById(
                        'initial-capital'
                    );

                const initialCapital =
                    Number(capitalInput.value);

                const transactionFee =
                    Number(
                        document.getElementById(
                            'transaction-fee'
                        ).value
                    );

                const slippage =
                    Number(
                        document.getElementById(
                            'slippage'
                        ).value
                    );

                const errorBox =
                    document.getElementById(
                        'backtest-error'
                    );

                errorBox.innerText = '';

                if (
                    !Number.isFinite(initialCapital)
                    || initialCapital <= 0
                ) {
                    errorBox.innerText =
                        'Başlangıç sermayesi '
                        + 'sıfırdan büyük olmalıdır.';

                    return;
                }

                if (
                    !Number.isFinite(transactionFee)
                    || transactionFee < 0
                    || transactionFee > 5
                ) {
                    errorBox.innerText =
                        'Transaction fee 0 ile 5 '
                        + 'arasında olmalıdır.';

                    return;
                }

                if (
                    !Number.isFinite(slippage)
                    || slippage < 0
                    || slippage > 5
                ) {
                    errorBox.innerText =
                        'Slippage 0 ile 5 '
                        + 'arasında olmalıdır.';

                    return;
                }

                const url =
                    '/api/backtest?initial_capital='
                    + encodeURIComponent(
                        initialCapital
                    )
                    + '&transaction_fee_pct='
                    + encodeURIComponent(
                        transactionFee
                    )
                    + '&slippage_pct='
                    + encodeURIComponent(
                        slippage
                    )
                    + '&v='
                    + new Date().getTime();

                fetch(
                    url,
                    {
                        headers: {
                            'Authorization':
                                'Bearer ' + token
                        }
                    }
                )
                .then(response => {
                    if (response.status === 401) {
                        logout();

                        throw new Error(
                            'Oturum süresi doldu.'
                        );
                    }

                    if (!response.ok) {
                        return response.json()
                            .then(body => {
                                const detail =
                                    body.detail
                                    || 'Backtest API hatası';

                                throw new Error(detail);
                            });
                    }

                    return response.json();
                })
                .then(data => {
                    const summary = data.summary;

                    const strategyReturn =
                        document.getElementById(
                            'metric-strategy-return'
                        );

                    strategyReturn.innerText =
                        formatPercent(
                            summary.total_return_pct
                        );

                    setMetricTrend(
                        'metric-strategy-return',
                        summary.total_return_pct
                    );


                    const buyHold =
                        document.getElementById(
                            'metric-buy-hold'
                        );

                    buyHold.innerText =
                        formatPercent(
                            summary.buy_hold_return_pct
                        );

                    setMetricTrend(
                        'metric-buy-hold',
                        summary.buy_hold_return_pct
                    );


                    document.getElementById(
                        'metric-drawdown'
                    ).innerText =
                        '-'
                        + Number(
                            summary.max_drawdown_pct
                        ).toFixed(2)
                        + '%';


                    document.getElementById(
                        'metric-win-rate'
                    ).innerText =
                        Number(
                            summary.win_rate_pct
                        ).toFixed(2)
                        + '%';


                    document.getElementById(
                        'metric-final-value'
                    ).innerText =
                        formatCurrency(
                            summary.final_value
                        );


                    document.getElementById(
                        'metric-trades'
                    ).innerText =
                        summary.closed_trades;


                    const excessReturn =
                        document.getElementById(
                            'metric-excess-return'
                        );

                    excessReturn.innerText =
                        formatPercent(
                            summary.excess_return_pct
                        );

                    setMetricTrend(
                        'metric-excess-return',
                        summary.excess_return_pct
                    );


                    const sharpe =
                        document.getElementById(
                            'metric-sharpe'
                        );

                    sharpe.innerText =
                        Number(
                            summary.sharpe_ratio
                        ).toFixed(3);

                    setMetricTrend(
                        'metric-sharpe',
                        summary.sharpe_ratio
                    );


                    document.getElementById(
                        'metric-volatility'
                    ).innerText =
                        Number(
                            summary
                            .annualized_volatility_pct
                        ).toFixed(2)
                        + '%';


                    document.getElementById(
                        'metric-total-fees'
                    ).innerText =
                        formatCurrency(
                            summary.total_fees_paid
                        );


                    document.getElementById(
                        'backtest-meta'
                    ).innerText =
                        data.strategy
                        + ' | '
                        + summary.first_date
                        + ' → '
                        + summary.last_date
                        + ' | Wins: '
                        + summary.winning_trades
                        + ' | Losses: '
                        + summary.losing_trades
                        + ' | Fee: '
                        + Number(
                            summary.transaction_fee_pct
                        ).toFixed(2)
                        + '%'
                        + ' | Slippage: '
                        + Number(
                            summary.slippage_pct
                        ).toFixed(2)
                        + '%';


                    const container =
                        document.getElementById(
                            'equity-chart-container'
                        );

                    container.innerHTML = '';

                    const equityChart =
                        LightweightCharts.createChart(
                            container,
                            {
                                layout: {
                                    textColor: '#d1d4dc',
                                    background: {
                                        type: 'solid',
                                        color: '#1E222D'
                                    }
                                },

                                grid: {
                                    vertLines: {
                                        color: '#2B2B43'
                                    },
                                    horzLines: {
                                        color: '#2B2B43'
                                    }
                                },

                                crosshair: {
                                    mode:
                                        LightweightCharts
                                        .CrosshairMode
                                        .Normal
                                },

                                rightPriceScale: {
                                    borderColor:
                                        '#434651'
                                },

                                timeScale: {
                                    borderColor:
                                        '#434651'
                                }
                            }
                        );

                    const equitySeries =
                        equityChart.addLineSeries({
                            color: '#7C4DFF',
                            lineWidth: 2,
                            title: 'Portfolio Equity',
                            priceFormat: {
                                type: 'price',
                                precision: 2,
                                minMove: 0.01
                            }
                        });

                    const equityData =
                        data.equity_curve.map(
                            point => ({
                                time: point.date,
                                value:
                                    Number(point.equity)
                            })
                        );

                    equitySeries.setData(
                        equityData
                    );

                    equityChart.timeScale()
                        .fitContent();
                })
                .catch(error => {
                    errorBox.innerText =
                        'Backtest Hatası: '
                        + error.message;
                });
            }


            // --- STRATEGY OPTIMIZER ---

            function loadOptimizerData() {
                const token =
                    localStorage.getItem('quant_token');

                const objective =
                    document.getElementById(
                        'optimizer-objective'
                    ).value;

                const initialCapital =
                    Number(
                        document.getElementById(
                            'initial-capital'
                        ).value
                    );

                const transactionFee =
                    Number(
                        document.getElementById(
                            'transaction-fee'
                        ).value
                    );

                const slippage =
                    Number(
                        document.getElementById(
                            'slippage'
                        ).value
                    );

                const errorBox =
                    document.getElementById(
                        'optimizer-error'
                    );

                errorBox.innerText = '';

                const params =
                    new URLSearchParams({
                        objective: objective,
                        top_n: '5',
                        initial_capital:
                            String(initialCapital),
                        transaction_fee_pct:
                            String(transactionFee),
                        slippage_pct:
                            String(slippage)
                    });

                fetch(
                    '/api/optimize?' + params.toString(),
                    {
                        headers: {
                            'Authorization':
                                'Bearer ' + token
                        }
                    }
                )
                .then(response => {
                    if (response.status === 401) {
                        logout();

                        throw new Error(
                            'Oturum süresi doldu.'
                        );
                    }

                    if (!response.ok) {
                        return response.json()
                            .then(body => {
                                throw new Error(
                                    body.detail
                                    || 'Optimizer API hatası'
                                );
                            });
                    }

                    return response.json();
                })
                .then(data => {
                    const best = data.best;

                    document.getElementById(
                        'optimizer-best-pair'
                    ).innerText =
                        'SMA'
                        + best.short_window
                        + ' / SMA'
                        + best.long_window;


                    const optimizedReturn =
                        document.getElementById(
                            'optimizer-return'
                        );

                    optimizedReturn.innerText =
                        formatPercent(
                            best.total_return_pct
                        );

                    setMetricTrend(
                        'optimizer-return',
                        best.total_return_pct
                    );


                    const optimizedSharpe =
                        document.getElementById(
                            'optimizer-sharpe'
                        );

                    optimizedSharpe.innerText =
                        Number(
                            best.sharpe_ratio
                        ).toFixed(3);

                    setMetricTrend(
                        'optimizer-sharpe',
                        best.sharpe_ratio
                    );


                    document.getElementById(
                        'optimizer-drawdown'
                    ).innerText =
                        '-'
                        + Number(
                            best.max_drawdown_pct
                        ).toFixed(2)
                        + '%';


                    const excess =
                        document.getElementById(
                            'optimizer-excess'
                        );

                    excess.innerText =
                        formatPercent(
                            best.excess_return_pct
                        );

                    setMetricTrend(
                        'optimizer-excess',
                        best.excess_return_pct
                    );


                    document.getElementById(
                        'optimizer-count'
                    ).innerText =
                        data.tested_combinations;


                    const tbody =
                        document.getElementById(
                            'optimizer-results-body'
                        );

                    tbody.innerHTML = '';

                    data.top_results.forEach(
                        (row, index) => {
                            const tr =
                                document.createElement(
                                    'tr'
                                );

                            tr.innerHTML =
                                '<td>'
                                + (index + 1)
                                + '</td>'
                                + '<td>SMA'
                                + row.short_window
                                + ' / SMA'
                                + row.long_window
                                + '</td>'
                                + '<td>'
                                + formatPercent(
                                    row.total_return_pct
                                )
                                + '</td>'
                                + '<td>'
                                + Number(
                                    row.sharpe_ratio
                                ).toFixed(3)
                                + '</td>'
                                + '<td>-'
                                + Number(
                                    row.max_drawdown_pct
                                ).toFixed(2)
                                + '%</td>'
                                + '<td>'
                                + row.closed_trades
                                + '</td>';

                            tbody.appendChild(tr);
                        }
                    );
                })
                .catch(error => {
                    errorBox.innerText =
                        'Optimizer Hatası: '
                        + error.message;
                });
            }


            // --- WALK-FORWARD VALIDATION ---

            function loadWalkForwardData() {
                const token =
                    localStorage.getItem('quant_token');

                const objective =
                    document.getElementById(
                        'walk-forward-objective'
                    ).value;

                const trainSize =
                    Number(
                        document.getElementById(
                            'walk-forward-train-size'
                        ).value
                    );

                const testSize =
                    Number(
                        document.getElementById(
                            'walk-forward-test-size'
                        ).value
                    );

                const initialCapital =
                    Number(
                        document.getElementById(
                            'initial-capital'
                        ).value
                    );

                const transactionFee =
                    Number(
                        document.getElementById(
                            'transaction-fee'
                        ).value
                    );

                const slippage =
                    Number(
                        document.getElementById(
                            'slippage'
                        ).value
                    );

                const errorBox =
                    document.getElementById(
                        'walk-forward-error'
                    );

                errorBox.innerText = '';

                const params =
                    new URLSearchParams({
                        objective: objective,
                        initial_train_size:
                            String(trainSize),
                        test_size:
                            String(testSize),
                        initial_capital:
                            String(initialCapital),
                        transaction_fee_pct:
                            String(transactionFee),
                        slippage_pct:
                            String(slippage)
                    });

                fetch(
                    '/api/walk-forward?'
                    + params.toString(),
                    {
                        headers: {
                            'Authorization':
                                'Bearer ' + token
                        }
                    }
                )
                .then(response => {
                    if (response.status === 401) {
                        logout();

                        throw new Error(
                            'Oturum süresi doldu.'
                        );
                    }

                    if (!response.ok) {
                        return response.json()
                            .then(body => {
                                throw new Error(
                                    body.detail
                                    || 'Walk-forward API hatası'
                                );
                            });
                    }

                    return response.json();
                })
                .then(data => {
                    const summary = data.summary;

                    const oos =
                        document.getElementById(
                            'wf-oos-return'
                        );

                    oos.innerText =
                        formatPercent(
                            summary.out_of_sample_return_pct
                        );

                    setMetricTrend(
                        'wf-oos-return',
                        summary.out_of_sample_return_pct
                    );


                    document.getElementById(
                        'wf-benchmark'
                    ).innerText =
                        formatPercent(
                            summary.benchmark_return_pct
                        );


                    const excess =
                        document.getElementById(
                            'wf-excess'
                        );

                    excess.innerText =
                        formatPercent(
                            summary.excess_return_pct
                        );

                    setMetricTrend(
                        'wf-excess',
                        summary.excess_return_pct
                    );


                    document.getElementById(
                        'wf-average-sharpe'
                    ).innerText =
                        Number(
                            summary.average_test_sharpe
                        ).toFixed(3);


                    document.getElementById(
                        'wf-median-sharpe'
                    ).innerText =
                        Number(
                            summary.median_test_sharpe
                        ).toFixed(3);


                    document.getElementById(
                        'wf-drawdown'
                    ).innerText =
                        '-'
                        + Number(
                            summary.worst_fold_drawdown_pct
                        ).toFixed(2)
                        + '%';


                    document.getElementById(
                        'wf-profitable-folds'
                    ).innerText =
                        summary.profitable_folds
                        + ' / '
                        + summary.folds;


                    const pair =
                        summary.most_selected_pair;

                    document.getElementById(
                        'wf-pair'
                    ).innerText =
                        'SMA'
                        + pair.short_window
                        + ' / SMA'
                        + pair.long_window;


                    document.getElementById(
                        'wf-trades'
                    ).innerText =
                        summary.total_closed_trades;


                    document.getElementById(
                        'wf-zero-folds'
                    ).innerText =
                        summary.zero_closed_trade_folds;


                    const tbody =
                        document.getElementById(
                            'walk-forward-results-body'
                        );

                    tbody.innerHTML = '';

                    data.folds.forEach(fold => {
                        const tr =
                            document.createElement('tr');

                        tr.innerHTML =
                            '<td>'
                            + fold.fold
                            + '</td>'
                            + '<td>SMA'
                            + fold.selected_short_window
                            + ' / SMA'
                            + fold.selected_long_window
                            + '</td>'
                            + '<td>'
                            + fold.test_start
                            + ' → '
                            + fold.test_end
                            + '</td>'
                            + '<td>'
                            + formatPercent(
                                fold.test_return_pct
                            )
                            + '</td>'
                            + '<td>'
                            + formatPercent(
                                fold.test_buy_hold_return_pct
                            )
                            + '</td>'
                            + '<td>'
                            + Number(
                                fold.test_sharpe_ratio
                            ).toFixed(3)
                            + '</td>'
                            + '<td>-'
                            + Number(
                                fold.test_max_drawdown_pct
                            ).toFixed(2)
                            + '%</td>'
                            + '<td>'
                            + fold.test_closed_trades
                            + '</td>';

                        tbody.appendChild(tr);
                    });
                })
                .catch(error => {
                    errorBox.innerText =
                        'Walk-Forward Hatası: '
                        + error.message;
                });
            }


            // --- TRADE ANALYTICS ---

            function loadTradeAnalytics() {
                const token =
                    localStorage.getItem(
                        'quant_token'
                    );

                const initialCapital =
                    Number(
                        document.getElementById(
                            'initial-capital'
                        ).value
                    );

                const transactionFee =
                    Number(
                        document.getElementById(
                            'transaction-fee'
                        ).value
                    );

                const slippage =
                    Number(
                        document.getElementById(
                            'slippage'
                        ).value
                    );

                const errorBox =
                    document.getElementById(
                        'trade-analytics-error'
                    );

                errorBox.innerText = '';

                const params =
                    new URLSearchParams({
                        initial_capital:
                            String(initialCapital),
                        transaction_fee_pct:
                            String(transactionFee),
                        slippage_pct:
                            String(slippage),
                        force_close_at_end:
                            'false'
                    });

                fetch(
                    '/api/trades?'
                    + params.toString(),
                    {
                        headers: {
                            'Authorization':
                                'Bearer ' + token
                        }
                    }
                )
                .then(response => {
                    if (response.status === 401) {
                        logout();

                        throw new Error(
                            'Oturum süresi doldu.'
                        );
                    }

                    if (!response.ok) {
                        return response
                            .json()
                            .then(body => {
                                throw new Error(
                                    body.detail
                                    || 'Trade Analytics API hatası'
                                );
                            });
                    }

                    return response.json();
                })
                .then(data => {
                    const summary = data.summary;

                    document.getElementById(
                        'trade-closed-count'
                    ).innerText =
                        summary.closed_trades;


                    const winRate =
                        document.getElementById(
                            'trade-win-rate'
                        );

                    winRate.innerText =
                        Number(
                            summary.win_rate_pct
                        ).toFixed(2)
                        + '%';


                    const netPnl =
                        document.getElementById(
                            'trade-net-pnl'
                        );

                    netPnl.innerText =
                        '$'
                        + Number(
                            summary.total_net_pnl
                        ).toFixed(2);

                    setMetricTrend(
                        'trade-net-pnl',
                        summary.total_net_pnl
                    );


                    document.getElementById(
                        'trade-profit-factor'
                    ).innerText =
                        summary.profit_factor === null
                        ? 'N/A'
                        : Number(
                            summary.profit_factor
                        ).toFixed(3);


                    const avgReturn =
                        document.getElementById(
                            'trade-average-return'
                        );

                    avgReturn.innerText =
                        formatPercent(
                            summary.average_trade_return_pct
                        );

                    setMetricTrend(
                        'trade-average-return',
                        summary.average_trade_return_pct
                    );


                    document.getElementById(
                        'trade-average-holding'
                    ).innerText =
                        Number(
                            summary.average_holding_days
                        ).toFixed(1)
                        + ' days';


                    document.getElementById(
                        'trade-total-fees'
                    ).innerText =
                        '$'
                        + Number(
                            summary.total_fees
                        ).toFixed(2);


                    document.getElementById(
                        'trade-slippage-cost'
                    ).innerText =
                        '$'
                        + Number(
                            summary.total_slippage_cost
                        ).toFixed(2);


                    const position =
                        data.open_position;

                    if (position) {
                        document.getElementById(
                            'position-entry-date'
                        ).innerText =
                            position.entry_date;

                        document.getElementById(
                            'position-entry-price'
                        ).innerText =
                            '$'
                            + Number(
                                position.market_entry_price
                            ).toFixed(2);

                        document.getElementById(
                            'position-current-price'
                        ).innerText =
                            '$'
                            + Number(
                                position.current_market_price
                            ).toFixed(2);

                        const pnl =
                            document.getElementById(
                                'position-unrealized-pnl'
                            );

                        pnl.innerText =
                            '$'
                            + Number(
                                position.unrealized_pnl
                            ).toFixed(2);

                        setMetricTrend(
                            'position-unrealized-pnl',
                            position.unrealized_pnl
                        );

                        const positionReturn =
                            document.getElementById(
                                'position-unrealized-return'
                            );

                        positionReturn.innerText =
                            formatPercent(
                                position.unrealized_return_pct
                            );

                        setMetricTrend(
                            'position-unrealized-return',
                            position.unrealized_return_pct
                        );

                        document.getElementById(
                            'position-holding-days'
                        ).innerText =
                            position.holding_days
                            + ' days';
                    } else {
                        document.getElementById(
                            'position-entry-date'
                        ).innerText = 'None';

                        document.getElementById(
                            'position-entry-price'
                        ).innerText = '--';

                        document.getElementById(
                            'position-current-price'
                        ).innerText = '--';

                        document.getElementById(
                            'position-unrealized-pnl'
                        ).innerText = '$0.00';

                        document.getElementById(
                            'position-unrealized-return'
                        ).innerText = '0.00%';

                        document.getElementById(
                            'position-holding-days'
                        ).innerText = '0 days';
                    }


                    const tbody =
                        document.getElementById(
                            'trade-history-body'
                        );

                    tbody.innerHTML = '';

                    data.trades.forEach(trade => {
                        const tr =
                            document.createElement(
                                'tr'
                            );

                        tr.innerHTML =
                            '<td>'
                            + trade.trade_number
                            + '</td>'
                            + '<td>'
                            + trade.entry_date
                            + '</td>'
                            + '<td>'
                            + trade.exit_date
                            + '</td>'
                            + '<td>'
                            + trade.holding_days
                            + '</td>'
                            + '<td>$'
                            + Number(
                                trade.market_entry_price
                            ).toFixed(2)
                            + '</td>'
                            + '<td>$'
                            + Number(
                                trade.market_exit_price
                            ).toFixed(2)
                            + '</td>'
                            + '<td>'
                            + formatPercent(
                                trade.return_pct
                            )
                            + '</td>'
                            + '<td>$'
                            + Number(
                                trade.net_pnl
                            ).toFixed(2)
                            + '</td>'
                            + '<td>$'
                            + Number(
                                trade.total_fees
                            ).toFixed(2)
                            + '</td>'
                            + '<td>'
                            + trade.result
                            + '</td>';

                        tbody.appendChild(tr);
                    });
                })
                .catch(error => {
                    errorBox.innerText =
                        'Trade Analytics Hatası: '
                        + error.message;
                });
            }


            // --- BINANCE LIVE MARKET ---

            let liveMarketSocket = null;
            let liveMarketChart = null;
            let liveCandlestickSeries = null;
            let liveSma20Series = null;
            let liveSma50Series = null;
            let liveVolumeSeries = null;
            let liveBbUpperSeries = null;
            let liveBbMiddleSeries = null;
            let liveBbLowerSeries = null;
            let liveSignalMarkers = [];
            let liveProvisionalMarker = null;
            let liveSignalHistory = [];
            let liveLastConfirmedCrossoverKey = null;
            let liveSignalToastTimer = null;
            let liveLastClose = null;
            let live24hStatsTimer = null;
            let liveReconnectTimer = null;
            let liveReconnectAttempts = 0;
            let liveMarketConfig = null;


            function formatMarketPrice(value) {
                const number = Number(value);

                if (!Number.isFinite(number)) {
                    return '--';
                }

                if (number >= 1000) {
                    return '$'
                        + number.toLocaleString(
                            undefined,
                            {
                                minimumFractionDigits: 2,
                                maximumFractionDigits: 2
                            }
                        );
                }

                return '$'
                    + number.toLocaleString(
                        undefined,
                        {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 6
                        }
                    );
            }


            function setLiveMarketStatus(
                state,
                text
            ) {
                const element =
                    document.getElementById(
                        'live-market-status'
                    );

                element.className =
                    'live-market-status ' + state;

                element.innerText = text;
            }


            function initializeLiveMarketChart() {
                const container =
                    document.getElementById(
                        'live-market-chart'
                    );

                if (!container) {
                    return;
                }

                if (liveMarketChart) {
                    liveMarketChart.remove();
                }

                liveMarketChart =
                    LightweightCharts.createChart(
                        container,
                        {
                            width:
                                container.clientWidth,

                            height: 480,

                            layout: {
                                background: {
                                    color: '#1E222D'
                                },

                                textColor: '#d1d4dc'
                            },

                            grid: {
                                vertLines: {
                                    color: '#2B2B43'
                                },

                                horzLines: {
                                    color: '#2B2B43'
                                }
                            },

                            timeScale: {
                                timeVisible: true,
                                secondsVisible: false,
                                borderColor: '#434651'
                            },

                            rightPriceScale: {
                                borderColor: '#434651'
                            }
                        }
                    );

                liveCandlestickSeries =
                    liveMarketChart
                    .addCandlestickSeries();

                liveSma20Series =
                    liveMarketChart.addLineSeries(
                        {
                            color: '#ff9800',
                            lineWidth: 2,
                            priceLineVisible: false,
                            lastValueVisible: true
                        }
                    );

                liveSma50Series =
                    liveMarketChart.addLineSeries(
                        {
                            color: '#29b6f6',
                            lineWidth: 2,
                            priceLineVisible: false,
                            lastValueVisible: true
                        }
                    );

                liveBbUpperSeries =
                    liveMarketChart.addLineSeries(
                        {
                            color:
                                'rgba(171, 71, 188, 0.70)',
                            lineWidth: 1,
                            priceLineVisible: false,
                            lastValueVisible: false
                        }
                    );

                liveBbMiddleSeries =
                    liveMarketChart.addLineSeries(
                        {
                            color:
                                'rgba(171, 71, 188, 0.35)',
                            lineWidth: 1,
                            priceLineVisible: false,
                            lastValueVisible: false
                        }
                    );

                liveBbLowerSeries =
                    liveMarketChart.addLineSeries(
                        {
                            color:
                                'rgba(171, 71, 188, 0.70)',
                            lineWidth: 1,
                            priceLineVisible: false,
                            lastValueVisible: false
                        }
                    );

                liveVolumeSeries =
                    liveMarketChart.addHistogramSeries(
                        {
                            priceFormat: {
                                type: 'volume'
                            },

                            priceScaleId: 'volume',

                            priceLineVisible: false,
                            lastValueVisible: false
                        }
                    );

                liveVolumeSeries
                    .priceScale()
                    .applyOptions(
                        {
                            scaleMargins: {
                                top: 0.78,
                                bottom: 0
                            }
                        }
                    );

                window.addEventListener(
                    'resize',
                    () => {
                        if (
                            liveMarketChart
                            && container
                        ) {
                            liveMarketChart.applyOptions(
                                {
                                    width:
                                        container.clientWidth
                                }
                            );
                        }
                    }
                );
            }


            function calculateBollingerSeries(
                candles,
                windowSize = 20,
                deviations = 2
            ) {
                const upper = [];
                const middle = [];
                const lower = [];

                for (
                    let index =
                        windowSize - 1;
                    index < candles.length;
                    index += 1
                ) {
                    const values =
                        candles
                        .slice(
                            index
                            - windowSize
                            + 1,
                            index + 1
                        )
                        .map(
                            candle =>
                                candle.close
                        );

                    const average =
                        values.reduce(
                            (total, value) =>
                                total + value,
                            0
                        )
                        / windowSize;

                    const variance =
                        values.reduce(
                            (total, value) =>
                                total
                                + Math.pow(
                                    value
                                    - average,
                                    2
                                ),
                            0
                        )
                        / windowSize;

                    const deviation =
                        Math.sqrt(
                            variance
                        );

                    const time =
                        candles[index].time;

                    middle.push(
                        {
                            time: time,
                            value: average
                        }
                    );

                    upper.push(
                        {
                            time: time,
                            value:
                                average
                                + deviations
                                * deviation
                        }
                    );

                    lower.push(
                        {
                            time: time,
                            value:
                                average
                                - deviations
                                * deviation
                        }
                    );
                }

                return {
                    upper: upper,
                    middle: middle,
                    lower: lower
                };
            }


            function calculateCrossoverMarkers(
                candles,
                shortWindow = 20,
                longWindow = 50
            ) {
                const markers = [];

                if (
                    candles.length
                    <= longWindow
                ) {
                    return markers;
                }

                function averageRange(
                    endIndex,
                    windowSize
                ) {
                    let total = 0;

                    for (
                        let index =
                            endIndex
                            - windowSize
                            + 1;
                        index <= endIndex;
                        index += 1
                    ) {
                        total +=
                            candles[index].close;
                    }

                    return (
                        total / windowSize
                    );
                }

                for (
                    let index = longWindow;
                    index < candles.length;
                    index += 1
                ) {
                    const previousShort =
                        averageRange(
                            index - 1,
                            shortWindow
                        );

                    const previousLong =
                        averageRange(
                            index - 1,
                            longWindow
                        );

                    const currentShort =
                        averageRange(
                            index,
                            shortWindow
                        );

                    const currentLong =
                        averageRange(
                            index,
                            longWindow
                        );

                    if (
                        previousShort
                        <= previousLong
                        && currentShort
                        > currentLong
                    ) {
                        markers.push(
                            {
                                time:
                                    candles[index].time,

                                position:
                                    'belowBar',

                                color:
                                    '#00e676',

                                shape:
                                    'arrowUp',

                                text:
                                    'BUY'
                            }
                        );

                    } else if (
                        previousShort
                        >= previousLong
                        && currentShort
                        < currentLong
                    ) {
                        markers.push(
                            {
                                time:
                                    candles[index].time,

                                position:
                                    'aboveBar',

                                color:
                                    '#ff5252',

                                shape:
                                    'arrowDown',

                                text:
                                    'SELL'
                            }
                        );
                    }
                }

                return markers;
            }


            function calculateSmaSeries(
                candles,
                windowSize
            ) {
                const result = [];
                let rollingSum = 0;

                candles.forEach(
                    (candle, index) => {
                        rollingSum += candle.close;

                        if (index >= windowSize) {
                            rollingSum -=
                                candles[
                                    index - windowSize
                                ].close;
                        }

                        if (
                            index
                            >= windowSize - 1
                        ) {
                            result.push(
                                {
                                    time:
                                        candle.time,

                                    value:
                                        rollingSum
                                        / windowSize
                                }
                            );
                        }
                    }
                );

                return result;
            }


            async function load24hMarketStats(
                symbol
            ) {
                const token =
                    localStorage.getItem(
                        'quant_token'
                    );

                const params =
                    new URLSearchParams({
                        symbol: symbol
                    });

                const response =
                    await fetch(
                        '/api/market/ticker/24h?'
                        + params.toString(),
                        {
                            headers: {
                                'Authorization':
                                    'Bearer ' + token
                            }
                        }
                    );

                if (response.status === 401) {
                    logout();

                    throw new Error(
                        'Oturum süresi doldu.'
                    );
                }

                if (!response.ok) {
                    const body =
                        await response.json();

                    throw new Error(
                        body.detail
                        || '24h market verisi alınamadı.'
                    );
                }

                const data =
                    await response.json();

                const ticker =
                    data.ticker;

                const change =
                    Number(
                        ticker.price_change_pct
                    );

                const changeElement =
                    document.getElementById(
                        'live-24h-change'
                    );

                changeElement.innerText =
                    (
                        change >= 0
                        ? '+'
                        : ''
                    )
                    + change.toFixed(2)
                    + '%';

                setMetricTrend(
                    'live-24h-change',
                    change
                );


                document.getElementById(
                    'live-24h-high'
                ).innerText =
                    formatMarketPrice(
                        ticker.high_price
                    );


                document.getElementById(
                    'live-24h-low'
                ).innerText =
                    formatMarketPrice(
                        ticker.low_price
                    );


                const quoteVolume =
                    Number(
                        ticker.quote_volume
                    );

                document.getElementById(
                    'live-24h-volume'
                ).innerText =
                    quoteVolume
                    .toLocaleString(
                        undefined,
                        {
                            notation: 'compact',
                            maximumFractionDigits: 2
                        }
                    );
            }


            async function loadHistoricalMarketData(
                symbol,
                interval
            ) {
                const token =
                    localStorage.getItem(
                        'quant_token'
                    );

                const params =
                    new URLSearchParams({
                        symbol: symbol,
                        interval: interval,
                        limit: '200'
                    });

                const response =
                    await fetch(
                        '/api/market/klines?'
                        + params.toString(),
                        {
                            headers: {
                                'Authorization':
                                    'Bearer ' + token
                            }
                        }
                    );

                if (response.status === 401) {
                    logout();

                    throw new Error(
                        'Oturum süresi doldu.'
                    );
                }

                if (!response.ok) {
                    const body =
                        await response.json();

                    throw new Error(
                        body.detail
                        || 'Market geçmişi alınamadı.'
                    );
                }

                const data =
                    await response.json();

                const candles =
                    data.candles.map(
                        candle => ({
                            time:
                                Math.floor(
                                    candle.open_time_ms
                                    / 1000
                                ),

                            open:
                                Number(candle.open),

                            high:
                                Number(candle.high),

                            low:
                                Number(candle.low),

                            close:
                                Number(candle.close)
                        })
                    );

                liveCandlestickSeries.setData(
                    candles
                );

                liveSignalMarkers =
                    calculateCrossoverMarkers(
                        candles,
                        20,
                        50
                    );

                liveProvisionalMarker = null;

                liveCandlestickSeries.setMarkers(
                    liveSignalMarkers
                );

                const volumeData =
                    data.candles.map(
                        candle => {
                            const open =
                                Number(
                                    candle.open
                                );

                            const close =
                                Number(
                                    candle.close
                                );

                            return {
                                time:
                                    Math.floor(
                                        candle.open_time_ms
                                        / 1000
                                    ),

                                value:
                                    Number(
                                        candle.volume
                                    ),

                                color:
                                    close >= open
                                    ? 'rgba(0, 230, 118, 0.45)'
                                    : 'rgba(255, 82, 82, 0.45)'
                            };
                        }
                    );

                liveVolumeSeries.setData(
                    volumeData
                );

                liveSma20Series.setData(
                    calculateSmaSeries(
                        candles,
                        20
                    )
                );

                liveSma50Series.setData(
                    calculateSmaSeries(
                        candles,
                        50
                    )
                );

                const bollinger =
                    calculateBollingerSeries(
                        candles,
                        20,
                        2
                    );

                liveBbUpperSeries.setData(
                    bollinger.upper
                );

                liveBbMiddleSeries.setData(
                    bollinger.middle
                );

                liveBbLowerSeries.setData(
                    bollinger.lower
                );

                if (candles.length > 0) {
                    const last =
                        candles[
                            candles.length - 1
                        ];

                    liveLastClose =
                        last.close;

                    document.getElementById(
                        'live-price'
                    ).innerText =
                        formatMarketPrice(
                            last.close
                        );

                    document.getElementById(
                        'live-open'
                    ).innerText =
                        formatMarketPrice(
                            last.open
                        );

                    document.getElementById(
                        'live-high'
                    ).innerText =
                        formatMarketPrice(
                            last.high
                        );

                    document.getElementById(
                        'live-low'
                    ).innerText =
                        formatMarketPrice(
                            last.low
                        );
                }

                liveMarketChart
                    .timeScale()
                    .fitContent();
            }


            function updateLiveTechnicalAnalysis(
                technical,
                candleTime = null
            ) {
                if (
                    !technical
                    || !technical.ready
                ) {
                    return;
                }

                document.getElementById(
                    'live-rsi14'
                ).innerText =
                    Number(
                        technical.rsi14
                    ).toFixed(2);


                document.getElementById(
                    'live-macd'
                ).innerText =
                    Number(
                        technical.macd.macd
                    ).toFixed(4);


                document.getElementById(
                    'live-macd-signal'
                ).innerText =
                    Number(
                        technical.macd.signal
                    ).toFixed(4);


                const histogram =
                    Number(
                        technical.macd.histogram
                    );

                document.getElementById(
                    'live-macd-histogram'
                ).innerText =
                    (
                        histogram >= 0
                        ? '+'
                        : ''
                    )
                    + histogram.toFixed(4);

                setMetricTrend(
                    'live-macd-histogram',
                    histogram
                );


                const bb =
                    technical.bollinger;

                document.getElementById(
                    'live-bb-upper'
                ).innerText =
                    formatMarketPrice(
                        bb.upper
                    );

                document.getElementById(
                    'live-bb-middle'
                ).innerText =
                    formatMarketPrice(
                        bb.middle
                    );

                document.getElementById(
                    'live-bb-lower'
                ).innerText =
                    formatMarketPrice(
                        bb.lower
                    );

                document.getElementById(
                    'live-bb-width'
                ).innerText =
                    Number(
                        bb.bandwidth_pct
                    ).toFixed(4)
                    + '%';


                const score =
                    Number(
                        technical.score
                    );

                const scoreElement =
                    document.getElementById(
                        'live-technical-score'
                    );

                scoreElement.innerText =
                    (
                        score > 0
                        ? '+'
                        : ''
                    )
                    + score
                    + ' / '
                    + technical.max_score;

                setMetricTrend(
                    'live-technical-score',
                    score
                );


                const rating =
                    document.getElementById(
                        'live-technical-rating'
                    );

                const displayRating =
                    technical.rating.replace(
                        '_',
                        ' '
                    );

                rating.innerText =
                    displayRating;

                rating.className =
                    'metric-value '
                    + (
                        technical.rating
                        === 'BUY'
                        || technical.rating
                        === 'STRONG_BUY'
                        ? 'metric-positive'
                        : (
                            technical.rating
                            === 'SELL'
                            || technical.rating
                            === 'STRONG_SELL'
                            ? 'metric-negative'
                            : 'metric-neutral'
                        )
                    );


                const rsi =
                    Number(
                        technical.rsi14
                    );

                const rsiElement =
                    document.getElementById(
                        'live-rsi14'
                    );

                rsiElement.className =
                    'metric-value '
                    + (
                        rsi >= 70
                        ? 'metric-negative'
                        : (
                            rsi <= 30
                            ? 'metric-positive'
                            : 'metric-neutral'
                        )
                    );


                if (
                    candleTime !== null
                ) {
                    liveBbUpperSeries.update(
                        {
                            time:
                                candleTime,

                            value:
                                Number(
                                    bb.upper
                                )
                        }
                    );

                    liveBbMiddleSeries.update(
                        {
                            time:
                                candleTime,

                            value:
                                Number(
                                    bb.middle
                                )
                        }
                    );

                    liveBbLowerSeries.update(
                        {
                            time:
                                candleTime,

                            value:
                                Number(
                                    bb.lower
                                )
                        }
                    );
                }
            }


            function renderLiveSignalHistory() {
                const container =
                    document.getElementById(
                        'live-signal-history'
                    );

                const empty =
                    document.getElementById(
                        'live-signal-history-empty'
                    );

                container.innerHTML = '';

                if (
                    liveSignalHistory.length
                    === 0
                ) {
                    empty.style.display =
                        'block';

                    return;
                }

                empty.style.display =
                    'none';

                liveSignalHistory.forEach(
                    event => {
                        const row =
                            document.createElement(
                                'div'
                            );

                        row.className =
                            'signal-history-item';

                        const signalClass =
                            event.signal === 'BUY'
                            ? 'signal-history-buy'
                            : 'signal-history-sell';

                        row.innerHTML =
                            '<div class="signal-history-time">'
                            + event.timeLabel
                            + '</div>'

                            + '<div class="signal-history-symbol">'
                            + event.symbol
                            + '</div>'

                            + '<div class="signal-history-interval">'
                            + event.interval
                            + '</div>'

                            + '<div class="'
                            + signalClass
                            + '">'
                            + event.signal
                            + '</div>'

                            + '<div class="signal-history-price">'
                            + formatMarketPrice(
                                event.price
                            )
                            + '</div>';

                        container.appendChild(
                            row
                        );
                    }
                );
            }


            function clearLiveSignalHistory() {
                liveSignalHistory = [];

                renderLiveSignalHistory();
            }


            function showLiveSignalToast(
                event
            ) {
                const toast =
                    document.getElementById(
                        'live-signal-toast'
                    );

                toast.className =
                    'signal-toast '
                    + (
                        event.signal
                        === 'BUY'
                        ? 'buy'
                        : 'sell'
                    );

                toast.innerHTML =
                    '<strong>'
                    + event.signal
                    + ' CONFIRMED'
                    + '</strong>'
                    + '<br>'
                    + event.symbol
                    + ' · '
                    + event.interval
                    + '<br>'
                    + formatMarketPrice(
                        event.price
                    );

                requestAnimationFrame(
                    () => {
                        toast.classList.add(
                            'visible'
                        );
                    }
                );

                if (liveSignalToastTimer) {
                    clearTimeout(
                        liveSignalToastTimer
                    );
                }

                liveSignalToastTimer =
                    setTimeout(
                        () => {
                            toast.classList.remove(
                                'visible'
                            );
                        },
                        5000
                    );
            }


            function recordConfirmedCrossover(
                indicators,
                price
            ) {
                if (
                    !indicators
                    || !indicators.last_crossover
                ) {
                    return;
                }

                const crossover =
                    indicators.last_crossover;

                const key =
                    crossover.signal
                    + ':'
                    + crossover.time_ms;

                if (
                    key
                    === liveLastConfirmedCrossoverKey
                ) {
                    return;
                }

                liveLastConfirmedCrossoverKey =
                    key;

                const config =
                    liveMarketConfig || {
                        symbol:
                            document.getElementById(
                                'live-market-symbol'
                            ).value,

                        interval:
                            document.getElementById(
                                'live-market-interval'
                            ).value
                    };

                const event = {
                    signal:
                        crossover.signal,

                    time:
                        crossover.time,

                    timeLabel:
                        new Date(
                            crossover.time
                        ).toLocaleTimeString(
                            [],
                            {
                                hour: '2-digit',
                                minute: '2-digit'
                            }
                        ),

                    symbol:
                        config.symbol,

                    interval:
                        config.interval,

                    price:
                        Number(price)
                };

                liveSignalHistory.unshift(
                    event
                );

                liveSignalHistory =
                    liveSignalHistory.slice(
                        0,
                        20
                    );

                renderLiveSignalHistory();

                showLiveSignalToast(
                    event
                );
            }


            function updateLiveCrossoverMarker(
                indicators,
                candleTime
            ) {
                if (
                    !indicators
                    || candleTime === null
                ) {
                    return;
                }

                const crossover =
                    indicators.crossover;

                if (
                    crossover === 'HOLD'
                ) {
                    liveProvisionalMarker = null;

                    liveCandlestickSeries.setMarkers(
                        liveSignalMarkers
                    );

                    return;
                }

                const marker = {
                    time: candleTime,

                    position:
                        crossover === 'BUY'
                        ? 'belowBar'
                        : 'aboveBar',

                    color:
                        crossover === 'BUY'
                        ? '#00e676'
                        : '#ff5252',

                    shape:
                        crossover === 'BUY'
                        ? 'arrowUp'
                        : 'arrowDown',

                    text:
                        crossover
                        + (
                            indicators.candle_closed
                            ? ''
                            : ' LIVE'
                        )
                };

                if (
                    indicators.candle_closed
                ) {
                    const alreadyExists =
                        liveSignalMarkers.some(
                            existing =>
                                existing.time
                                === marker.time
                                && existing.text
                                === crossover
                        );

                    if (!alreadyExists) {
                        liveSignalMarkers.push(
                            {
                                ...marker,
                                text: crossover
                            }
                        );

                        liveSignalMarkers.sort(
                            (a, b) =>
                                a.time - b.time
                        );
                    }

                    liveProvisionalMarker = null;

                    liveCandlestickSeries.setMarkers(
                        liveSignalMarkers
                    );

                } else {
                    liveProvisionalMarker =
                        marker;

                    liveCandlestickSeries.setMarkers(
                        [
                            ...liveSignalMarkers,
                            liveProvisionalMarker
                        ].sort(
                            (a, b) =>
                                a.time - b.time
                        )
                    );
                }
            }


            function updateLiveIndicators(
                indicators,
                candleTime = null
            ) {
                if (
                    !indicators
                    || !indicators.ready
                ) {
                    return;
                }

                document.getElementById(
                    'live-sma20'
                ).innerText =
                    formatMarketPrice(
                        indicators.sma_short
                    );

                document.getElementById(
                    'live-sma50'
                ).innerText =
                    formatMarketPrice(
                        indicators.sma_long
                    );


                const signal =
                    document.getElementById(
                        'live-signal'
                    );

                signal.innerText =
                    indicators.signal;

                signal.className =
                    'metric-value '
                    + (
                        indicators.signal
                        === 'BUY'
                        ? 'metric-positive'
                        : (
                            indicators.signal
                            === 'SELL'
                            ? 'metric-negative'
                            : 'metric-neutral'
                        )
                    );


                const trend =
                    document.getElementById(
                        'live-trend'
                    );

                trend.innerText =
                    indicators.trend;

                trend.className =
                    'metric-value '
                    + (
                        indicators.trend
                        === 'BULLISH'
                        ? 'metric-positive'
                        : (
                            indicators.trend
                            === 'BEARISH'
                            ? 'metric-negative'
                            : 'metric-neutral'
                        )
                    );


                const spread =
                    Number(
                        indicators.spread_pct
                    );

                document.getElementById(
                    'live-sma-spread'
                ).innerText =
                    (
                        spread >= 0
                        ? '+'
                        : ''
                    )
                    + spread.toFixed(4)
                    + '%';

                setMetricTrend(
                    'live-sma-spread',
                    spread
                );


                const crossover =
                    document.getElementById(
                        'live-crossover'
                    );

                if (
                    indicators.crossover
                    === 'HOLD'
                ) {
                    crossover.innerText =
                        'NONE';

                    crossover.className =
                        'metric-value metric-neutral';

                } else {
                    crossover.innerText =
                        indicators.crossover
                        + (
                            indicators.candle_closed
                            ? ' ✓'
                            : ' LIVE'
                        );

                    crossover.className =
                        'metric-value '
                        + (
                            indicators.crossover
                            === 'BUY'
                            ? 'metric-positive'
                            : 'metric-negative'
                        );
                }


                const last =
                    indicators.last_crossover;

                const lastElement =
                    document.getElementById(
                        'live-last-crossover'
                    );

                if (last) {
                    lastElement.innerText =
                        last.signal
                        + ' · '
                        + new Date(
                            last.time
                        ).toLocaleString();

                    lastElement.className =
                        'metric-value '
                        + (
                            last.signal
                            === 'BUY'
                            ? 'metric-positive'
                            : 'metric-negative'
                        );

                } else {
                    lastElement.innerText =
                        'NONE';

                    lastElement.className =
                        'metric-value metric-neutral';
                }


                if (
                    candleTime !== null
                ) {
                    updateLiveCrossoverMarker(
                        indicators,
                        candleTime
                    );
                }

                if (
                    candleTime !== null
                    && indicators.sma_short !== null
                    && indicators.sma_long !== null
                ) {
                    liveSma20Series.update(
                        {
                            time: candleTime,
                            value:
                                Number(
                                    indicators.sma_short
                                )
                        }
                    );

                    liveSma50Series.update(
                        {
                            time: candleTime,
                            value:
                                Number(
                                    indicators.sma_long
                                )
                        }
                    );
                }
            }


            function scheduleLiveMarketReconnect() {
                if (!liveMarketConfig) {
                    return;
                }

                if (liveReconnectTimer) {
                    clearTimeout(
                        liveReconnectTimer
                    );
                }

                liveReconnectAttempts += 1;

                const delay =
                    Math.min(
                        1000
                        * Math.pow(
                            2,
                            liveReconnectAttempts - 1
                        ),
                        15000
                    );

                setLiveMarketStatus(
                    'connecting',
                    '● RECONNECTING'
                );

                liveReconnectTimer =
                    setTimeout(
                        () => {
                            connectLiveMarketSocket(
                                liveMarketConfig.symbol,
                                liveMarketConfig.interval
                            );
                        },
                        delay
                    );
            }


            function connectLiveMarketSocket(
                symbol,
                interval
            ) {
                if (liveMarketSocket) {
                    liveMarketSocket.onclose = null;
                    liveMarketSocket.close();
                    liveMarketSocket = null;
                }

                const protocol =
                    window.location.protocol
                    === 'https:'
                    ? 'wss'
                    : 'ws';

                const websocketUrl =
                    protocol
                    + '://'
                    + window.location.host
                    + '/ws/market/'
                    + encodeURIComponent(symbol)
                    + '?interval='
                    + encodeURIComponent(interval);

                setLiveMarketStatus(
                    'connecting',
                    '● CONNECTING'
                );

                liveMarketSocket =
                    new WebSocket(
                        websocketUrl
                    );

                liveMarketSocket.onopen =
                    () => {
                        setLiveMarketStatus(
                            'connecting',
                            '● CONNECTING'
                        );
                    };


                liveMarketSocket.onmessage =
                    event => {
                        const message =
                            JSON.parse(
                                event.data
                            );

                        if (
                            message.type
                            === 'connected'
                        ) {
                            setLiveMarketStatus(
                                'connected',
                                '● LIVE'
                            );

                            liveReconnectAttempts = 0;

                            updateLiveIndicators(
                                message.indicators
                            );

                            updateLiveTechnicalAnalysis(
                                message.technical
                            );

                            return;
                        }

                        if (
                            message.type
                            === 'error'
                        ) {
                            document.getElementById(
                                'live-market-error'
                            ).innerText =
                                message.detail;

                            return;
                        }

                        if (
                            message.type
                            !== 'kline'
                        ) {
                            return;
                        }

                        const candle =
                            message.data;

                        const close =
                            Number(
                                candle.close
                            );

                        const candleTime =
                            Math.floor(
                                candle.open_time_ms
                                / 1000
                            );

                        liveCandlestickSeries.update(
                            {
                                time: candleTime,

                                open:
                                    Number(
                                        candle.open
                                    ),

                                high:
                                    Number(
                                        candle.high
                                    ),

                                low:
                                    Number(
                                        candle.low
                                    ),

                                close: close
                            }
                        );

                        updateLiveIndicators(
                            message.indicators,
                            candleTime
                        );

                        updateLiveTechnicalAnalysis(
                            message.technical,
                            candleTime
                        );

                        recordConfirmedCrossover(
                            message.indicators,
                            close
                        );

                        liveVolumeSeries.update(
                            {
                                time:
                                    candleTime,

                                value:
                                    Number(
                                        candle.volume
                                    ),

                                color:
                                    close
                                    >= Number(
                                        candle.open
                                    )
                                    ? 'rgba(0, 230, 118, 0.45)'
                                    : 'rgba(255, 82, 82, 0.45)'
                            }
                        );

                        const priceElement =
                            document.getElementById(
                                'live-price'
                            );

                        priceElement.innerText =
                            formatMarketPrice(
                                close
                            );

                        if (
                            liveLastClose !== null
                        ) {
                            setMetricTrend(
                                'live-price',
                                close
                                - liveLastClose
                            );
                        }

                        liveLastClose =
                            close;

                        document.getElementById(
                            'live-open'
                        ).innerText =
                            formatMarketPrice(
                                candle.open
                            );

                        document.getElementById(
                            'live-high'
                        ).innerText =
                            formatMarketPrice(
                                candle.high
                            );

                        document.getElementById(
                            'live-low'
                        ).innerText =
                            formatMarketPrice(
                                candle.low
                            );

                        document.getElementById(
                            'live-volume'
                        ).innerText =
                            Number(
                                candle.volume
                            ).toLocaleString(
                                undefined,
                                {
                                    maximumFractionDigits: 4
                                }
                            );

                        document.getElementById(
                            'live-trade-count'
                        ).innerText =
                            Number(
                                candle.trade_count
                            ).toLocaleString();

                        document.getElementById(
                            'live-candle-status'
                        ).innerText =
                            candle.closed
                            ? 'CLOSED'
                            : 'LIVE';
                    };


                liveMarketSocket.onerror =
                    () => {
                        document.getElementById(
                            'live-market-error'
                        ).innerText =
                            'WebSocket bağlantı hatası.';
                    };


                liveMarketSocket.onclose =
                    () => {
                        setLiveMarketStatus(
                            'disconnected',
                            '● OFFLINE'
                        );

                        scheduleLiveMarketReconnect();
                    };
            }


            async function startLiveMarket() {
                const symbol =
                    document.getElementById(
                        'live-market-symbol'
                    ).value;

                const interval =
                    document.getElementById(
                        'live-market-interval'
                    ).value;

                const errorBox =
                    document.getElementById(
                        'live-market-error'
                    );

                errorBox.innerText = '';

                liveLastConfirmedCrossoverKey =
                    null;

                document.getElementById(
                    'live-symbol'
                ).innerText =
                    symbol;

                setLiveMarketStatus(
                    'connecting',
                    '● LOADING'
                );

                try {
                    initializeLiveMarketChart();

                    await Promise.all([
                        loadHistoricalMarketData(
                            symbol,
                            interval
                        ),
                        load24hMarketStats(
                            symbol
                        )
                    ]);

                    if (live24hStatsTimer) {
                        clearInterval(
                            live24hStatsTimer
                        );
                    }

                    live24hStatsTimer =
                        setInterval(
                            () => {
                                load24hMarketStats(
                                    symbol
                                ).catch(
                                    error => {
                                        console.error(
                                            "24h stats:",
                                            error
                                        );
                                    }
                                );
                            },
                            30000
                        );

                    liveReconnectAttempts = 0;

                    liveMarketConfig = {
                        symbol: symbol,
                        interval: interval
                    };

                    connectLiveMarketSocket(
                        symbol,
                        interval
                    );

                } catch (error) {
                    setLiveMarketStatus(
                        'disconnected',
                        '● OFFLINE'
                    );

                    errorBox.innerText =
                        'Live Market Hatası: '
                        + error.message;
                }
            }


            // --- GRAFİK ÇİZİM MANTIĞI ---
            function loadChartData() {
                const chartOptions = { 
                    layout: { textColor: '#d1d4dc', background: { type: 'solid', color: '#1E222D' } },
                    grid: { vertLines: { color: '#2B2B43' }, horzLines: { color: '#2B2B43' } },
                    crosshair: { mode: LightweightCharts.CrosshairMode.Normal }
                };
                
                // Eğer daha önce grafik çizildiyse temizle (çift çizimi engeller)
                document.getElementById('chart-container').innerHTML = '';
                const chart = LightweightCharts.createChart(document.getElementById('chart-container'), chartOptions);

                const priceSeries = chart.addLineSeries({ color: '#2962FF', lineWidth: 2, title: 'Fiyat' });
                const sma20Series = chart.addLineSeries({ color: '#FF6D00', lineWidth: 2, title: 'SMA 20' });
                const sma50Series = chart.addLineSeries({ color: '#00E676', lineWidth: 2, title: 'SMA 50' });

                // ÖNEMLİ: API'ye istek atarken yetki belgemizi (Token) gönderiyoruz!
                const token = localStorage.getItem('quant_token');
                
                fetch('/api/data?v=' + new Date().getTime(), {
                    headers: { 'Authorization': 'Bearer ' + token }
                })
                .then(response => {
                    if (response.status === 401) {
                        logout(); // Token süresi bitmişse zorla çıkış yaptır
                        throw new Error("Oturum süresi doldu.");
                    }
                    if (!response.ok) throw new Error("API Hatası");
                    return response.json();
                })
                .then(data => {
                    if (data.error) throw new Error(data.error);

                    const priceData = [];
                    const sma20Data = [];
                    const sma50Data = [];
                    const markers = [];

                    data.forEach(row => {
                        const time = row.Date;
                        priceData.push({ time: time, value: parseFloat(row.Close) });
                        if (row.SMA20 > 0) sma20Data.push({ time: time, value: parseFloat(row.SMA20) });
                        if (row.SMA50 > 0) sma50Data.push({ time: time, value: parseFloat(row.SMA50) });

                        if (row.Signal === 1) markers.push({ time: time, position: 'belowBar', color: '#00E676', shape: 'arrowUp', text: 'AL' });
                        else if (row.Signal === -1) markers.push({ time: time, position: 'aboveBar', color: '#FF5252', shape: 'arrowDown', text: 'SAT' });
                    });

                    priceSeries.setData(priceData);
                    sma20Series.setData(sma20Data);
                    sma50Series.setData(sma50Data);
                    priceSeries.setMarkers(markers);
                    chart.timeScale().fitContent();
                })
                .catch(err => {
                    document.getElementById('error-box').innerText = "Grafik Hatası: " + err.message;
                });
            }
        </script>
    </body>
    </html>
    """
    return html_content