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