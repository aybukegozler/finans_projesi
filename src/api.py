from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from datetime import datetime, timedelta
import pandas as pd
import subprocess
import os
import bcrypt

# Oluşturduğumuz veritabanı modülünü içe aktarıyoruz
from src.database import SessionLocal, User

# --- GÜVENLİK AYARLARI ---
SECRET_KEY = "finans_projesi_super_gizli_anahtar"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

app = FastAPI()
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
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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
    
    # ROL KONTROLÜ: Sadece Admin ise C++ Motorunu Tetikle!
    if current_user.role == "admin":
        subprocess.run(['./src/engine'], capture_output=True)
        print(f"[{current_user.username}] Admin yetkisiyle C++ motoru tetiklendi.")
    else:
        print(f"[{current_user.username}] Standart kullanıcı mevcut veriyi okuyor.")
        
    csv_path = "data/signals.csv"
    if not os.path.exists(csv_path):
        return {"error": "Sinyal dosyası bulunamadı."}
        
    df = pd.read_csv(csv_path)
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
            #chart-container { width: 100%; height: 600px; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
            #error-box { color: #FF5252; text-align: center; margin-top: 20px; font-weight: bold; }
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
                
                loadChartData(); // Grafiği çizmeye başla
            }

            // Sayfa yüklendiğinde oturum açık mı kontrol et
            if (localStorage.getItem('quant_token')) {
                showDashboard();
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