import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. Sembolü belirleyin (Örn: Apple için 'AAPL', BIST 100 için 'XU100.IS', Bitcoin için 'BTC-USD')
ticker_symbol = "AAPL"

# 2. Ticker objesini oluşturun
ticker = yf.Ticker(ticker_symbol)

# --- ANLIK / CANLI VERİ BİLGİSİ ---
# Şirketin veya varlığın güncel bilgilerini çeker
info = ticker.info
print(f"=== {info.get('longName', ticker_symbol)} ===")
print(f"Güncel Fiyat: ${info.get('currentPrice') or info.get('regularMarketPrice')}")
print(f"52 Haftalık En Yüksek: ${info.get('fiftyTwoWeekHigh')}")
print(f"52 Haftalık En Düşük: ${info.get('fiftyTwoWeekLow')}\n")

# --- GEÇMİŞ FİYAT VERİLERİ (HISTORICAL DATA) ---
# Son 1 yıllık günlük veriyi çekelim
df = ticker.history(period="1y")

# DataFrame'in ilk 5 satırına göz atalım
print("Son 5 Günlük İşlem Verisi:")
print(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail())

# --- BASİT FİNANSAL ANALİZ METRİKLERİ ---

# A. Günlük Yüzdesel Getiri (Daily Returns)
df['Daily_Return'] = df['Close'].pct_change()

# B. Short-term (20 günlük) ve Long-term (50 günlük) Basit Hareketli Ortalama (SMA)
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# C. Volatilite (Standart Sapma - Son 20 günlük)
df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()

print("\n--- Analiz Sonuç Özeti ---")
print(f"Yıllık Toplam Getiri: %{((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}")
print(f"Ortalama Günlük Volatilite: %{df['Daily_Return'].std() * 100:.2f}")

# --- GÖRSELLEŞTİRME ---
plt.figure(figsize=(12, 6))

# Kapanış fiyatı ve Hareketli Ortalamalar
plt.plot(df.index, df['Close'], label='Kapanış Fiyatı', alpha=0.6)
plt.plot(df.index, df['SMA_20'], label='20 Günlük SMA (Kısa Vadeli Trend)', linestyle='--')
plt.plot(df.index, df['SMA_50'], label='50 Günlük SMA (Uzun Vadeli Trend)', linestyle='--')

plt.title(f"{ticker_symbol} Fiyat Analizi ve Hareketli Ortalamalar")
plt.xlabel("Tarih")
plt.ylabel("Fiyat ($)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()