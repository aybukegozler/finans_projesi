import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. VERİ ÇEKİMİ
ticker_symbol = "AAPL"
print(f"{ticker_symbol} verileri çekiliyor...")
# Analizin net olması için son 2 yıllık veriyi alıyoruz
data = yf.download(ticker_symbol, period="2y")

# 2. HAREKETLİ ORTALAMALARI HESAPLAMA
short_window = 20
long_window = 50

data['SMA_20'] = data['Close'].rolling(window=short_window).mean()
data['SMA_50'] = data['Close'].rolling(window=long_window).mean()

# 3. ALGORİTMİK SİNYALLERİ ÜRETME (Mantık Katmanı)
# Önce sinyal kolonunu 0 ile dolduruyoruz
data['Signal'] = 0.0

# 20 günlük ortalama, 50 günlük ortalamadan büyükse 1 (Elde Tut/Al), değilse 0 (Sat/Bekle)
# İlk 50 günü atlıyoruz çünkü hesaplanması için o kadar süre geçmesi lazım
data.iloc[long_window:, data.columns.get_loc('Signal')] = np.where(
    data['SMA_20'][long_window:] > data['SMA_50'][long_window:], 1.0, 0.0
)

# Emirleri bulmak için Sinyalin türevini (farkını) alıyoruz. 
# 0'dan 1'e geçiş (1) = AL, 1'den 0'a geçiş (-1) = SAT
data['Position'] = data['Signal'].diff()

# 4. STRATEJİYİ TEST ETME (Backtesting)
# Piyasayı sadece elimizde tutsaydık ne kazanırdık? (Günlük Getiri)
data['Market_Return'] = data['Close'].pct_change()

# Bizim stratejimiz ne kazandırdı? (Sinyal 1 ise o günkü getiriyi alıyoruz)
# shift(1) yapıyoruz çünkü dünkü sinyale göre bugün pozisyondayız
data['Strategy_Return'] = data['Market_Return'] * data['Signal'].shift(1)

# Kümülatif (Toplam) Getirileri Hesaplama
data['Cumulative_Market'] = (1 + data['Market_Return']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()

print("\n=== BACKTEST SONUÇLARI ===")
print(f"Sadece Hissede Kalsaydık Toplam Getiri: %{(data['Cumulative_Market'].iloc[-1] - 1) * 100:.2f}")
print(f"Algoritmamızın Toplam Getirisi: %{(data['Cumulative_Strategy'].iloc[-1] - 1) * 100:.2f}")

# 5. GÖRSELLEŞTİRME
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

# Üst Grafik: Fiyat, Ortalamalar ve Al/Sat Noktaları
ax1.plot(data.index, data['Close'], label='Fiyat', alpha=0.5)
ax1.plot(data.index, data['SMA_20'], label='20 Günlük SMA', linestyle='--')
ax1.plot(data.index, data['SMA_50'], label='50 Günlük SMA', linestyle='--')

# AL Sinyallerini (Position == 1) Yeşil Ok ile çiz
buy_signals = data[data['Position'] == 1.0]
ax1.plot(buy_signals.index, data['SMA_20'][buy_signals.index], '^', markersize=10, color='g', label='AL Sinyali')

# SAT Sinyallerini (Position == -1) Kırmızı Ok ile çiz
sell_signals = data[data['Position'] == -1.0]
ax1.plot(sell_signals.index, data['SMA_20'][sell_signals.index], 'v', markersize=10, color='r', label='SAT Sinyali')

ax1.set_title(f"{ticker_symbol} Algoritmik Ticaret Simülasyonu")
ax1.set_ylabel('Fiyat ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Alt Grafik: Portföy Performansı (Biz vs Piyasa)
ax2.plot(data.index, data['Cumulative_Market'], label='Pasif Yatırım (Sadece Tut)', color='gray')
ax2.plot(data.index, data['Cumulative_Strategy'], label='Algoritma Performansı', color='purple')
ax2.set_title('Kümülatif Getiri Karşılaştırması')
ax2.set_ylabel('Getiri Çarpanı')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()