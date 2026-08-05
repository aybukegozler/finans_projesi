import yfinance as yf
import pandas as pd
import os

def fetch_and_save_data(ticker="AAPL", period="2y", filename="../data/market_data.csv"):
    print(f"[{ticker}] verisi çekiliyor...")
    df = yf.download(ticker, period=period)
    
    # Sadece Tarih ve Kapanış fiyatını alıyoruz, C++ için işleri basitleştiriyoruz
    df = df[['Close']].copy()
    df.columns = ['Close'] # Sütun adını temizle
    df.index.name = 'Date'
    
    # Veriyi CSV olarak data klasörüne kaydet
    filepath = os.path.join(os.path.dirname(__file__), filename)
    df.to_csv(filepath)
    print(f"Veri başarıyla kaydedildi: {filepath}")

if __name__ == "__main__":
    fetch_and_save_data()