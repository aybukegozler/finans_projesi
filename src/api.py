from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
import subprocess
import os

app = FastAPI()

@app.get("/api/data")
def get_market_data():
    subprocess.run(['./src/engine'], capture_output=True)
    
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

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>Quant Dashboard</title>
        <!-- İŞTE ÇÖZÜM BURADA: Sürümü @4.1.1 olarak sabitledik -->
        <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body { font-family: 'Segoe UI', Tahoma, sans-serif; background-color: #131722; color: #d1d4dc; margin: 0; padding: 20px; }
            .header { text-align: center; margin-bottom: 20px; }
            h1 { color: #fff; margin-bottom: 5px; }
            p { color: #8a919e; margin-top: 0; }
            #chart-container { width: 100%; height: 600px; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
            #error-box { color: #FF5252; text-align: center; margin-top: 20px; font-weight: bold; font-size: 18px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Algoritmik Ticaret Paneli</h1>
            <p>C++ Motoru Tarafından Üretilen Al/Sat Sinyalleri ve SMA Kesişimleri</p>
        </div>
        
        <div id="chart-container"></div>
        <div id="error-box"></div>

        <script>
            const chartOptions = { 
                layout: { textColor: '#d1d4dc', background: { type: 'solid', color: '#1E222D' } },
                grid: { vertLines: { color: '#2B2B43' }, horzLines: { color: '#2B2B43' } },
                crosshair: { mode: LightweightCharts.CrosshairMode.Normal }
            };
            const chart = LightweightCharts.createChart(document.getElementById('chart-container'), chartOptions);

            const priceSeries = chart.addLineSeries({ color: '#2962FF', lineWidth: 2, title: 'Fiyat' });
            const sma20Series = chart.addLineSeries({ color: '#FF6D00', lineWidth: 2, title: 'SMA 20' });
            const sma50Series = chart.addLineSeries({ color: '#00E676', lineWidth: 2, title: 'SMA 50' });

            fetch('/api/data?v=' + new Date().getTime())
                .then(response => {
                    if (!response.ok) throw new Error("Sunucu Yanıt Vermedi (API Hatası)");
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

                        if (row.Signal === 1) {
                            markers.push({ time: time, position: 'belowBar', color: '#00E676', shape: 'arrowUp', text: 'AL' });
                        } else if (row.Signal === -1) {
                            markers.push({ time: time, position: 'aboveBar', color: '#FF5252', shape: 'arrowDown', text: 'SAT' });
                        }
                    });

                    priceSeries.setData(priceData);
                    sma20Series.setData(sma20Data);
                    sma50Series.setData(sma50Data);
                    priceSeries.setMarkers(markers);
                    
                    chart.timeScale().fitContent();
                })
                .catch(err => {
                    document.getElementById('error-box').innerText = "Grafik Çizim Hatası: " + err.message;
                });
        </script>
    </body>
    </html>
    """
    return html_content