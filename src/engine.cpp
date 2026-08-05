#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <numeric>

struct MarketData {
    std::string date;
    double close_price;
    double sma_20 = 0.0;
    double sma_50 = 0.0;
    int signal = 0; // 1: Al, -1: Sat, 0: Bekle
};

int main() {
    std::string filename = "data/market_data.csv";
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Hata: CSV dosyasi acilamadi!" << std::endl;
        return 1;
    }

    std::vector<MarketData> dataList;
    std::string line, word;
    std::getline(file, line); // Basligi atla

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        MarketData data;
        std::getline(ss, data.date, ',');
        std::getline(ss, word, ',');
        try {
            data.close_price = std::stod(word);
            dataList.push_back(data);
        } catch (...) {
            continue;
        }
    }
    file.close();

    int n = dataList.size();
    int short_window = 20;
    int long_window = 50;

    // Hareketli Ortalama (SMA) Hesaplama
    for (int i = 0; i < n; ++i) {
        if (i >= short_window - 1) {
            double sum = 0.0;
            for (int j = i - short_window + 1; j <= i; ++j) {
                sum += dataList[j].close_price;
            }
            dataList[i].sma_20 = sum / short_window;
        }

        if (i >= long_window - 1) {
            double sum = 0.0;
            for (int j = i - long_window + 1; j <= i; ++j) {
                sum += dataList[j].close_price;
            }
            dataList[i].sma_50 = sum / long_window;
        }
    }

    // Al/Sat Sinyallerini Üretme (Crossover Mantığı)
    int total_buy_signals = 0;
    int total_sell_signals = 0;

    for (int i = long_window; i < n; ++i) {
        bool prev_bullish = dataList[i-1].sma_20 > dataList[i-1].sma_50;
        bool curr_bullish = dataList[i].sma_20 > dataList[i].sma_50;

        // Kısa ortalama uzun ortalamayı aşağıdan yukarı kesti -> AL
        if (!prev_bullish && curr_bullish) {
            dataList[i].signal = 1;
            total_buy_signals++;
        }
        // Kısa ortalama uzun ortalamayı yukarıdan aşağı kesti -> SAT
        else if (prev_bullish && !curr_bullish) {
            dataList[i].signal = -1;
            total_sell_signals++;
        }
    }

    std::cout << "--- C++ Quant Engine Sinyal Analizi ---" << std::endl;
    std::cout << "Toplam İşlem Günü: " << n << std::endl;
    std::cout << "Üretilen AL Sinyali Sayısı: " << total_buy_signals << std::endl;
    std::cout << "Üretilen SAT Sinyali Sayısı: " << total_sell_signals << std::endl;

    // Son birkaç gündeki sinyalleri konsola yazdıralım
    std::cout << "\nSon 5 Günün Durumu:" << std::endl;
    for (int i = n - 5; i < n; ++i) {
        std::cout << dataList[i].date 
                  << " | Fiyat: $" << dataList[i].close_price 
                  << " | SMA20: " << dataList[i].sma_20 
                  << " | SMA50: " << dataList[i].sma_50;
        if (dataList[i].signal == 1) std::cout << " --> [ AL SİNYALİ ]";
        else if (dataList[i].signal == -1) std::cout << " --> [ SAT SİNYALİ ]";
        std::cout << std::endl;
    }

    return 0;
}