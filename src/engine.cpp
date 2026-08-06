#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct MarketData {
    std::string date;
    double close_price = 0.0;
    double sma_20 = 0.0;
    double sma_50 = 0.0;
    int signal = 0; // 1: AL, -1: SAT, 0: BEKLE
};

int main() {
    const std::string input_path = "data/market_data.csv";
    const std::string output_path = "data/signals.csv";
    const std::string temp_output_path = "data/signals.tmp";

    constexpr std::size_t short_window = 20;
    constexpr std::size_t long_window = 50;

    std::ifstream input_file(input_path);

    if (!input_file.is_open()) {
        std::cerr << "Hata: " << input_path << " dosyasi acilamadi." << std::endl;
        return 1;
    }

    std::vector<MarketData> data_list;
    std::string line;

    // CSV başlığını atla.
    std::getline(input_file, line);

    while (std::getline(input_file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream row_stream(line);
        MarketData market_data;
        std::string close_value;

        std::getline(row_stream, market_data.date, ',');
        std::getline(row_stream, close_value, ',');

        if (market_data.date.empty() || close_value.empty()) {
            continue;
        }

        try {
            market_data.close_price = std::stod(close_value);
            data_list.push_back(market_data);
        } catch (const std::exception&) {
            std::cerr << "Uyari: Gecersiz satir atlandi: " << line << std::endl;
        }
    }

    input_file.close();

    if (data_list.size() < long_window) {
        std::cerr
            << "Hata: SMA50 hesaplamak icin en az "
            << long_window
            << " satir gerekir. Mevcut satir: "
            << data_list.size()
            << std::endl;
        return 1;
    }

    // Hareketli ortalamalar kayan toplam kullanılarak O(n) zamanda hesaplanır.
    double short_sum = 0.0;
    double long_sum = 0.0;

    for (std::size_t i = 0; i < data_list.size(); ++i) {
        short_sum += data_list[i].close_price;
        long_sum += data_list[i].close_price;

        if (i >= short_window) {
            short_sum -= data_list[i - short_window].close_price;
        }

        if (i >= long_window) {
            long_sum -= data_list[i - long_window].close_price;
        }

        if (i + 1 >= short_window) {
            data_list[i].sma_20 =
                short_sum / static_cast<double>(short_window);
        }

        if (i + 1 >= long_window) {
            data_list[i].sma_50 =
                long_sum / static_cast<double>(long_window);
        }
    }

    int total_buy_signals = 0;
    int total_sell_signals = 0;

    // SMA20, SMA50'yi aşağıdan yukarı keserse AL;
    // yukarıdan aşağı keserse SAT sinyali üret.
    for (std::size_t i = long_window; i < data_list.size(); ++i) {
        const bool previous_bullish =
            data_list[i - 1].sma_20 > data_list[i - 1].sma_50;

        const bool current_bullish =
            data_list[i].sma_20 > data_list[i].sma_50;

        if (!previous_bullish && current_bullish) {
            data_list[i].signal = 1;
            ++total_buy_signals;
        } else if (previous_bullish && !current_bullish) {
            data_list[i].signal = -1;
            ++total_sell_signals;
        }
    }

    std::filesystem::create_directories("data");

    // Önce geçici dosyaya yazılır; işlem tamamlanınca asıl dosyayla değiştirilir.
    std::ofstream output_file(temp_output_path);

    if (!output_file.is_open()) {
        std::cerr
            << "Hata: "
            << temp_output_path
            << " dosyasi olusturulamadi."
            << std::endl;
        return 1;
    }

    output_file << "Date,Close,SMA20,SMA50,Signal\n";
    output_file << std::fixed << std::setprecision(6);

    for (const MarketData& row : data_list) {
        output_file
            << row.date << ','
            << row.close_price << ','
            << row.sma_20 << ','
            << row.sma_50 << ','
            << row.signal << '\n';
    }

    output_file.close();

    try {
        if (std::filesystem::exists(output_path)) {
            std::filesystem::remove(output_path);
        }

        std::filesystem::rename(temp_output_path, output_path);
    } catch (const std::filesystem::filesystem_error& error) {
        std::cerr
            << "Hata: Sinyal dosyasi yayinlanamadi: "
            << error.what()
            << std::endl;
        return 1;
    }

    std::cout << "--- C++ Quant Engine ---" << std::endl;
    std::cout << "Islenen gun sayisi: " << data_list.size() << std::endl;
    std::cout << "AL sinyali sayisi: " << total_buy_signals << std::endl;
    std::cout << "SAT sinyali sayisi: " << total_sell_signals << std::endl;
    std::cout << "Cikti dosyasi: " << output_path << std::endl;

    const std::size_t preview_count =
        std::min<std::size_t>(5, data_list.size());

    std::cout << "\nSon " << preview_count << " gun:" << std::endl;

    for (
        std::size_t i = data_list.size() - preview_count;
        i < data_list.size();
        ++i
    ) {
        std::cout
            << data_list[i].date
            << " | Close: " << data_list[i].close_price
            << " | SMA20: " << data_list[i].sma_20
            << " | SMA50: " << data_list[i].sma_50
            << " | Signal: " << data_list[i].signal
            << std::endl;
    }

    return 0;
}
