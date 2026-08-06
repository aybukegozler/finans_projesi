#!/usr/bin/env bash

set -Eeuo pipefail

echo "=== Python bağımlılıkları kuruluyor ==="
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "=== C++ hesaplama motoru derleniyor ==="

if command -v g++ >/dev/null 2>&1; then
    CXX_COMPILER="g++"
elif command -v clang++ >/dev/null 2>&1; then
    CXX_COMPILER="clang++"
else
    echo "Hata: C++ derleyicisi bulunamadı."
    exit 1
fi

"$CXX_COMPILER" \
    -std=c++17 \
    -O2 \
    -Wall \
    -Wextra \
    -pedantic \
    src/engine.cpp \
    -o src/engine

chmod +x src/engine

echo "=== Finansal sinyaller oluşturuluyor ==="
./src/engine

if [[ ! -s data/signals.csv ]]; then
    echo "Hata: data/signals.csv oluşturulamadı."
    exit 1
fi

echo "=== Render build başarıyla tamamlandı ==="
