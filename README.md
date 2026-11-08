# 📊 Tedarikçi Performans ve Risk Skorlama Sistemi

Bu proje, satın alma ve tedarik zinciri süreçlerinde tedarikçileri sadece fiyat odaklı değil; **teslimat performansı (OTIF)**, **kalite (Defect Rate)**, **gecikme trendleri** ve **çok kriterli karar verme (AHP & TOPSIS)** yöntemleriyle uçtan uca skorlayan analitik bir portfolyo uygulamasıdır.

## 🛠️ Mimari ve Proje Yapısı

```text
supplier-scorecard/
├── data/       # Sipariş, teslimat ve kalite verileri (CSV)
├── sql/        # OTIF ve Window Functions ile hareketli gecikme sorguları
├── src/
│   ├── kpi/       # OTIF, Lead Time ve Defect Rate hesaplama motoru
│   ├── scoring/   # Saaty AHP ağırlıklandırma & TOPSIS sıralama algoritmaları
│   ├── risk/      # Lojistik Regresyon ile teslimat gecikme tahmin modeli
│   └── reporting/ # Raporlama modülleri
├── app/        # Streamlit interaktif web dashboard
└── docs/       # Proje dokümantasyonu
