# 📊 Tedarikçi Performans ve Risk Skorlama Sistemi (DSS Prototipi)

[![Live Demo](https://img.shields.io/badge/Streamlit-Canlı%20Demo-red?logo=streamlit)](https://supplier-scorecard-ngn7vbvuuvdvtfnsjeqmq5j.streamlit.app/#tedarikci-siralamasi)
[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Decision%20Support%20Prototype-green)]()

> **Tedarikçi seçimi, gecikme riski ve sipariş tahsisi için AHP/TOPSIS, lojistik regresyon, Monte Carlo ve maliyet modellerini birleştiren bir karar destek prototipi.**

---

## 🔗 Canlı Demo
🚀 **Uygulamayı Canlı Deneyin:** [Supplier Scorecard & DSS App](https://supplier-scorecard-ngn7vbvuuvdvtfnsjeqmq5j.streamlit.app/#tedarikci-siralamasi)

---

## 💡 Çözülen İş Problemi: Fiyata Göre Seçim Neden Yanıltıcı?

Geleneksel satın alma süreçlerinde tedarikçiler sıklıkla sadece **yalın birim fiyat** üzerinden değerlendirilir. Ancak en düşük teklifi veren firma; yüksek gümrük vergileri, lojistik maliyetleri, elverişsiz ödeme vadeleri (işletme sermayesi finansman yükü), kalite kusurları ve teslimat gecikmeleri eklendiğinde **en pahalı seçeneğe dönüşebilir.**

Bu karar destek paneli, **Yalın Birim Fiyat** sıralaması ile **Birim TCO (Toplam Sahip Olma Maliyeti)** sıralaması arasındaki stratejik sapmayı ortaya koyarak karar vericilere objektif bir analitik katman sağlar.

---

## 🛠️ Metodoloji ve Modüller

### 1. 💵 Toplam Sahip Olma Maliyeti (TCO) & Finansal Katman
Tedarikçi maliyetini sadece fatura fiyatıyla değil, tüm gizli maliyet bileşenleriyle modeler:
$$TCO = Landed\_Cost + Expected\_Delay\_Cost + Quality\_Cost + Working\_Capital\_Cost$$
* **Landed Cost:** Satın alma fiyatı, gümrük vergisi ve navlun bedelini birleştirir.
* **Finansal Sağlık & Vade (WACC):** Tedarikçi ödeme vadesi ve emniyet stoğunun işletme sermayesi üzerindeki yükünü Ağırlıklı Ortalama Sermaye Maliyeti (WACC) ile hesaba katar.

### 2. 🏆 AHP & TOPSIS Çok Kriterli Tedarikçi Skorlama
* **AHP (Analytic Hierarchy Process):** Kullanıcı tanımlı OTIF, Gecikme, Kalite ve TCO kriter ağırlıklarının tutarlılığını matrisel olarak doğrular. Tutarlılık Oranı $CR = \frac{CI}{RI} < 0.10$ koşulu aranır.
* **TOPSIS:** Tedarikçileri ideal pozitif çözüme yakınlık ($C_i^*$) katsayısına göre sıralar:
  $$C_i^* = \frac{S_i^-}{S_i^* + S_i^-}$$

### 3. 📦 ABC / XYZ Stok & Talep Analizi
Ürün gruplarını harcama hacmine (ABC) ve talep değişkenliğine (CV - XYZ) göre 9'lu matriste sınıflandırarak kritik malzeme risklerini tespit eder.

### 4. 🔮 Yapay Zeka Tabanlı Gecikme Risk Tahmini
Lojistik regresyon modeli; sipariş miktarı ve planlanan teslimat süresini girdi alarak sipariş bazlı gecikme olasılığını ($P(Gecikme)$) tahmin eder.

### 5. 🎲 Monte Carlo Stoksuz Kalma (Stockout) Simülasyonu
Tedarikçi teslimat sürelerindeki değişkenliği 1.000 senaryo üzerinden simüle ederek depodaki stok kırılma olasılığını hesaplar.

---

## ⚠️ Varsayımlar ve Sınırlılıklar (Limitations)

Bu araç bir karar verme mercii değil, **karar destek amaçlı bir prototiptir.**
* **Veri Yapısı:** Demo verisi DataCo Global Supply Chain veri seti baz alınarak filtrelenmiştir (1.500 sipariş). Gerçek tedarikçi performansını değil, yöntemin uygulanabilirliğini gösterir.
* **Gecikme Modeli:** Lojistik regresyon modeli sınırlı sayıda özelliğe (miktar ve lead time) dayanır. Sınıf dengesizliği önlemleri alınmış olup AUC/ROC kalibrasyon metrikleri henüz raporlanmamıştır.
* **TOPSIS Sıralaması (Rank Reversal):** TOPSIS metodolojisi gereği sisteme yeni bir tedarikçi eklendiğinde mevcut tedarikçilerin sıralamasında değişme (rank reversal) gözlemlenebilir.
* **Monte Carlo Varsayımları:** Simülasyondaki teslimat süresi dağılımları teorik varsayımlara dayanır; gerçek saha entegrasyonunda tarihsel parametrik dağılımlarla uydurulmalıdır.
* **Tahsis Motoru:** Sipariş tahsis algoritması sezgisel (heuristic) bir kısıtlı tahsis önerisidir; tam bir doğrusal/tamsayılı optimizasyon (LP/IP) çözücüsü değildir.
* **Sektörel Kriterler:** Savunma sanayii ve havacılık sektörlerine özgü kritik kriterler (AS9100/ISO sertifikasyonu, tek kaynak riski, alt tedarikçi bağımlılığı ve ihrakat kontrolü/ITAR) henüz modele dahil edilmemiştir.

---

## 🏗️ Proje Mimarisi

```text
supplier-scorecard/
├── app/
│   └── main.py              # Streamlit kullanıcı arayüzü ve ana акış
├── src/
│   ├── kpi/                 # OTIF, Gecikme ve Hata Oranı hesaplayıcı
│   ├── finance/             # TCO ve WACC finansal çalışma sermayesi modeli
│   ├── scoring/             # AHP ve TOPSIS skorlama algoritmaları
│   ├── risk/                # Sklearn Lojistik Regresyon risk tahmincisi
│   └── analytics/           # ABC/XYZ, Monte Carlo ve Tahsis motoru
├── data/
│   └── orders.csv           # İşlenmiş tedarik zinciri veri seti
├── scripts/
│   └── convert_dataco.py    # DataCo veri seti dönüştürme script'i
├── tests/
│   └── test_core.py         # AHP, TOPSIS ve KPI birim testleri
├── LICENSE                  # MIT Lisansı
└── requirements.txt         # Bağımlılıklar