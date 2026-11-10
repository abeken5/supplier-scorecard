# 📊 Tedarikçi Performans ve Risk Skorlama Sistemi (Supplier Scorecard & Risk Engine)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Logistic%20Regression-orange)
![License](https://img.shields.io/badge/Status-Live-success)

Bu proje, modern tedarik zinciri ve talep planlama operasyonlarında tedarikçi performansını çok kriterli karar verme (MCDM) teknikleri, makine öğrenmesi ve olasılıksal simülasyon modelleri ile uçtan uca analiz eden ve optimizasyon sağlayan **endüstriyel seviyede bir Tedarik Zinciri Kontrol Kulesi** portfolyo uygulamasıdır.

---

## 🚀 Öne Çıkan Modüller ve Mimari

Proje 5 temel analitik katmandan oluşmaktadır:

### 1. 🏆 AHP & TOPSIS Çok Kriterli Skorlama (MCDM)
* **Analitik Hiyerarşi Süreci (AHP):** Kullanıcının dinamik olarak belirlediği kriter önem derecelerine göre tutarlılık oranını ($CR < 0.10$) kontrol ederek ağırlık matrisi türetir.
* **TOPSIS Yöntemi:** OTIF (Zamanında ve Eksiksiz Teslimat), Ortalam Gecikme Günü, Hata Oranı ve Fiyat kriterlerini İdeal Pozitif ve İdeal Negatif çözümlere olan uzaklıklarına göre vektörel normalize ederek tedarikçi sıralaması oluşturur.

### 2. 📦 Ürün Bazlı ABC/XYZ Stok & Talep Analizi
* **ABC Analizi (Ciro Katkısı):** Pareto ilkesine dayalı olarak ürünleri ciro paylarına göre A (%70), B (%20) ve C (%10) sınıflarına ayırır.
* **XYZ Analizi (Talep Oynaklığı):** Sipariş miktarlarının Değişim Katsayısı ($CV = \sigma / \mu$) ile talebin kararlılığını (X), sezonluğunu (Y) veya düzensizliğini (Z) sınıflandırır.
* **AX Önceliklendirmesi:** Kritik AX ürünlerini tedarik eden firmalar için gecikme toleransını dinamik olarak daraltır.

### 3. 🔮 Makine Öğrenmesi ile Gecikme Riski Tahmini
* **Lojistik Regresyon (Logistic Regression):** Geçmiş sipariş miktarları ve planlanan teslimat sürelerini (Lead Time) girdi alarak yeni bir siparişin gecikme olasılığını ($P(Delay)$) sınıflandırır.

### 4. 🎲 Monte Carlo Stok Kırılma (Stockout) Simülasyonu
* **1.000 Olasılıksal Senaryo:** Tedarikçinin teslimat süresi varyasyonunu (Normal Dağılım) simüle eder. Günlük talep ve mevcut depo stoğuna göre önümüzdeki süreçte stoksuz kalma olasılığını hesaba katar.

### 5. 🧩 Dinamik Sipariş Tahsis (Order Allocation) Öneri Motoru
* **Optimum Sipariş Bölüştürme:** TOPSIS başarı skoru yüksek ve gecikme riski düşük tedarikçilere ağırlık vererek, girilen toplam sipariş miktarını tedarikçiler arasında optimum yüzdelerle böler.

---

## 🛠️ Teknoloji Yığını

* **Programlama Dili:** Python
* **Veri Analizi & İşleme:** Pandas, NumPy
* **Makine Öğrenmesi & İstatistik:** Scikit-Learn (Logistic Regression, StandardScaler)
* **Kullanıcı Arayüzü & Görselleştirme:** Streamlit

---

## 📁 Proje Yapısı

```text
supplier-scorecard/
├── app/
│   └── main.py              # Streamlit Kullanıcı Arayüzü & Dashboard
├── data/
│   └── orders.csv           # Lojistik & Sipariş Veri Seti (200 Satır)
├── src/
│   ├── analytics/
│   │   ├── abc_xyz.py       # ABC/XYZ Matris Hesaplama
│   │   ├── allocation.py    # Sipariş Tahsis Motoru
│   │   └── monte_carlo.py    # Monte Carlo Simülasyon Motoru
│   ├── kpi/
│   │   └── otif.py          # OTIF & KPI Metrikleri
│   ├── risk/
│   │   └── delay_model.py   # Lojistik Regresyon Risk Modeli
│   └── scoring/
│       ├── ahp_weights.py   # AHP Ağırlıklandırma & CR Kontrolü
│       └── topsis.py        # TOPSIS Skorlama Algoritması
└── requirements.txt