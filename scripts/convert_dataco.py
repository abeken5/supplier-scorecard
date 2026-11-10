import pandas as pd
import numpy as np
import os

# Yol Tanımlamaları
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "DataCoSupplyChainDataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "orders.csv")

def process_dataco():
    print("⏳ DataCo veri seti okunuyor ve işleniyor...")
    
    # Veriyi oku (encoding hatalarına karşı latin1/utf-8-sig)
    try:
        df_raw = pd.read_csv(INPUT_PATH, encoding='latin1')
    except Exception:
        df_raw = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')

    # İlk 1.500 satırı al (Performans ve temsil gücü için)
    df = df_raw.head(1500).copy()

    # DataCo Sütunlarını Bizim DSS Mimaramıza Eşleme
    # DataCo'daki Tedarikçi/Bölge/Kategori mantığını Tedarikçilere atıyoruz
    supplier_mapping = {
        'EE. UU.': 'Apex Global Logistics',
        'Puerto Rico': 'Bursa Parça Sanayi',
        'Caguas': 'TeknoLojistik A.Ş.',
        'Chicago': 'Ege Ambalaj Sistemleri',
        'Los Angeles': 'Global Freight Co.'
    }

    # Sütun dönüşümleri
    df['supplier_name'] = df['Order Country'].map(supplier_mapping).fillna('Global Freight Co.')
    df['product_name'] = df['Category Name'].astype(str)
    df['order_quantity'] = pd.to_numeric(df['Order Item Quantity'], errors='coerce').fillna(1).astype(int)
    df['unit_price'] = pd.to_numeric(df['Order Item Product Price'], errors='coerce').fillna(25.0)

    # Tarih ve Teslimat Süreleri (DataCo Gerçek & Planlanan Teslimat Süreleri)
    df['order_date'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
    
    # DataCo'daki planlanan ve gerçekleşen nakliye günleri
    scheduled_days = pd.to_numeric(df['Days for shipment (scheduled)'], errors='coerce').fillna(4).astype(int)
    real_days = pd.to_numeric(df['Days for shipping (real)'], errors='coerce').fillna(4).astype(int)

    df['promised_delivery_date'] = df['order_date'] + pd.to_timedelta(scheduled_days, unit='D')
    df['actual_delivery_date'] = df['order_date'] + pd.to_timedelta(real_days, unit='D')

    # Tarih formatlarını düzeltme (DD.MM.YYYY)
    df['order_date'] = df['order_date'].dt.strftime('%d.%m.%Y')
    df['promised_delivery_date'] = df['promised_delivery_date'].dt.strftime('%d.%m.%Y')
    df['actual_delivery_date'] = df['actual_delivery_date'].dt.strftime('%d.%m.%Y')

    # Kalite / Kusurlu Ürün ve Finans Sütunları
    # Late_delivery_risk ve Delivery Status üzerinden sentetik kusur türetme
    df['accepted_quantity'] = np.where(df['Delivery Status'] == 'ADVANCE SHIPPING', df['order_quantity'], np.maximum(0, df['order_quantity'] - np.random.choice([0, 1, 2], size=len(df), p=[0.85, 0.10, 0.05])))
    
    # TCO Parametreleri
    df['freight_cost'] = pd.to_numeric(df['Order Zipcode'], errors='coerce').fillna(120.0) % 150 + 50.0
    df['customs_duty_rate'] = 0.05
    df['payment_terms_days'] = np.random.choice([30, 45, 60, 90], size=len(df), p=[0.4, 0.3, 0.2, 0.1])
    df['lead_time_std'] = 1.5
    df['rework_cost_factor'] = 1.4
    df['financial_health'] = np.random.choice(['A', 'B', 'C'], size=len(df), p=[0.5, 0.35, 0.15])

    # İhtiyacımız Olan Sütun Seçimi
    final_cols = [
        'supplier_name', 'product_name', 'order_date', 'promised_delivery_date',
        'actual_delivery_date', 'order_quantity', 'accepted_quantity', 'unit_price',
        'freight_cost', 'customs_duty_rate', 'payment_terms_days', 'lead_time_std',
        'rework_cost_factor', 'financial_health'
    ]

    df_final = df[final_cols]
    df_final.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ Başarılı! {len(df_final)} satırlık gerçek DataCo veri seti 'data/orders.csv' olarak oluşturuldu.")

if __name__ == "__main__":
    process_dataco()