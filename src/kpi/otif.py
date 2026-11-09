import pandas as pd
import numpy as np

def calculate_kpis(df):
    """
    Sipariş verisi üzerinden OTIF (On-Time In-Full), ortalama gecikme ve hata oranını hesaplar.
    """
    # Tarih dönüşümleri
    df['promised_delivery_date'] = pd.to_datetime(df['promised_delivery_date'])
    df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'])
    
    # 1. Zamanında Teslimat Kontrolü (On-Time)
    df['is_on_time'] = df['actual_delivery_date'] <= df['promised_delivery_date']
    
    # 2. Tam Miktar Teslimat Kontrolü (In-Full)
    df['is_in_full'] = df['accepted_quantity'] >= df['order_quantity']
    
    # 3. OTIF Metriği (Hem Zamanında Hem Tam)
    df['is_otif'] = df['is_on_time'] & df['is_in_full']
    
    # 4. Gecikme Günü Hesaplama
    df['delay_days'] = (df['actual_delivery_date'] - df['promised_delivery_date']).dt.days
    df['delay_days'] = df['delay_days'].apply(lambda x: max(0, x))
    
    # 5. Hata Oranı (Defect Rate)
    df['defect_units'] = df['order_quantity'] - df['accepted_quantity']
    
    # Tedarikçi bazında özetleme (Sadece supplier_name ile gruplama)
    summary = df.groupby('supplier_name').agg(
        total_orders=('order_id', 'count'),
        otif_orders=('is_otif', 'sum'),
        avg_delay_days=('delay_days', 'mean'),
        total_ordered=('order_quantity', 'sum'),
        total_defects=('defect_units', 'sum')
    ).reset_index()
    
    summary['otif_rate'] = (summary['otif_orders'] / summary['total_orders']) * 100
    summary['defect_rate'] = summary['total_defects'] / summary['total_ordered']
    
    return summary

if __name__ == "__main__":
    df = pd.read_csv("data/orders.csv")
    kpis = calculate_kpis(df)
    print("=== TEDARİKÇİ KPI ÖZETİ ===")
    print(kpis[['supplier_name', 'total_orders', 'otif_rate', 'avg_delay_days', 'defect_rate']])