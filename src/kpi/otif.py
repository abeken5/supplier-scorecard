import pandas as pd
import numpy as np

def calculate_kpis(df):
    # Gecikme günü hesaplama
    df['promised_delivery_date'] = pd.to_datetime(df['promised_delivery_date'])
    df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'])
    df['delay_days'] = (df['actual_delivery_date'] - df['promised_delivery_date']).dt.days
    df['delay_days'] = df['delay_days'].apply(lambda x: max(0, x))

    # Tam ve Zamanında Teslimat Kontrolü
    df['is_on_time'] = df['actual_delivery_date'] <= df['promised_delivery_date']
    df['is_in_full'] = df['accepted_quantity'] >= df['order_quantity']
    df['is_otif'] = df['is_on_time'] & df['is_in_full']
    
    # Hata Oranı (Defect Rate)
    df['defective_units'] = df['order_quantity'] - df['accepted_quantity']

    # Tedarikçi Bazında Özet Metrikler
    summary = df.groupby(['supplier_id', 'supplier_name']).agg(
        total_orders=('order_id', 'count'),
        otif_rate=('is_otif', 'mean'),
        avg_delay_days=('delay_days', 'mean'),
        total_ordered=('order_quantity', 'sum'),
        total_defective=('defective_units', 'sum')
    ).reset_index()

    summary['defect_rate'] = summary['total_defective'] / summary['total_ordered']
    summary['otif_rate'] = summary['otif_rate'] * 100

    return summary

if __name__ == "__main__":
    orders_df = pd.read_csv("data/orders.csv")
    kpi_summary = calculate_kpis(orders_df)
    print("--- TEDARİKÇİ KPI ÖZETİ ---")
    print(kpi_summary[['supplier_name', 'otif_rate', 'avg_delay_days', 'defect_rate']])