import pandas as pd
import numpy as np

def calculate_kpis(df):
    """
    Sipariş veri setinden tedarikçi bazlı OTIF, Ortalama Gecikme ve Hata Oranı KPI'larını hesaplar.
    """
    df = df.copy()

    # 1. Gizli UTF-8 BOM Karakterlerini ve Boşlukları Sütun İsimlerinden Temizle
    df.columns = df.columns.astype(str).str.replace('\ufeff', '').str.strip()

    # 2. Esnek Sütun İsim Eşleştirmesi
    column_mapping = {
        'defective_quantity': 'defect_units',
        'accepted_quantity': 'accepted_units',
        'delivered_quantity': 'delivered_units',
        'order_quantity': 'order_units'
    }
    df = df.rename(columns=column_mapping)

    # 3. Hatalı Miktar Güvencesi
    if 'defect_units' not in df.columns:
        if 'accepted_units' in df.columns and 'order_units' in df.columns:
            df['defect_units'] = df['order_units'] - df['accepted_units']
        else:
            df['defect_units'] = 0

    # 4. Tarih Formatına Çevirme
    df['promised_date'] = pd.to_datetime(df['promised_date'], dayfirst=True, errors='coerce')
    df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'], dayfirst=True, errors='coerce')

    # 5. Gecikme Günü Hesaplama
    df['delay_days'] = (df['actual_delivery_date'] - df['promised_date']).dt.days
    df['delay_days'] = df['delay_days'].apply(lambda x: max(0, x) if pd.notnull(x) else 0)

    # 6. OTIF Koşulları
    df['is_on_time'] = df['delay_days'] == 0
    df['is_perfect'] = df['defect_units'] == 0
    df['is_otif'] = df['is_on_time'] & df['is_perfect']

    # 7. Sayma Sütunu Güvencesi
    count_col = 'order_id' if 'order_id' in df.columns else df.columns[0]
    qty_col = 'order_units' if 'order_units' in df.columns else 'order_quantity'

    # 8. Tedarikçi Bazlı İbre/Özet Hesaplama
    summary = df.groupby('supplier_name').agg(
        total_orders=(count_col, 'count'),
        otif_count=('is_otif', 'sum'),
        avg_delay_days=('delay_days', 'mean'),
        total_defects=('defect_units', 'sum'),
        total_qty=(qty_col, 'sum') if qty_col in df.columns else (df.columns[0], 'count')
    ).reset_index()

    summary['otif_rate'] = (summary['otif_count'] / summary['total_orders']) * 100
    summary['defect_rate'] = np.where(summary['total_qty'] > 0, (summary['total_defects'] / summary['total_qty']) * 100, 0)

    return summary