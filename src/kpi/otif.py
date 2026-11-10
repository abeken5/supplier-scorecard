import pandas as pd
import numpy as np

def calculate_kpis(df):
    """
    Sipariş veri setinden tedarikçi bazlı OTIF, Ortalama Gecikme ve Hata Oranı KPI'larını hesaplar.
    """
    df = df.copy()

    # 1. UTF-8 BOM Karakterlerini ve Boşlukları Sütun İsimlerinden Temizle
    df.columns = df.columns.astype(str).str.replace('\ufeff', '').str.strip()

    # 2. Esnek Sütun İsim Eşleştirmesi
    column_mapping = {
        'defective_quantity': 'defect_units',
        'accepted_quantity': 'accepted_units',
        'delivered_quantity': 'delivered_units',
        'order_quantity': 'order_units'
    }
    df = df.rename(columns=column_mapping)

    # 3. Sayısal Sütunları Güvenli Sayıya Çevirme (Virgül -> Nokta Dönüşümü)
    numeric_cols = ['order_units', 'accepted_units', 'defect_units', 'unit_price']
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 4. Hatalı Miktar Güvencesi
    if 'defect_units' not in df.columns:
        if 'accepted_units' in df.columns and 'order_units' in df.columns:
            df['defect_units'] = df['order_units'] - df['accepted_units']
        else:
            df['defect_units'] = 0
            
    df['defect_units'] = df['defect_units'].apply(lambda x: max(0, x))

    # 5. Tarih Formatına Çevirme (Akıllı Format Algılama)
    df['promised_date'] = pd.to_datetime(df['promised_date'], dayfirst=True, errors='coerce')
    df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'], dayfirst=True, errors='coerce')

    # 6. Gecikme Günü Hesaplama
    df['delay_days'] = (df['actual_delivery_date'] - df['promised_date']).dt.days
    df['delay_days'] = df['delay_days'].apply(lambda x: max(0, x) if pd.notnull(x) else 0)

    # 7. OTIF Koşulları
    df['is_on_time'] = df['delay_days'] == 0
    df['is_perfect'] = df['defect_units'] == 0
    df['is_otif'] = df['is_on_time'] & df['is_perfect']

    # 8. Sütun Seçim Güvencesi
    count_col = 'order_id' if 'order_id' in df.columns else df.columns[0]
    qty_col = 'order_units' if 'order_units' in df.columns else df.columns[0]

    # 9. Tedarikçi Bazlı Özet Hesaplama
    summary = df.groupby('supplier_name').agg(
        total_orders=(count_col, 'count'),
        otif_count=('is_otif', 'sum'),
        avg_delay_days=('delay_days', 'mean'),
        total_defects=('defect_units', 'sum'),
        total_qty=(qty_col, 'sum')
    ).reset_index()

    summary['otif_rate'] = (summary['otif_count'] / summary['total_orders']) * 100
    summary['defect_rate'] = np.where(summary['total_qty'] > 0, (summary['total_defects'] / summary['total_qty']) * 100, 0)

    return summary