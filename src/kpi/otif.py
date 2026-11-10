import pandas as pd
import numpy as np

def calculate_kpis(df):
    """
    Sipariş veri setinden tedarikçi bazlı OTIF, Ortalama Gecikme ve Hata Oranı KPI'larını hesaplar.
    """
    df = df.copy()

    # 1. Sütun İsimleri Temizliği
    df.columns = df.columns.astype(str).str.replace('\ufeff', '').str.strip()

    # 2. Esnek Sütun Eşleştirmesi
    column_mapping = {
        'defective_quantity': 'defect_units',
        'accepted_quantity': 'accepted_units',
        'delivered_quantity': 'delivered_units',
        'order_quantity': 'order_units'
    }
    df = df.rename(columns=column_mapping)

    # 3. Sayısal Sütunları Temizleme & Float Yapma
    for col in ['order_units', 'accepted_units', 'defect_units', 'unit_price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)

    # 4. Hatalı Miktar Hesabı
    if 'defect_units' not in df.columns:
        if 'accepted_units' in df.columns and 'order_units' in df.columns:
            df['defect_units'] = np.maximum(0.0, df['order_units'] - df['accepted_units'])
        else:
            df['defect_units'] = 0.0

    # 5. Tarih Formatı Güvencesi (GG.AA.YYYY / YYYY-AA-GG Esnekliği)
    df['promised_date'] = pd.to_datetime(df['promised_date'], dayfirst=True, errors='coerce')
    df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'], dayfirst=True, errors='coerce')

    # 6. Gecikme Günü Hesaplama
    delay = (df['actual_delivery_date'] - df['promised_date']).dt.days
    df['delay_days'] = np.maximum(0.0, delay.fillna(0.0))

    # 7. OTIF Şartları
    df['is_on_time'] = df['delay_days'] == 0
    df['is_perfect'] = df['defect_units'] == 0
    df['is_otif'] = df['is_on_time'] & df['is_perfect']

    # 8. Sütun Seçimi
    count_col = 'order_id' if 'order_id' in df.columns else df.columns[0]
    qty_col = 'order_units' if 'order_units' in df.columns else df.columns[0]

    # 9. Tedarikçi Özet Tablosu
    summary = df.groupby('supplier_name', as_index=False).agg(
        total_orders=(count_col, 'count'),
        otif_count=('is_otif', 'sum'),
        avg_delay_days=('delay_days', 'mean'),
        total_defects=('defect_units', 'sum'),
        total_qty=(qty_col, 'sum')
    )

    summary['otif_rate'] = (summary['otif_count'] / np.maximum(1, summary['total_orders'])) * 100.0
    summary['defect_rate'] = np.where(summary['total_qty'] > 0, (summary['total_defects'] / summary['total_qty']) * 100.0, 0.0)

    return summary