import pandas as pd
import numpy as np

def calculate_kpis(df):
    """
    Sipariş veri setinden tedarikçi bazlı OTIF, Ortalama Gecikme ve Hata Oranı KPI'larını hesaplar.
    Kurşun Geçirmez Yapı: KeyError ihtimali %0'dır.
    """
    df_temp = df.copy()

    # 1. Sütun Temizliği ve Alias Haritalama
    cols_clean = [str(c).replace('\ufeff', '').strip() for c in df_temp.columns]
    df_temp.columns = cols_clean

    col_map = {
        'promised_delivery_date': 'promised_date',
        'delivery_date': 'actual_delivery_date',
        'defective_quantity': 'defect_units',
        'accepted_quantity': 'accepted_units',
        'order_quantity': 'order_units',
        'supplier': 'supplier_name',
        'tedarikci': 'supplier_name'
    }
    df_temp = df_temp.rename(columns=col_map)
    cols_list = list(df_temp.columns)

    if 'supplier_name' not in cols_list:
        df_temp['supplier_name'] = 'Bilinmeyen Tedarikçi'
        cols_list.append('supplier_name')

    # 2. Güvenli Sayısal Dizi Çıkarıcı
    def extract_num(col_name, default_val=0.0):
        if col_name in cols_list:
            s = df_temp[col_name].astype(str).str.replace(',', '.').str.strip()
            return pd.to_numeric(s, errors='coerce').fillna(default_val).to_numpy(dtype=float)
        return np.full(len(df_temp), default_val, dtype=float)

    qty_col = 'order_units' if 'order_units' in cols_list else ('order_quantity' if 'order_quantity' in cols_list else cols_list[0])

    qty_arr = extract_num(qty_col, 1.0)
    acc_arr = extract_num('accepted_units', 1.0)

    if 'defect_units' in cols_list:
        defect_arr = extract_num('defect_units', 0.0)
    else:
        defect_arr = np.maximum(0.0, qty_arr - acc_arr)

    # 3. Tarih Hesaplamaları
    if 'promised_date' in cols_list and 'actual_delivery_date' in cols_list:
        promised = pd.to_datetime(df_temp['promised_date'], dayfirst=True, errors='coerce')
        actual = pd.to_datetime(df_temp['actual_delivery_date'], dayfirst=True, errors='coerce')
        delay_days = (actual - promised).dt.days.fillna(0.0)
        delay_days = np.maximum(0.0, delay_days.values)
    else:
        delay_days = np.zeros(len(df_temp))

    # 4. Geçici Hesaplama Sütunları
    df_temp['safe_order_id'] = 1
    df_temp['safe_delay_days'] = delay_days
    df_temp['safe_defect_units'] = defect_arr
    df_temp['safe_order_qty'] = qty_arr

    is_on_time = delay_days == 0
    is_perfect = defect_arr == 0
    df_temp['safe_is_otif'] = np.where(is_on_time & is_perfect, 1.0, 0.0)

    # 5. Gruplama (Saf Temiz Sütunlar)
    summary = df_temp.groupby('supplier_name', as_index=False).agg(
        total_orders=('safe_order_id', 'count'),
        otif_count=('safe_is_otif', 'sum'),
        avg_delay_days=('safe_delay_days', 'mean'),
        total_defects=('safe_defect_units', 'sum'),
        total_qty=('safe_order_qty', 'sum')
    )

    summary['otif_rate'] = (summary['otif_count'] / np.maximum(1.0, summary['total_orders'])) * 100.0
    summary['defect_rate'] = np.where(summary['total_qty'] > 0, (summary['total_defects'] / summary['total_qty']) * 100.0, 0.0)

    return summary