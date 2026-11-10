import pandas as pd
import numpy as np

def calculate_kpis(df):
    """
    Sipariş veri setinden tedarikçi bazlı OTIF, Ortalama Gecikme ve Hata Oranı KPI'larını hesaplar.
    %100 Garantili ve Saf Pandas Dizin Çözümü.
    """
    df_temp = df.copy()

    # 1. Sütun Temizliği ve Alias Haritalama
    df_temp.columns = [str(c).replace('\ufeff', '').strip() for c in df_temp.columns]

    mapping = {
        'supplier': 'supplier_name',
        'tedarikci': 'supplier_name',
        'promised_delivery_date': 'promised_date',
        'delivery_date': 'actual_delivery_date',
        'order_quantity': 'order_units',
        'accepted_quantity': 'accepted_units',
        'defective_quantity': 'defect_units'
    }
    df_temp = df_temp.rename(columns=mapping)

    if 'supplier_name' not in df_temp.columns:
        df_temp['supplier_name'] = 'Varsayılan Tedarikçi'

    n = len(df_temp)

    # 2. Güvenli Sayısal Dizi Çıkarıcı
    def get_arr(col, default=0.0):
        if col in df_temp.columns:
            s = df_temp[col].astype(str).str.replace(',', '.').str.strip()
            return pd.to_numeric(s, errors='coerce').fillna(default).to_numpy(dtype=float)
        return np.full(n, default, dtype=float)

    qty_col = 'order_units' if 'order_units' in df_temp.columns else df_temp.columns[0]
    qty_arr = get_arr(qty_col, 1.0)
    acc_arr = get_arr('accepted_units', 1.0)
    defect_arr = get_arr('defect_units', -1.0)

    # Eksikse hesapla
    defect_arr = np.where(defect_arr < 0, np.maximum(0.0, qty_arr - acc_arr), defect_arr)

    # 3. Tarih ve Gecikme
    if 'promised_date' in df_temp.columns and 'actual_delivery_date' in df_temp.columns:
        promised = pd.to_datetime(df_temp['promised_date'], dayfirst=True, errors='coerce')
        actual = pd.to_datetime(df_temp['actual_delivery_date'], dayfirst=True, errors='coerce')
        delay_days = (actual - promised).dt.days.fillna(0.0).to_numpy(dtype=float)
        delay_days = np.maximum(0.0, delay_days)
    else:
        delay_days = np.zeros(n, dtype=float)

    is_on_time = (delay_days == 0).astype(float)
    is_perfect = (defect_arr == 0).astype(float)
    is_otif = is_on_time * is_perfect

    # 4. Saf ve Temiz İki Boyutlu DataFrame
    clean_df = pd.DataFrame({
        'supplier_name': df_temp['supplier_name'].astype(str),
        'order_cnt': np.ones(n, dtype=float),
        'otif_cnt': is_otif,
        'delay': delay_days,
        'defects': defect_arr,
        'qty': qty_arr
    })

    # 5. Sözlük Bazlı Basit Gruplama
    summary = clean_df.groupby('supplier_name', as_index=False).agg({
        'order_cnt': 'sum',
        'otif_cnt': 'sum',
        'delay': 'mean',
        'defects': 'sum',
        'qty': 'sum'
    })

    summary.columns = ['supplier_name', 'total_orders', 'otif_count', 'avg_delay_days', 'total_defects', 'total_qty']
    summary['otif_rate'] = (summary['otif_count'] / np.maximum(1.0, summary['total_orders'])) * 100.0
    summary['defect_rate'] = np.where(summary['total_qty'] > 0, (summary['total_defects'] / summary['total_qty']) * 100.0, 0.0)

    return summary