import pandas as pd
import numpy as np

def calculate_kpis(df):
    df_temp = df.copy()

    # Sütun isimlerini stringe çevirip temizle
    df_temp.columns = [str(c).replace('\ufeff', '').strip() for c in df_temp.columns]

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

    if 'supplier_name' not in df_temp.columns:
        df_temp['supplier_name'] = 'Varsayılan Tedarikçi'

    def get_num(col_name, default_val=0.0):
        if col_name in df_temp.columns:
            s = df_temp[col_name].astype(str).str.replace(',', '.').str.strip()
            return pd.to_numeric(s, errors='coerce').fillna(default_val).to_numpy(dtype=float)
        return np.full(len(df_temp), default_val, dtype=float)

    qty_col = 'order_units' if 'order_units' in df_temp.columns else ('order_quantity' if 'order_quantity' in df_temp.columns else df_temp.columns[0])

    qty_arr = get_num(qty_col, 1.0)
    acc_arr = get_num('accepted_units', 1.0)

    if 'defect_units' in df_temp.columns:
        defect_arr = get_num('defect_units', 0.0)
    else:
        defect_arr = np.maximum(0.0, qty_arr - acc_arr)

    if 'promised_date' in df_temp.columns and 'actual_delivery_date' in df_temp.columns:
        promised = pd.to_datetime(df_temp['promised_date'], dayfirst=True, errors='coerce')
        actual = pd.to_datetime(df_temp['actual_delivery_date'], dayfirst=True, errors='coerce')
        delay_days = (actual - promised).dt.days.fillna(0.0)
        delay_days = np.maximum(0.0, delay_days.values)
    else:
        delay_days = np.zeros(len(df_temp))

    df_temp['safe_order_id'] = 1
    df_temp['safe_delay_days'] = delay_days
    df_temp['safe_defect_units'] = defect_arr
    df_temp['safe_order_qty'] = qty_arr

    is_on_time = delay_days == 0
    is_perfect = defect_arr == 0
    df_temp['safe_is_otif'] = np.where(is_on_time & is_perfect, 1.0, 0.0)

    # Agg kullanmadan saf gruplama (Pandas 2.2+ tam uyumlu)
    gr = df_temp.groupby('supplier_name', as_index=False)
    
    summary = pd.DataFrame()
    summary['supplier_name'] = gr['supplier_name'].first()['supplier_name']
    summary['total_orders'] = gr['safe_order_id'].count()['safe_order_id']
    summary['otif_count'] = gr['safe_is_otif'].sum()['safe_is_otif']
    summary['avg_delay_days'] = gr['safe_delay_days'].mean()['safe_delay_days']
    summary['total_defects'] = gr['safe_defect_units'].sum()['safe_defect_units']
    summary['total_qty'] = gr['safe_order_qty'].sum()['safe_order_qty']

    summary['otif_rate'] = (summary['otif_count'] / np.maximum(1.0, summary['total_orders'])) * 100.0
    summary['defect_rate'] = np.where(summary['total_qty'] > 0, (summary['total_defects'] / summary['total_qty']) * 100.0, 0.0)

    return summary