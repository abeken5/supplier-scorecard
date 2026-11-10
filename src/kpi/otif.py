import pandas as pd
import numpy as np

def calculate_kpis(df):
    df_temp = df.copy()

    # Sütun İsimlerini Temizleme
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
    cols_list = list(df_temp.columns)

    if 'supplier_name' not in cols_list:
        df_temp['supplier_name'] = 'Varsayılan Tedarikçi'
        cols_list.append('supplier_name')

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

    if 'promised_date' in cols_list and 'actual_delivery_date' in cols_list:
        promised = pd.to_datetime(df_temp['promised_date'], dayfirst=True, errors='coerce')
        actual = pd.to_datetime(df_temp['actual_delivery_date'], dayfirst=True, errors='coerce')
        delay_days = (actual - promised).dt.days.fillna(0.0)
        delay_days = np.maximum(0.0, delay_days.values)
    else:
        delay_days = np.zeros(len(df_temp))

    is_on_time = delay_days == 0
    is_perfect = defect_arr == 0
    otif_arr = np.where(is_on_time & is_perfect, 1.0, 0.0)

    # Saf Python Sözlük Toplaması (Pandas groupby/agg tamamen bypass)
    unique_suppliers = df_temp['supplier_name'].unique()
    records = []

    for supp in unique_suppliers:
        mask = (df_temp['supplier_name'] == supp).to_numpy()
        tot_orders = int(np.sum(mask))
        otif_cnt = float(np.sum(otif_arr[mask]))
        avg_delay = float(np.mean(delay_days[mask])) if tot_orders > 0 else 0.0
        tot_defects = float(np.sum(defect_arr[mask]))
        tot_qty = float(np.sum(qty_arr[mask]))

        otif_rate = (otif_cnt / max(1, tot_orders)) * 100.0
        defect_rate = (tot_defects / tot_qty * 100.0) if tot_qty > 0 else 0.0

        records.append({
            'supplier_name': supp,
            'total_orders': tot_orders,
            'otif_count': otif_cnt,
            'avg_delay_days': avg_delay,
            'total_defects': tot_defects,
            'total_qty': tot_qty,
            'otif_rate': otif_rate,
            'defect_rate': defect_rate
        })

    return pd.DataFrame(records)