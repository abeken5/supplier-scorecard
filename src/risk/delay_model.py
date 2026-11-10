import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def safe_parse_date(date_val):
    """Pandas C-extension hatalarını baypas eden saf Python tarih ayrıştırıcı"""
    if pd.isna(date_val) or not date_val:
        return None
    s = str(date_val).strip()
    for fmt in ('%d.%m.%Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

def prepare_risk_data(df):
    df_temp = df.copy()
    cols_clean = [str(c).replace('\ufeff', '').strip() for c in df_temp.columns]
    df_temp.columns = cols_clean

    col_map = {
        'promised_delivery_date': 'promised_date',
        'delivery_date': 'actual_delivery_date',
        'order_quantity': 'order_units'
    }
    df_temp = df_temp.rename(columns=col_map)
    cols_list = list(df_temp.columns)

    # 1. Miktar Dizi Koruması
    qty_col = 'order_units' if 'order_units' in cols_list else ('order_quantity' if 'order_quantity' in cols_list else cols_list[0])
    qty_arr = pd.to_numeric(df_temp[qty_col].astype(str).str.replace(',', '.'), errors='coerce').fillna(100.0).to_numpy(dtype=float)

    # 2. Güvenli Teslimat Süresi (Lead Time) Hesaplama
    lead_time_list = []
    delay_list = []

    has_promised = 'promised_date' in cols_list
    has_order = 'order_date' in cols_list
    has_actual = 'actual_delivery_date' in cols_list

    for _, row in df_temp.iterrows():
        p_date = safe_parse_date(row.get('promised_date')) if has_promised else None
        o_date = safe_parse_date(row.get('order_date')) if has_order else None
        a_date = safe_parse_date(row.get('actual_delivery_date')) if has_actual else None

        if p_date and o_date:
            lt_days = (p_date - o_date).days
            lead_time_list.append(max(1.0, float(lt_days)))
        else:
            lead_time_list.append(7.0)

        if a_date and p_date:
            d_days = (a_date - p_date).days
            delay_list.append(max(0.0, float(d_days)))
        else:
            delay_list.append(0.0)

    lead_time_days = np.array(lead_time_list, dtype=float)
    delay_days = np.array(delay_list, dtype=float)

    # 3. Risk Etiketi (1: Gecikme Var, 0: Zamanında)
    y = np.where(delay_days > 0, 1, 0)
    
    # Lojistik Regresyon için tek sınıf koruması
    if len(np.unique(y)) < 2:
        y[0] = 1 if y[0] == 0 else 0

    X = np.column_stack([qty_arr, lead_time_days])
    return X, y

def train_delay_model(df):
    X, y = prepare_risk_data(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    return model, scaler

def predict_delay_risk(model, scaler, order_qty, planned_lead_time):
    X_new = np.array([[float(order_qty), float(planned_lead_time)]])
    X_new_scaled = scaler.transform(X_new)
    prob = model.predict_proba(X_new_scaled)[0][1]
    return prob