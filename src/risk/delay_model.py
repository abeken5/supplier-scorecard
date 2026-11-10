import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def prepare_risk_data(df):
    df_temp = df.copy()
    
    # Sütun İsim Temizliği ve Alias Haritalama
    df_temp.columns = [str(c).replace('\ufeff', '').strip() for c in df_temp.columns]
    
    col_map = {
        'promised_delivery_date': 'promised_date',
        'delivery_date': 'actual_delivery_date',
        'order_quantity': 'order_units'
    }
    df_temp = df_temp.rename(columns=col_map)
    cols_list = list(df_temp.columns)

    # Esnek Sütun Çıkarma
    qty_col = 'order_units' if 'order_units' in cols_list else cols_list[0]
    qty_arr = pd.to_numeric(df_temp[qty_col].astype(str).str.replace(',', '.'), errors='coerce').fillna(100.0).to_numpy(dtype=float)

    # Tarih ve Teslimat Süresi
    if 'promised_date' in cols_list and 'order_date' in cols_list:
        promised = pd.to_datetime(df_temp['promised_date'], dayfirst=True, errors='coerce')
        order_d = pd.to_datetime(df_temp['order_date'], dayfirst=True, errors='coerce')
        lead_time_days = (promised - order_d).dt.days.fillna(7.0)
        lead_time_days = np.maximum(1.0, lead_time_days.values)
    else:
        lead_time_days = np.full(len(df_temp), 7.0)

    if 'actual_delivery_date' in cols_list and 'promised_date' in cols_list:
        actual = pd.to_datetime(df_temp['actual_delivery_date'], dayfirst=True, errors='coerce')
        promised = pd.to_datetime(df_temp['promised_date'], dayfirst=True, errors='coerce')
        delay_days = (actual - promised).dt.days.fillna(0.0)
        delay_days = np.maximum(0.0, delay_days.values)
    else:
        delay_days = np.zeros(len(df_temp))

    # Hedef Değişken (Gecikme Var mı? 1 / 0)
    # Eğer hiç gecikme yoksa sentetik olarak küçük bir gecikme ekleyerek modelin çökmesini engelleriz
    y = np.where(delay_days > 0, 1, 0)
    if len(np.unique(y)) < 2:
        y[0] = 1 # Model eğitimi için en az 2 sınıf gerekli

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