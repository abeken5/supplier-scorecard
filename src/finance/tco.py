import numpy as np
import pandas as pd

def calculate_landed_cost(qty, unit_price, freight_cost, customs_duty_rate):
    return (qty * unit_price * (1.0 + customs_duty_rate)) + freight_cost

def calculate_expected_delay_cost(qty, unit_price, p_delay, avg_delay_days, daily_delay_rate=0.001):
    return p_delay * avg_delay_days * daily_delay_rate * (qty * unit_price)

def calculate_quality_cost(qty, unit_price, defect_rate, rework_factor=1.5):
    return qty * unit_price * defect_rate * rework_factor

def calculate_safety_stock(z, daily_demand, demand_std, lead_time, lead_time_std):
    var_term = (lead_time * (demand_std ** 2)) + ((daily_demand ** 2) * (lead_time_std ** 2))
    return z * np.sqrt(np.maximum(0.0, var_term))

def calculate_working_capital_cost(order_value, lead_time, payment_terms_days, ss_units, unit_price, wacc=0.40, days=365):
    daily_spend = np.where(lead_time > 0, order_value / lead_time, order_value)
    ss_inventory_days = np.where(daily_spend > 0, (ss_units * unit_price) / daily_spend, 0.0)
    inventory_days = lead_time + ss_inventory_days
    net_financing_days = inventory_days - payment_terms_days
    net_days_clamped = np.maximum(0.0, net_financing_days)
    return order_value * wacc * (net_days_clamped / days)

def calculate_supplier_tco(df, wacc=0.40, z_score=1.65, daily_demand=15.0, demand_std=3.0, daily_delay_rate=0.001):
    df_temp = df.copy()

    # 1. Sütun Temizliği ve İsim Esnekliği
    df_temp.columns = df_temp.columns.astype(str).str.replace('\ufeff', '').str.strip()
    
    column_mapping = {
        'promised_delivery_date': 'promised_date',
        'delivery_date': 'actual_delivery_date',
        'defective_quantity': 'defect_units',
        'accepted_quantity': 'accepted_units',
        'order_quantity': 'order_units',
        'supplier': 'supplier_name',
        'tedarikci': 'supplier_name'
    }
    df_temp = df_temp.rename(columns=column_mapping)

    if 'supplier_name' not in df_temp.columns:
        df_temp['supplier_name'] = 'Bilinmeyen Tedarikçi'

    # 2. Varsayılan Finansal Parametre Koruması
    default_values = {
        'unit_price': 50.0,
        'freight_cost': 150.0,
        'customs_duty_rate': 0.05,
        'payment_terms_days': 30.0,
        'lead_time_std': 2.0,
        'rework_cost_factor': 1.5,
        'financial_health': 'B'
    }

    for col, val in default_values.items():
        if col not in df_temp.columns:
            df_temp[col] = val
        else:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(val)

    # 3. Sayısal Dizilerin Standart Standı
    qty_col = 'order_units' if 'order_units' in df_temp.columns else ('order_quantity' if 'order_quantity' in df_temp.columns else df_temp.columns[0])

    qty_arr = pd.to_numeric(df_temp[qty_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    price_arr = pd.to_numeric(df_temp['unit_price'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    freight_arr = pd.to_numeric(df_temp['freight_cost'], errors='coerce').fillna(150.0).to_numpy(dtype=float)
    duty_arr = pd.to_numeric(df_temp['customs_duty_rate'], errors='coerce').fillna(0.05).to_numpy(dtype=float)
    pay_terms_arr = pd.to_numeric(df_temp['payment_terms_days'], errors='coerce').fillna(30.0).to_numpy(dtype=float)
    lt_std_arr = pd.to_numeric(df_temp['lead_time_std'], errors='coerce').fillna(2.0).to_numpy(dtype=float)
    rework_arr = pd.to_numeric(df_temp['rework_cost_factor'], errors='coerce').fillna(1.5).to_numpy(dtype=float)

    # 4. Tarih ve Teslimat Süreleri Koruması
    p_col = 'promised_date' if 'promised_date' in df_temp.columns else None
    o_col = 'order_date' if 'order_date' in df_temp.columns else None
    a_col = 'actual_delivery_date' if 'actual_delivery_date' in df_temp.columns else None

    if p_col and o_col:
        promised = pd.to_datetime(df_temp[p_col], dayfirst=True, errors='coerce')
        order_d = pd.to_datetime(df_temp[o_col], dayfirst=True, errors='coerce')
        lead_time_days = (promised - order_d).dt.days.fillna(7.0)
        lead_time_days = np.maximum(1.0, lead_time_days.values)
    else:
        lead_time_days = np.full(len(df_temp), 7.0)

    if a_col and p_col:
        actual = pd.to_datetime(df_temp[a_col], dayfirst=True, errors='coerce')
        promised = pd.to_datetime(df_temp[p_col], dayfirst=True, errors='coerce')
        delay_days = (actual - promised).dt.days.fillna(0.0)
        delay_days = np.maximum(0.0, delay_days.values)
    else:
        delay_days = np.zeros(len(df_temp))

    p_delay_arr = np.where(delay_days > 0, 0.40, 0.10)

    # 5. Kusurlu Miktar Koruması
    acc_col = 'accepted_units' if 'accepted_units' in df_temp.columns else ('accepted_quantity' if 'accepted_quantity' in df_temp.columns else None)
    if acc_col and acc_col in df_temp.columns:
        defect_qty = np.maximum(0.0, qty_arr - pd.to_numeric(df_temp[acc_col], errors='coerce').fillna(0.0).to_numpy(dtype=float))
    else:
        defect_qty = np.zeros(len(df_temp))
    
    defect_rate_arr = np.where(qty_arr > 0, defect_qty / qty_arr, 0.0)

    # 6. TCO Bileşenlerinin Doğrudan Satır Hesabı
    landed = calculate_landed_cost(qty_arr, price_arr, freight_arr, duty_arr)
    delay_cost = calculate_expected_delay_cost(qty_arr, price_arr, p_delay_arr, delay_days, daily_delay_rate)
    quality = calculate_quality_cost(qty_arr, price_arr, defect_rate_arr, rework_arr)

    order_values = qty_arr * price_arr
    ss_units_arr = calculate_safety_stock(z_score, daily_demand, demand_std, lead_time_days, lt_std_arr)
    wacc_cost = calculate_working_capital_cost(order_values, lead_time_days, pay_terms_arr, ss_units_arr, price_arr, wacc)

    df_temp['row_price_spend'] = order_values
    df_temp['row_tco'] = landed + delay_cost + quality + wacc_cost
    df_temp['row_landed'] = landed
    df_temp['row_delay_cost'] = delay_cost
    df_temp['row_quality_cost'] = quality
    df_temp['row_wacc_cost'] = wacc_cost

    # 7. Tedarikçi Gruplaması
    summary = df_temp.groupby('supplier_name', as_index=False).agg(
        total_qty=(qty_col, 'sum'),
        total_price_spend=('row_price_spend', 'sum'),
        total_tco=('row_tco', 'sum'),
        landed_cost=('row_landed', 'sum'),
        delay_cost=('row_delay_cost', 'sum'),
        quality_cost=('row_quality_cost', 'sum'),
        wacc_cost=('row_wacc_cost', 'sum'),
        avg_payment_terms=('payment_terms_days', 'mean'),
        financial_health=('financial_health', 'first')
    )

    summary['unit_price_avg'] = summary['total_price_spend'] / np.maximum(1.0, summary['total_qty'])
    summary['unit_tco'] = summary['total_tco'] / np.maximum(1.0, summary['total_qty'])
    summary['tco_premium_perc'] = ((summary['unit_tco'] - summary['unit_price_avg']) / np.maximum(0.001, summary['unit_price_avg'])) * 100.0

    return summary