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
    price_arr = get_num('unit_price', 50.0)
    freight_arr = get_num('freight_cost', 150.0)
    duty_arr = get_num('customs_duty_rate', 0.05)
    pay_terms_arr = get_num('payment_terms_days', 30.0)
    lt_std_arr = get_num('lead_time_std', 2.0)
    rework_arr = get_num('rework_cost_factor', 1.5)

    if 'promised_date' in df_temp.columns and 'order_date' in df_temp.columns:
        promised = pd.to_datetime(df_temp['promised_date'], dayfirst=True, errors='coerce')
        order_d = pd.to_datetime(df_temp['order_date'], dayfirst=True, errors='coerce')
        lead_time_days = (promised - order_d).dt.days.fillna(7.0)
        lead_time_days = np.maximum(1.0, lead_time_days.values)
    else:
        lead_time_days = np.full(len(df_temp), 7.0)

    if 'actual_delivery_date' in df_temp.columns and 'promised_date' in df_temp.columns:
        actual = pd.to_datetime(df_temp['actual_delivery_date'], dayfirst=True, errors='coerce')
        promised = pd.to_datetime(df_temp['promised_date'], dayfirst=True, errors='coerce')
        delay_days = (actual - promised).dt.days.fillna(0.0)
        delay_days = np.maximum(0.0, delay_days.values)
    else:
        delay_days = np.zeros(len(df_temp))

    p_delay_arr = np.where(delay_days > 0, 0.40, 0.10)

    acc_col = 'accepted_units' if 'accepted_units' in df_temp.columns else ('accepted_quantity' if 'accepted_quantity' in df_temp.columns else None)
    if acc_col and acc_col in df_temp.columns:
        accepted_arr = get_num(acc_col, 0.0)
        defect_qty = np.maximum(0.0, qty_arr - accepted_arr)
    else:
        defect_qty = np.zeros(len(df_temp))
    
    defect_rate_arr = np.where(qty_arr > 0, defect_qty / qty_arr, 0.0)

    landed = calculate_landed_cost(qty_arr, price_arr, freight_arr, duty_arr)
    delay_cost = calculate_expected_delay_cost(qty_arr, price_arr, p_delay_arr, delay_days, daily_delay_rate)
    quality = calculate_quality_cost(qty_arr, price_arr, defect_rate_arr, rework_arr)

    order_values = qty_arr * price_arr
    ss_units_arr = calculate_safety_stock(z_score, daily_demand, demand_std, lead_time_days, lt_std_arr)
    wacc_cost = calculate_working_capital_cost(order_values, lead_time_days, pay_terms_arr, ss_units_arr, price_arr, wacc)

    df_temp['safe_qty'] = qty_arr
    df_temp['safe_spend'] = order_values
    df_temp['safe_tco'] = landed + delay_cost + quality + wacc_cost
    df_temp['safe_landed'] = landed
    df_temp['safe_delay_cost'] = delay_cost
    df_temp['safe_quality_cost'] = quality
    df_temp['safe_wacc_cost'] = wacc_cost
    df_temp['safe_pay_terms'] = pay_terms_arr
    df_temp['safe_fin_health'] = df_temp['financial_health'] if 'financial_health' in df_temp.columns else 'B'

    # Saf gruplama
    gr = df_temp.groupby('supplier_name', as_index=False)
    summary = pd.DataFrame()
    summary['supplier_name'] = gr['supplier_name'].first()['supplier_name']
    summary['total_qty'] = gr['safe_qty'].sum()['safe_qty']
    summary['total_price_spend'] = gr['safe_spend'].sum()['safe_spend']
    summary['total_tco'] = gr['safe_tco'].sum()['safe_tco']
    summary['landed_cost'] = gr['safe_landed'].sum()['safe_landed']
    summary['delay_cost'] = gr['safe_delay_cost'].sum()['safe_delay_cost']
    summary['quality_cost'] = gr['safe_quality_cost'].sum()['safe_quality_cost']
    summary['wacc_cost'] = gr['safe_wacc_cost'].sum()['safe_wacc_cost']
    summary['avg_payment_terms'] = gr['safe_pay_terms'].mean()['safe_pay_terms']
    summary['financial_health'] = gr['safe_fin_health'].first()['safe_fin_health']

    summary['unit_price_avg'] = summary['total_price_spend'] / np.maximum(1.0, summary['total_qty'])
    summary['unit_tco'] = summary['total_tco'] / np.maximum(1.0, summary['total_qty'])
    summary['tco_premium_perc'] = ((summary['unit_tco'] - summary['unit_price_avg']) / np.maximum(0.001, summary['unit_price_avg'])) * 100.0

    return summary