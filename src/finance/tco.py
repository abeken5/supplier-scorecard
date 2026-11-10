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
        df_temp['supplier_name'] = 'Varsayılan Tedarikçi'
        cols_list.append('supplier_name')

    def extract_num(col_name, default_val=0.0):
        if col_name in cols_list:
            s = df_temp[col_name].astype(str).str.replace(',', '.').str.strip()
            return pd.to_numeric(s, errors='coerce').fillna(default_val).to_numpy(dtype=float)
        return np.full(len(df_temp), default_val, dtype=float)

    qty_col = 'order_units' if 'order_units' in cols_list else ('order_quantity' if 'order_quantity' in cols_list else cols_list[0])

    qty_arr = extract_num(qty_col, 1.0)
    price_arr = extract_num('unit_price', 50.0)
    freight_arr = extract_num('freight_cost', 150.0)
    duty_arr = extract_num('customs_duty_rate', 0.05)
    pay_terms_arr = extract_num('payment_terms_days', 30.0)
    lt_std_arr = extract_num('lead_time_std', 2.0)
    rework_arr = extract_num('rework_cost_factor', 1.5)

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

    p_delay_arr = np.where(delay_days > 0, 0.40, 0.10)

    acc_col = 'accepted_units' if 'accepted_units' in cols_list else ('accepted_quantity' if 'accepted_quantity' in cols_list else None)
    if acc_col and acc_col in cols_list:
        defect_qty = np.maximum(0.0, qty_arr - extract_num(acc_col, 0.0))
    else:
        defect_qty = np.zeros(len(df_temp))
    
    defect_rate_arr = np.where(qty_arr > 0, defect_qty / qty_arr, 0.0)

    landed = calculate_landed_cost(qty_arr, price_arr, freight_arr, duty_arr)
    delay_cost = calculate_expected_delay_cost(qty_arr, price_arr, p_delay_arr, delay_days, daily_delay_rate)
    quality = calculate_quality_cost(qty_arr, price_arr, defect_rate_arr, rework_arr)

    order_values = qty_arr * price_arr
    ss_units_arr = calculate_safety_stock(z_score, daily_demand, demand_std, lead_time_days, lt_std_arr)
    wacc_cost = calculate_working_capital_cost(order_values, lead_time_days, pay_terms_arr, ss_units_arr, price_arr, wacc)

    tco_row_arr = landed + delay_cost + quality + wacc_cost

    # Saf Python Özetleme
    unique_suppliers = df_temp['supplier_name'].unique()
    records = []

    for supp in unique_suppliers:
        mask = (df_temp['supplier_name'] == supp).to_numpy()
        tot_qty = float(np.sum(qty_arr[mask]))
        tot_price_spend = float(np.sum(order_values[mask]))
        tot_tco = float(np.sum(tco_row_arr[mask]))
        tot_landed = float(np.sum(landed[mask]))
        tot_delay = float(np.sum(delay_cost[mask]))
        tot_quality = float(np.sum(quality[mask]))
        tot_wacc = float(np.sum(wacc_cost[mask]))
        avg_pay = float(np.mean(pay_terms_arr[mask])) if len(pay_terms_arr[mask]) > 0 else 30.0

        unit_price_avg = tot_price_spend / max(1.0, tot_qty)
        unit_tco = tot_tco / max(1.0, tot_qty)
        tco_premium_perc = ((unit_tco - unit_price_avg) / max(0.001, unit_price_avg)) * 100.0

        records.append({
            'supplier_name': supp,
            'total_qty': tot_qty,
            'total_price_spend': tot_price_spend,
            'total_tco': tot_tco,
            'landed_cost': tot_landed,
            'delay_cost': tot_delay,
            'quality_cost': tot_quality,
            'wacc_cost': tot_wacc,
            'avg_payment_terms': avg_pay,
            'financial_health': 'B',
            'unit_price_avg': unit_price_avg,
            'unit_tco': unit_tco,
            'tco_premium_perc': tco_premium_perc
        })

    return pd.DataFrame(records)