import numpy as np
import pandas as pd

def calculate_landed_cost(qty, unit_price, freight_cost, customs_duty_rate):
    """
    Mal Oluş Maliyeti (Landed Cost)
    Not: İndirilebilir KDV maliyet unsuruna girmez; gümrük vergisi maliyete dahil edilir.
    """
    return (qty * unit_price * (1.0 + customs_duty_rate)) + freight_cost

def calculate_expected_delay_cost(qty, unit_price, p_delay, avg_delay_days, daily_delay_rate=0.001):
    """
    Gecikmenin Beklenen Maliyeti
    """
    return p_delay * avg_delay_days * daily_delay_rate * (qty * unit_price)

def calculate_quality_cost(qty, unit_price, defect_rate, rework_factor=1.5):
    """
    Kalite ve Yeniden İşleme Maliyeti
    """
    return qty * unit_price * defect_rate * rework_factor

def calculate_safety_stock(z, daily_demand, demand_std, lead_time, lead_time_std):
    """
    Emniyet Stoğu Miktarı (Adet)
    """
    var_term = (lead_time * (demand_std ** 2)) + ((daily_demand ** 2) * (lead_time_std ** 2))
    return z * np.sqrt(np.maximum(0.0, var_term))

def calculate_working_capital_cost(order_value, lead_time, payment_terms_days, ss_units, unit_price, wacc=0.40, days=365):
    """
    İşletme Sermayesi Finansman Maliyeti
    """
    daily_spend = np.where(lead_time > 0, order_value / lead_time, order_value)
    ss_inventory_days = np.where(daily_spend > 0, (ss_units * unit_price) / daily_spend, 0.0)
    
    inventory_days = lead_time + ss_inventory_days
    net_financing_days = inventory_days - payment_terms_days
    
    net_days_clamped = np.maximum(0.0, net_financing_days)
    return order_value * wacc * (net_days_clamped / days)

def calculate_supplier_tco(df, wacc=0.40, z_score=1.65, daily_demand=15.0, demand_std=3.0, daily_delay_rate=0.001):
    """
    Tedarikçi Bazlı Toplam Sahip Olma Maliyeti (TCO) Hesaplama
    Fail-Safe Yapı: Tüm tarih ve finansal sütunlar yoksa otomatik tamamlanır.
    """
    df_temp = df.copy()

    # Sütun Boşluk ve BOM Temizliği
    df_temp.columns = df_temp.columns.astype(str).str.replace('\ufeff', '').str.strip()

    # Fail-Safe: Eksik Finansal Sütunları Varsayılan Değerlerle Tamamlama
    default_values = {
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

    # Sütun İsim Eşleştirmesi & Sayısal Zorlama
    qty_col = 'order_quantity' if 'order_quantity' in df_temp.columns else ('order_units' if 'order_units' in df_temp.columns else df_temp.columns[0])

    qty_arr = np.array(df_temp[qty_col].values, dtype=float)
    price_arr = np.array(df_temp['unit_price'].values, dtype=float) if 'unit_price' in df_temp.columns else np.zeros(len(df_temp))
    freight_arr = np.array(df_temp['freight_cost'].values, dtype=float)
    duty_arr = np.array(df_temp['customs_duty_rate'].values, dtype=float)
    pay_terms_arr = np.array(df_temp['payment_terms_days'].values, dtype=float)
    lt_std_arr = np.array(df_temp['lead_time_std'].values, dtype=float)
    rework_arr = np.array(df_temp['rework_cost_factor'].values, dtype=float)

    # Tarihler İçin Korumalı (Fail-Safe) Hesaplama
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

    # Gecikme Olasılığı
    p_delay_arr = np.where(delay_days > 0, 0.40, 0.10)

    # Hatalı Miktar Hesabı
    accepted_col = 'accepted_quantity' if 'accepted_quantity' in df_temp.columns else ('accepted_units' if 'accepted_units' in df_temp.columns else None)
    if accepted_col and accepted_col in df_temp.columns:
        defect_qty = np.maximum(0.0, qty_arr - np.array(df_temp[accepted_col].values, dtype=float))
    else:
        defect_qty = np.zeros(len(df_temp))
    
    defect_rate_arr = np.where(qty_arr > 0, defect_qty / qty_arr, 0.0)

    # TCO Hesaplamaları
    landed = calculate_landed_cost(qty_arr, price_arr, freight_arr, duty_arr)
    delay_cost = calculate_expected_delay_cost(qty_arr, price_arr, p_delay_arr, delay_days, daily_delay_rate)
    quality = calculate_quality_cost(qty_arr, price_arr, defect_rate_arr, rework_arr)

    order_values = qty_arr * price_arr
    ss_units_arr = calculate_safety_stock(z_score, daily_demand, demand_std, lead_time_days, lt_std_arr)
    wacc_cost = calculate_working_capital_cost(order_values, lead_time_days, pay_terms_arr, ss_units_arr, price_arr, wacc)

    df_temp['row_tco'] = landed + delay_cost + quality + wacc_cost
    df_temp['row_landed'] = landed
    df_temp['row_delay_cost'] = delay_cost
    df_temp['row_quality_cost'] = quality
    df_temp['row_wacc_cost'] = wacc_cost

    # Tedarikçi Bazında Özetleme
    summary = df_temp.groupby('supplier_name', as_index=False).agg(
        total_qty=(qty_col, 'sum'),
        total_price_spend=('unit_price', lambda x: np.sum(qty_arr[x.index] * price_arr[x.index])),
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