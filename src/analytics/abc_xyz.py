import pandas as pd
import numpy as np

def calculate_abc_xyz(df):
    """
    Ürün bazlı ciro katkısı (ABC) ve talep değişkenliği (XYZ) analizi yapar.
    PyArrow / Streamlit tip çakışmalarına karşı tam korumalıdır.
    """
    df = df.copy()

    # Sütun İsimlerini Temizleme
    df.columns = df.columns.astype(str).str.replace('\ufeff', '').str.strip()

    # Sayısal Sütun Tespiti
    qty_col = 'order_quantity' if 'order_quantity' in df.columns else 'order_units'
    if qty_col not in df.columns:
        qty_col = df.columns[0]

    # Sayısal Dönüştürme (Virgül -> Nokta Dönüşümü & Float Zorlaması)
    for col in [qty_col, 'unit_price']:
        if col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype).startswith('string'):
                df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
        else:
            df[col] = 0.0

    # Ciro Hesaplama (Artık float * float olduğu için PyArrow çökmez)
    df['total_spend'] = df[qty_col] * df['unit_price']

    # Ürün Bazlı Toplam Harcama ve Talep İstatistikleri
    product_summary = df.groupby('product_name').agg(
        total_spend=('total_spend', 'sum'),
        mean_demand=(qty_col, 'mean'),
        std_demand=(qty_col, 'std')
    ).reset_index()

    product_summary['std_demand'] = product_summary['std_demand'].fillna(0.0)

    # 1. ABC Analizi (Ciroya Göre)
    product_abc = product_summary.sort_values(by='total_spend', ascending=False).reset_index(drop=True)
    product_abc['cum_spend'] = product_abc['total_spend'].cumsum()
    total_spend_sum = product_abc['total_spend'].sum()
    
    if total_spend_sum > 0:
        product_abc['cum_perc'] = product_abc['cum_spend'] / total_spend_sum
    else:
        product_abc['cum_perc'] = 0.0

    def assign_abc(perc):
        if perc <= 0.70:
            return 'A'
        elif perc <= 0.90:
            return 'B'
        else:
            return 'C'

    product_abc['abc_class'] = product_abc['cum_perc'].apply(assign_abc)

    # 2. XYZ Analizi (Talep Değişkenliği CV = std / mean)
    def assign_xyz(row):
        if row['mean_demand'] == 0:
            return 'Z'
        cv = row['std_demand'] / row['mean_demand']
        if cv <= 0.20:
            return 'X'
        elif cv <= 0.50:
            return 'Y'
        else:
            return 'Z'

    product_abc['xyz_class'] = product_abc.apply(assign_xyz, axis=1)
    product_abc['abc_xyz'] = product_abc['abc_class'] + product_abc['xyz_class']

    return product_abc