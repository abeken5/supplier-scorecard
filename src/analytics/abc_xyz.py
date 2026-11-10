import pandas as pd
import numpy as np

def calculate_abc_xyz(df):
    """
    Ürün bazlı ciro katkısı (ABC) ve talep değişkenliği (XYZ) analizi yapar.
    PyArrow / Streamlit tip çakışmalarına karşı tam korumalıdır.
    """
    df = df.copy()

    # 1. Sütun İsimlerini Temizleme
    df.columns = df.columns.astype(str).str.replace('\ufeff', '').str.strip()

    # 2. Sayısal Sütun Tespiti
    qty_col = 'order_quantity' if 'order_quantity' in df.columns else 'order_units'
    if qty_col not in df.columns:
        qty_col = df.columns[0]

    # 3. Sütunları PyArrow Bağlamından Çıkarıp Saf Standart Pandas/NumPy Float Yapma
    def clean_to_float_array(series):
        # Eğer metin (string/object) ise virgülleri noktaya çevir
        s_str = series.astype(str).str.replace(',', '.').str.strip()
        # Sayıya çevir ve PyArrow nesnesinden sıyrılmak için .to_numpy() kullan
        return pd.to_numeric(s_str, errors='coerce').fillna(0.0).to_numpy(dtype=float)

    qty_values = clean_to_float_array(df[qty_col])
    price_values = clean_to_float_array(df['unit_price']) if 'unit_price' in df.columns else np.zeros(len(df))

    # 4. Ciro Hesaplama (Saf NumPy Çarpımı - PyArrow Engeline Takılmaz)
    df['total_spend'] = qty_values * price_values
    df[qty_col] = qty_values
    if 'unit_price' in df.columns:
        df['unit_price'] = price_values

    # 5. Ürün Bazlı Toplam Harcama ve Talep İstatistikleri
    product_summary = df.groupby('product_name').agg(
        total_spend=('total_spend', 'sum'),
        mean_demand=(qty_col, 'mean'),
        std_demand=(qty_col, 'std')
    ).reset_index()

    product_summary['std_demand'] = product_summary['std_demand'].fillna(0.0)

    # 6. ABC Analizi (Ciroya Göre)
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

    # 7. XYZ Analizi (Talep Değişkenliği CV = std / mean)
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