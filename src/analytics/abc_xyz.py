import pandas as pd
import numpy as np

def calculate_abc_xyz(df):
    """
    Ürün bazlı ciro katkısı (ABC) ve talep değişkenliği (XYZ) analizi yapar.
    Streamlit/PyArrow tip çakışmalarını tamamen engellemek için saf NumPy dizileri kullanır.
    """
    df = df.copy()

    # 1. Sütun Temizliği
    df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]

    # 2. Sayısal Sütun Tespiti
    qty_col = 'order_quantity' if 'order_quantity' in df.columns else 'order_units'
    if qty_col not in df.columns:
        qty_col = df.columns[0]

    # 3. Sayısal Temizlik & Metin/Virgül Dönüşümü
    def to_clean_float(series):
        s_str = series.astype(str).str.replace(',', '.').str.strip()
        return pd.to_numeric(s_str, errors='coerce').fillna(0.0).astype('float64')

    df[qty_col] = to_clean_float(df[qty_col])
    if 'unit_price' in df.columns:
        df['unit_price'] = to_clean_float(df['unit_price'])
    else:
        df['unit_price'] = 0.0

    # 4. Saf NumPy Çarpanı (PyArrow Karışamaz)
    df['total_spend'] = df[qty_col].to_numpy(dtype=np.float64) * df['unit_price'].to_numpy(dtype=np.float64)

    # 5. Gruplama
    product_summary = df.groupby('product_name', as_index=False).agg(
        total_spend=('total_spend', 'sum'),
        mean_demand=(qty_col, 'mean'),
        std_demand=(qty_col, 'std')
    )

    # 6. NumPy Saf Float Dizilerine Çıkarma
    spends = product_summary['total_spend'].to_numpy(dtype=np.float64)
    means = product_summary['mean_demand'].to_numpy(dtype=np.float64)
    stds = np.nan_to_num(product_summary['std_demand'].to_numpy(dtype=np.float64), nan=0.0)

    # 7. Ciroya Göre Sıralama
    sort_idx = np.argsort(-spends)
    product_names = product_summary['product_name'].to_numpy()[sort_idx]
    spends = spends[sort_idx]
    means = means[sort_idx]
    stds = stds[sort_idx]

    # 8. ABC Kumülatif Hesaplama
    total_spend_sum = np.sum(spends)
    cum_spend = np.cumsum(spends)
    cum_perc = cum_spend / total_spend_sum if total_spend_sum > 0 else np.zeros_like(spends)

    abc_classes = []
    for p in cum_perc:
        if p <= 0.70:
            abc_classes.append('A')
        elif p <= 0.90:
            abc_classes.append('B')
        else:
            abc_classes.append('C')

    # 9. XYZ Sınıflandırması
    xyz_classes = []
    for m, s in zip(means, stds):
        if m == 0:
            xyz_classes.append('Z')
        else:
            cv = s / m
            if cv <= 0.20:
                xyz_classes.append('X')
            elif cv <= 0.50:
                xyz_classes.append('Y')
            else:
                xyz_classes.append('Z')

    abc_xyz = [a + x for a, x in zip(abc_classes, xyz_classes)]

    # 10. Sonuç DataFrame Oluşturma
    result_df = pd.DataFrame({
        'product_name': product_names,
        'total_spend': spends,
        'mean_demand': means,
        'std_demand': stds,
        'cum_spend': cum_spend,
        'cum_perc': cum_perc,
        'abc_class': abc_classes,
        'xyz_class': xyz_classes,
        'abc_xyz': abc_xyz
    })

    return result_df