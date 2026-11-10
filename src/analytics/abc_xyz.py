import pandas as pd
import numpy as np

def calculate_abc_xyz(df):
    df_temp = df.copy()

    cols_clean = [str(c).replace('\ufeff', '').strip() for c in df_temp.columns]
    df_temp.columns = cols_clean

    qty_col = 'order_quantity' if 'order_quantity' in cols_clean else ('order_units' if 'order_units' in cols_clean else cols_clean[0])
    prod_col = 'product_name' if 'product_name' in cols_clean else cols_clean[0]

    qty_arr = pd.to_numeric(df_temp[qty_col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0).to_numpy(dtype=float)
    price_arr = pd.to_numeric(df_temp['unit_price'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0).to_numpy(dtype=float) if 'unit_price' in cols_clean else np.zeros(len(df_temp), dtype=float)

    spend_arr = qty_arr * price_arr

    unique_products = df_temp[prod_col].unique()
    records = []

    for prod in unique_products:
        mask = (df_temp[prod_col] == prod).to_numpy()
        tot_spend = float(np.sum(spend_arr[mask]))
        mean_dem = float(np.mean(qty_arr[mask])) if len(qty_arr[mask]) > 0 else 0.0
        std_dem = float(np.std(qty_arr[mask])) if len(qty_arr[mask]) > 1 else 0.0

        records.append({
            'product_name': prod,
            'total_spend': tot_spend,
            'mean_demand': mean_dem,
            'std_demand': std_dem
        })

    res_df = pd.DataFrame(records).sort_values(by='total_spend', ascending=False).reset_index(drop=True)

    spend_vals = res_df['total_spend'].to_numpy(dtype=float)
    cum_spend = np.cumsum(spend_vals)
    tot_sum = float(np.sum(spend_vals))

    cum_perc = (cum_spend / tot_sum) if tot_sum > 0 else np.zeros_like(cum_spend)

    abc_classes = []
    for p in cum_perc:
        if p <= 0.70:
            abc_classes.append('A')
        elif p <= 0.90:
            abc_classes.append('B')
        else:
            abc_classes.append('C')

    xyz_classes = []
    mean_vals = res_df['mean_demand'].to_numpy(dtype=float)
    std_vals = res_df['std_demand'].to_numpy(dtype=float)

    for m, s in zip(mean_vals, std_vals):
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

    res_df['abc_class'] = abc_classes
    res_df['xyz_class'] = xyz_classes
    res_df['abc_xyz'] = [a + x for a, x in zip(abc_classes, xyz_classes)]

    return res_df