import numpy as np
import pandas as pd

def calculate_order_allocation(kpi_df, total_order_qty):
    """
    Tedarikçilerin TOPSIS skorları ve risk durumlarına göre sipariş miktarını optimum şekilde paylaştırır.
    """
    df = kpi_df.copy()
    
    # 1. TOPSIS Skorlarını kontrol et ve NaN varsa 0.5 varsayılan değeri ver
    if 'TOPSIS Skor' not in df.columns or df['TOPSIS Skor'].isnull().all():
        df['TOPSIS Skor'] = 0.5
    else:
        df['TOPSIS Skor'] = df['TOPSIS Skor'].fillna(0.0)
    
    scores = df['TOPSIS Skor'].values
    
    # Tüm skorlar 0 ise eşit ağırlık dağıt
    if np.sum(scores) == 0:
        normalized_weights = np.ones(len(scores)) / len(scores)
    else:
        normalized_weights = scores / np.sum(scores)
    
    # 2. Risk Cezası (Gecikme Gününe Göre İnce Ayar)
    delays = df['avg_delay_days'].fillna(0).values
    penalty = 1 / (1 + delays)
    adjusted_weights = normalized_weights * penalty
    
    if np.sum(adjusted_weights) == 0:
        final_weights = np.ones(len(adjusted_weights)) / len(adjusted_weights)
    else:
        final_weights = adjusted_weights / np.sum(adjusted_weights)
    
    # 3. Sipariş Miktarını Bölüştürme
    df['Allocated_Ratio'] = final_weights
    df['Allocated_Qty'] = (final_weights * total_order_qty).round().astype(int)
    
    # Küsürat düzeltmesi (NaN korumalı idxmax)
    diff = total_order_qty - df['Allocated_Qty'].sum()
    if diff != 0:
        top_idx = df['TOPSIS Skor'].fillna(0).idxmax()
        df.loc[top_idx, 'Allocated_Qty'] += diff

    return df[['supplier_name', 'TOPSIS Skor', 'avg_delay_days', 'Allocated_Ratio', 'Allocated_Qty']]

if __name__ == "__main__":
    from src.kpi.otif import calculate_kpis
    from src.scoring.ahp_weights import calculate_ahp_weights
    from src.scoring.topsis import calculate_topsis
    
    orders_df = pd.read_csv("data/orders.csv")
    kpi_df = calculate_kpis(orders_df)
    
    weights = [0.4, 0.3, 0.2, 0.1]
    matrix = kpi_df[['otif_rate', 'avg_delay_days', 'defect_rate', 'total_orders']].values
    topsis_scores = calculate_topsis(matrix, weights, [True, False, False, True])
    kpi_df['TOPSIS Skor'] = topsis_scores
    
    result = calculate_order_allocation(kpi_df, 10000)
    print("=== SİPARİŞ TAHSİS TESTİ ===")
    print(result)