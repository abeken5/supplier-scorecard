import pandas as pd
import numpy as np

def calculate_abc_xyz(df):
    """
    Ürün bazında ciro katkısına göre ABC ve talep oynaklığına göre XYZ analizini hesaplar.
    """
    df = df.copy()
    df['total_value'] = df['order_quantity'] * df['unit_price']
    
    # 1. ABC Analizi (Ciro Katkısı)
    product_abc = df.groupby('product_name').agg(
        total_spend=('total_value', 'sum'),
        total_orders=('order_id', 'count')
    ).reset_index()
    
    product_abc = product_abc.sort_values(by='total_spend', ascending=False)
    product_abc['cum_spend'] = product_abc['total_spend'].cumsum()
    product_abc['cum_perc'] = product_abc['cum_spend'] / product_abc['total_spend'].sum()
    
    def assign_abc(perc):
        if perc <= 0.70:
            return 'A'
        elif perc <= 0.90:
            return 'B'
        else:
            return 'C'
            
    product_abc['abc_class'] = product_abc['cum_perc'].apply(assign_abc)
    
    # 2. XYZ Analizi (Talep Oynaklığı / Değişim Katsayısı - CV)
    # Ürün bazında sipariş miktarlarının standart sapması / ortalaması
    product_xyz = df.groupby('product_name')['order_quantity'].agg(['mean', 'std']).reset_index()
    product_xyz['std'] = product_xyz['std'].fillna(0)
    product_xyz['cv'] = product_xyz['std'] / product_xyz['mean']
    
    def assign_xyz(cv):
        if cv <= 0.20:
            return 'X' # Düzenli talep
        elif cv <= 0.50:
            return 'Y' # Sezonluk/Orta dalgalı talep
        else:
            return 'Z' # Düzensiz/Yüksek dalgalı talep
            
    product_xyz['xyz_class'] = product_xyz['cv'].apply(assign_xyz)
    
    # Birleştirme
    analysis_df = pd.merge(product_abc[['product_name', 'total_spend', 'abc_class']], 
                           product_xyz[['product_name', 'cv', 'xyz_class']], 
                           on='product_name')
    
    analysis_df['abc_xyz'] = analysis_df['abc_class'] + analysis_df['xyz_class']
    return analysis_df

if __name__ == "__main__":
    df = pd.read_csv("data/orders.csv")
    result = calculate_abc_xyz(df)
    print("=== ÜRÜN ABC/XYZ ANALİZİ ===")
    print(result[['product_name', 'total_spend', 'abc_class', 'xyz_class', 'abc_xyz']])