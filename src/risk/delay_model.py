import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def train_delay_model(csv_path):
    # Veriyi Oku
    df = pd.read_csv(csv_path)
    
    # Tarih İşlemleri
    df['promised_delivery_date'] = pd.to_datetime(df['promised_delivery_date'])
    df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'])
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Hedef Değişken (Gecikme Var mı? 1: Evet, 0: Hayır)
    df['is_delayed'] = (df['actual_delivery_date'] > df['promised_delivery_date']).astype(int)
    
    # Öznitelikler (Features)
    df['lead_time_days'] = (df['promised_delivery_date'] - df['order_date']).dt.days
    
    # Tedarikçi bazında geçmiş ortalama hata oranı ekleyelim
    df['defect_units'] = df['order_quantity'] - df['accepted_quantity']
    df['defect_rate'] = df['defect_units'] / df['order_quantity']

    X = df[['order_quantity', 'unit_price', 'lead_time_days', 'defect_rate']]
    y = df['is_delayed']

    # Veri setimiz küçük olduğu için modeli mevcut veride eğitiyoruz
    model = LogisticRegression()
    model.fit(X, y)

    # Tahminler
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print("=== GECİKME TAHMİN MODELİ PERFORMANSI ===")
    print("Tahmin Edilen Risk Olasılıkları:")
    for idx, row in df.iterrows():
        print(f"Sipariş {row['order_id']} ({row['supplier_name']}): Gecikme Riski %{y_prob[idx]*100:.1f}")

    return model

if __name__ == "__main__":
    train_delay_model("data/orders.csv")