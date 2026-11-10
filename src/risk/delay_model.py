import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def prepare_risk_data(df):
    """
    Sipariş verilerinden gecikme riskini tahmin etmek için veri seti hazırlar.
    """
    df = df.copy()
    df['promised_delivery_date'] = pd.to_datetime(df['promised_delivery_date'])
    df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'])
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Hedef Değişken: 1 gün veya daha fazla gecikme var mı? (1: Gecikme Var, 0: Zamanında)
    df['is_delayed'] = (df['actual_delivery_date'] > df['promised_delivery_date']).astype(int)
    
    # Öznitelikler (Features)
    df['planned_lead_time'] = (df['promised_delivery_date'] - df['order_date']).dt.days
    
    X = df[['order_quantity', 'planned_lead_time']]
    y = df['is_delayed']
    
    return X, y

def train_delay_model(df):
    """
    Lojistik Regresyon modelini eğitir.
    """
    X, y = prepare_risk_data(df)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    return model, scaler

def predict_delay_risk(model, scaler, order_quantity, planned_lead_time):
    """
    Yeni sipariş parametrelerine göre gecikme olasılığını tahmin eder.
    """
    input_data = np.array([[order_quantity, planned_lead_time]])
    input_scaled = scaler.transform(input_data)
    
    # 1 sınıfı (gecikme) olasılığı
    prob = model.predict_proba(input_scaled)[0][1]
    return prob

if __name__ == "__main__":
    df = pd.read_csv("data/orders.csv")
    model, scaler = train_delay_model(df)
    sample_prob = predict_delay_risk(model, scaler, 250, 7)
    print(f"Örnek Sipariş Gecikme Riski: %{sample_prob*100:.1f}")