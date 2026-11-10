import numpy as np
import pandas as pd

def run_monte_carlo_stockout(df, supplier_name, current_stock=100, daily_demand=10, n_simulations=1000):
    """
    Belirli bir tedarikçi için Monte Carlo simülasyonu çalıştırarak stoksuz kalma olasılığını hesaplar.
    """
    supplier_df = df[df['supplier_name'] == supplier_name].copy()
    
    if supplier_df.empty:
        return 0.0, np.array([])
        
    # Tedarikçinin geçmiş teslimat sürelerini (Lead Time) hesaplama
    supplier_df['promised_delivery_date'] = pd.to_datetime(supplier_df['promised_delivery_date'])
    supplier_df['actual_delivery_date'] = pd.to_datetime(supplier_df['actual_delivery_date'])
    supplier_df['order_date'] = pd.to_datetime(supplier_df['order_date'])
    
    # Gerçekleşen toplam teslimat süreleri (gün)
    actual_lead_times = (supplier_df['actual_delivery_date'] - supplier_df['order_date']).dt.days.values
    
    mean_lead = np.mean(actual_lead_times)
    std_lead = np.std(actual_lead_times) if np.std(actual_lead_times) > 0 else 1.0
    
    # Monte Carlo Simülasyonu: 1.000 senaryo için teslimat süresi üretme (Normal Dağılım)
    simulated_lead_times = np.random.normal(mean_lead, std_lead, n_simulations)
    simulated_lead_times = np.clip(simulated_lead_times, 1, None) # Minimum 1 gün
    
    # Her senaryoda harcanacak stok = Teslimat Süresi * Günlük Talep
    simulated_demand = simulated_lead_times * daily_demand
    
    # Stok kırılması gerçekleşen senaryolar (Talep > Mevcut Stok)
    stockouts = simulated_demand > current_stock
    stockout_probability = np.mean(stockouts)
    
    return stockout_probability, simulated_demand

if __name__ == "__main__":
    df = pd.read_csv("data/orders.csv")
    prob, demands = run_monte_carlo_stockout(df, "TeknoLojistik A.Ş.", current_stock=150, daily_demand=15)
    print(f"=== MONTE CARLO TESTİ (TeknoLojistik A.Ş.) ===")
    print(f"Stoksuz Kalma (Stockout) Riski: %{prob*100:.1f}")