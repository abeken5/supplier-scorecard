import pandas as pd
import numpy as np

def generate_real_world_supply_chain_data(output_path="data/orders.csv", n_samples=200):
    """
    DataCo Global Supply Chain verisi yapısında gerçekçi tedarik zinciri verisi üretir/hazırlar.
    """
    np.random.seed(42)
    
    suppliers = ['Global Logistics Corp', 'Apex Fasteners Inc', 'Bursa Otomotiv Yan Sanayi', 'TeknoLojistik A.Ş.', 'Ege Ambalaj Çözümleri']
    products = ['Rulman 6204', 'Civata M8x25', 'Plastik Enjeksiyon Gövde', 'Sensör Kartı V2', 'Karton Koli 40x40']
    
    start_date = pd.to_datetime('2026-01-01')
    order_dates = [start_date + pd.Timedelta(days=int(x)) for x in np.random.randint(0, 90, n_samples)]
    
    data = []
    for i in range(n_samples):
        order_id = 2000 + i
        supplier = np.random.choice(suppliers, p=[0.25, 0.20, 0.20, 0.15, 0.20])
        product = np.random.choice(products)
        
        order_qty = int(np.random.choice([50, 100, 250, 500, 1000]))
        unit_price = round(float(np.random.uniform(10.0, 150.0)), 2)
        
        # Tedarikçiye özel gecikme & kalite profili (Gerçekçilik katmak için)
        if supplier == 'Global Logistics Corp':
            planned_lead = np.random.randint(5, 10)
            actual_delay = np.random.choice([0, 1, 2, 5], p=[0.70, 0.15, 0.10, 0.05])
            defect_rate = np.random.uniform(0.00, 0.02)
        elif supplier == 'TeknoLojistik A.Ş.':
            planned_lead = np.random.randint(10, 20)
            actual_delay = np.random.choice([0, 3, 7, 12], p=[0.40, 0.30, 0.20, 0.10])
            defect_rate = np.random.uniform(0.02, 0.08)
        else:
            planned_lead = np.random.randint(3, 14)
            actual_delay = np.random.choice([-1, 0, 1, 3], p=[0.10, 0.65, 0.15, 0.10])
            defect_rate = np.random.uniform(0.005, 0.03)
            
        o_date = order_dates[i]
        promised_date = o_date + pd.Timedelta(days=planned_lead)
        actual_date = promised_date + pd.Timedelta(days=actual_delay)
        
        accepted_qty = int(order_qty * (1 - defect_rate))
        
        data.append({
            'order_id': order_id,
            'supplier_name': supplier,
            'product_name': product,
            'order_date': o_date.strftime('%Y-%m-%d'),
            'promised_delivery_date': promised_date.strftime('%Y-%m-%d'),
            'actual_delivery_date': actual_date.strftime('%Y-%m-%d'),
            'order_quantity': order_qty,
            'accepted_quantity': accepted_qty,
            'unit_price': unit_price
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Gerçekçi Tedarik Zinciri Verisi ({n_samples} Satır) '{output_path}' konumuna başarıyla kaydedildi!")

if __name__ == "__main__":
    generate_real_world_supply_chain_data()