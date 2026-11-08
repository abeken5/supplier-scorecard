import numpy as np
import pandas as pd

def run_topsis(decision_matrix, weights, is_benefit):
    """
    TOPSIS yöntemi ile tedarikçileri skorlar ve sıralar.
    is_benefit: [True, False, False, False] (OTIF yüksek olmalı, diğerleri düşük)
    """
    matrix = np.array(decision_matrix, dtype=float)
    
    # 1. Vektör Normalizasyonu
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    
    # 2. Ağırlıklı Normalize Matris
    weighted_matrix = norm_matrix * weights
    
    # 3. İdeal (PIS) ve Negatif İdeal (NIS) Çözümler
    ideal_best = []
    ideal_worst = []
    
    for j in range(matrix.shape[1]):
        if is_benefit[j]:
            ideal_best.append(np.max(weighted_matrix[:, j]))
            ideal_worst.append(np.min(weighted_matrix[:, j]))
        else:
            ideal_best.append(np.min(weighted_matrix[:, j]))
            ideal_worst.append(np.max(weighted_matrix[:, j]))
            
    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)
    
    # 4. İdeal Noktalara Mesafeler
    s_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    s_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
    
    # 5. Göreli Yakınlık Skoru (TOPSIS Score)
    scores = s_worst / (s_best + s_worst)
    return scores

if __name__ == "__main__":
    from ahp_weights import calculate_ahp_weights
    
    # Örnek Karar Matrisi: [OTIF, Avg Delay, Defect Rate, Unit Price]
    # Tedarikçi A, B, C
    decision_matrix = [
        [66.67, 1.0, 0.0066, 45.25],
        [0.00,  3.33, 0.0692, 41.83],
        [100.00, 0.0, 0.0100, 49.75]
    ]
    
    pairwise_matrix = [
        [1.0, 3.0, 4.0, 2.0],
        [1/3, 1.0, 2.0, 1/2],
        [1/4, 1/2, 1.0, 1/3],
        [1/2, 2.0, 3.0, 1.0]
    ]
    
    weights, _ = calculate_ahp_weights(pairwise_matrix)
    is_benefit = [True, False, False, False] # OTIF fayda (yüksek iyi), diğerleri maliyet (düşük iyi)
    
    scores = run_topsis(decision_matrix, weights, is_benefit)
    
    results = pd.DataFrame({
        'Tedarikçi': ['Tedarikci A', 'Tedarikci B', 'Tedarikci C'],
        'TOPSIS Skor': scores
    }).sort_values(by='TOPSIS Skor', ascending=False)
    
    print("=== TOPSIS TEDARİKÇİ SIRALAMASI ===")
    print(results.to_string(index=False))