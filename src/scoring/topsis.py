import numpy as np

def calculate_topsis(matrix, weights, is_benefit):
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) skorlarını hesaplar.
    """
    matrix = np.array(matrix, dtype=float)
    weights = np.array(weights, dtype=float)
    
    # 1. Matris Normalizasyonu
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    
    # 2. Ağırlıklı Normalize Matris
    weighted_matrix = norm_matrix * weights
    
    # 3. İdeal Pozitif (PIS) ve İdeal Negatif (NIS) Çözümler
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
    
    # 4. İdeal Çözümlere Mesafeler
    d_plus = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    d_minus = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
    
    # 5. Göreli Yakınlık / TOPSIS Skoru
    scores = d_minus / (d_plus + d_minus)
    return scores