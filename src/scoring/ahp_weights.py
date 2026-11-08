import numpy as np

def calculate_ahp_weights(pairwise_matrix):
    """
    Saaty'nin AHP yöntemi ile kriter ağırlıklarını ve Tutarılılık Oranını (CR) hesaplar.
    """
    matrix = np.array(pairwise_matrix)
    n = matrix.shape[0]
    
    # Özdeğer ve Özvektör Hesaplama
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_index = np.argmax(eigenvalues.real)
    max_eigenvalue = eigenvalues.real[max_index]
    weights = eigenvectors[:, max_index].real
    weights = weights / weights.sum()

    # Tutarlılık İndeksi (CI) ve Tutarlılık Oranı (CR)
    ci = (max_eigenvalue - n) / (n - 1)
    random_index = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12} # N=4 için RI = 0.90
    ri = random_index.get(n, 1.12)
    cr = ci / ri if ri != 0 else 0

    return weights, cr

if __name__ == "__main__":
    # Kriterler: [Teslimat Güvenilirliği (OTIF), Gecikme Günü, Hata Oranı, Birim Fiyat]
    # Ikili karşılaştırma matrisi (Örnek)
    pairwise_matrix = [
        [1.0,  3.0,  4.0,  2.0],
        [1/3,  1.0,  2.0,  1/2],
        [1/4,  1/2,  1.0,  1/3],
        [1/2,  2.0,  3.0,  1.0]
    ]

    weights, cr = calculate_ahp_weights(pairwise_matrix)
    print("=== AHP KRİTER AĞIRLIKLARI ===")
    criteria = ['OTIF', 'Avg Delay', 'Defect Rate', 'Unit Price']
    for c, w in zip(criteria, weights):
        print(f"{c}: {w:.4f}")
    print(f"\nTutarlılık Oranı (CR): {cr:.4f} {'(Tutarli - OK)' if cr < 0.10 else '(Tutarsiz)'}")