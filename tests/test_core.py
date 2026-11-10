import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scoring.ahp_weights import calculate_ahp_weights
from src.scoring.topsis import calculate_topsis

class TestCoreFunctions(unittest.TestCase):

    def test_ahp_consistency(self):
        """Test 1: Tamamen tutarlı bir AHP matrisinde CR < 0.05 olmalıdır."""
        perfect_matrix = np.array([
            [1.0, 2.0, 4.0],
            [0.5, 1.0, 2.0],
            [0.25, 0.5, 1.0]
        ])
        weights, cr = calculate_ahp_weights(perfect_matrix)
        self.assertLess(cr, 0.05, f"Beklenen CR < 0.05 ancak {cr} bulundu.")

    def test_ahp_weights_sum_to_one(self):
        """Test 2: AHP ağırlıklarının toplamı tam olarak 1.0 olmalıdır."""
        matrix = np.array([
            [1.0, 3.0, 5.0],
            [1/3, 1.0, 2.0],
            [1/5, 1/2, 1.0]
        ])
        weights, _ = calculate_ahp_weights(matrix)
        self.assertTrue(np.isclose(np.sum(weights), 1.0), "AHP ağırlıklar toplamı 1.0 değil.")

    def test_topsis_superior_supplier(self):
        """Test 3: Açıkça üstün tedarikçi TOPSIS'te 1. sırada çıkmalıdır."""
        matrix = np.array([
            [99.0, 0.1, 0.1, 10.0],
            [50.0, 5.0, 4.0, 50.0],
            [30.0, 10.0, 8.0, 100.0]
        ])
        weights = np.array([0.4, 0.2, 0.2, 0.2])
        is_benefit = [True, False, False, False]
        
        scores = calculate_topsis(matrix, weights, is_benefit)
        self.assertEqual(np.argmax(scores), 0, "En üstün tedarikçi TOPSIS'te 1. sırada çıkmadı.")

if __name__ == '__main__':
    unittest.main()