import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scoring.ahp_weights import calculate_ahp_weights
from src.scoring.topsis import calculate_topsis

def test_ahp_consistency():
    perfect_matrix = np.array([
        [1.0, 2.0, 4.0],
        [0.5, 1.0, 2.0],
        [0.25, 0.5, 1.0]
    ])
    weights, cr = calculate_ahp_weights(perfect_matrix)
    assert cr < 0.05, f"Beklenen CR < 0.05 ancak {cr} bulundu."

def test_ahp_weights_sum_to_one():
    matrix = np.array([
        [1.0, 3.0, 5.0],
        [1/3, 1.0, 2.0],
        [1/5, 1/2, 1.0]
    ])
    weights, _ = calculate_ahp_weights(matrix)
    assert np.isclose(np.sum(weights), 1.0), "AHP ağırlıklar toplamı 1.0 değil."

def test_topsis_superior_supplier():
    matrix = np.array([
        [99.0, 0.1, 0.1, 10.0],
        [50.0, 5.0, 4.0, 50.0],
        [30.0, 10.0, 8.0, 100.0]
    ])
    weights = np.array([0.4, 0.2, 0.2, 0.2])
    is_benefit = [True, False, False, False]
    
    scores = calculate_topsis(matrix, weights, is_benefit)
    assert np.argmax(scores) == 0, "En üstün tedarikçi TOPSIS'te 1. sırada çıkmadı."