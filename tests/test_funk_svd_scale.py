import time
import numpy as np
import pytest
from app.models.funk_svd import FunkSVD

@pytest.fixture(scope="module")
def large_matrix():
    rng = np.random.default_rng(42)
    n_users, n_items = 300, 100000
    density = 0.02

    # random sparsity
    matrix = np.zeros((n_users, n_items))
    mask = rng.random((n_users, n_items)) < density
    matrix[mask] = rng.uniform(1, 10, size=mask.sum())

    return matrix

@pytest.fixture(scope="module")
def trained_model(large_matrix):
    model = FunkSVD(n_factors=20, epochs=50)
    model.fit(large_matrix)
    return model

def test_fit_time(large_matrix):
    model = FunkSVD(n_factors=20, epochs=50)
    start = time.perf_counter()
    model.fit(large_matrix)
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0, f"fit() took {elapsed:.1f}s"

