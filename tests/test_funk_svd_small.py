import numpy as np
import pytest
from app.models.funk_svd import FunkSVD

@pytest.fixture
def small_matrix():
    # 4 users x 5 items, intentionally sparse to mimic real rating density
    m = np.zeros((4, 5))
    m[0][0] = 8.0
    m[0][2] = 6.0
    m[1][1] = 9.0
    m[1][3] = 4.0
    m[2][0] = 7.0
    m[2][4] = 5.0
    m[3][2] = 3.0
    m[3][3] = 10.0
    return m

@pytest.fixture
def trained_model(small_matrix):
    model = FunkSVD(n_factors=5, epochs=50)
    model.fit(small_matrix)
    return model

def test_fit_initializes_factors(trained_model, small_matrix):
    n_users, n_items = small_matrix.shape
    assert trained_model.U.shape == (n_users, 5)
    assert trained_model.V.shape == (n_items, 5)

def test_predict_within_rating_scale(trained_model, small_matrix):
    # every (u, i) pair, including unobserved ones, must stay in [0, 10]
    n_users, n_items = small_matrix.shape
    for u in range(n_users):
        for i in range(n_items):
            score = trained_model.predict(u, i)
            assert 0.0 <= score <= 10.0, f"predict({u}, {i}) = {score} out of [0, 10]"

def test_predict_known_ratings_close(trained_model, small_matrix):
    # model should reconstruct observed ratings reasonably after 50 epochs
    known = np.argwhere(small_matrix > 0)
    errors = [abs(trained_model.predict(u, i) - small_matrix[u][i]) for u, i in known]
    assert max(errors) < 3.0, f"max error on known ratings too high: {max(errors):.2f}"

def test_recommend_excludes_rated_items(trained_model, small_matrix):
    item_index = {f"track:item{i}": i for i in range(small_matrix.shape[1])}
    rated = set(np.where(small_matrix[0] > 0)[0].tolist())
    results = trained_model.recommend(0, item_index, rated, top_n=10)
    returned_ids = {int(sid.replace("item", "")) for _, sid, _ in results}
    assert returned_ids.isdisjoint(rated)

def test_recommend_returns_at_most_top_n(trained_model, small_matrix):
    item_index = {f"track:item{i}": i for i in range(small_matrix.shape[1])}
    rated = set(np.where(small_matrix[0] > 0)[0].tolist())
    results = trained_model.recommend(0, item_index, rated, top_n=2)
    assert len(results) <= 2

def test_recommend_sorted_descending(trained_model, small_matrix):
    item_index = {f"track:item{i}": i for i in range(small_matrix.shape[1])} \
    # no rated items so all are candidates
    rated: set[int] = set()
    results = trained_model.recommend(0, item_index, rated, top_n=10)
    scores = [score for _, _, score in results]
    assert scores == sorted(scores, reverse=True)

def test_recommend_strips_namespace(trained_model, small_matrix):
    # verifies that "track:abc123" → item_type="track", spotify_id="abc123"
    item_index = {
        "track:abc": 0,
        "album:xyz": 1,
        "artist:def": 2,
        "track:ghi": 3,
        "album:jkl": 4,
    }
    rated: set[int] = set()
    results = trained_model.recommend(0, item_index, rated, top_n=10)
    for item_type, spotify_id, _ in results:
        assert ":" not in item_type
        assert ":" not in spotify_id
        assert item_type in {"track", "album", "artist"}
