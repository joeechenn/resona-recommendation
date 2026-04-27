import numpy as np
import torch
from numpy.typing import NDArray

class FunkSVD:
    def __init__(self, n_factors: int = 20, learning_rate: float = 0.02, alpha: float = 0.02, beta: float = 0.02, epochs: int = 20) -> None:
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs
        # user latent factor matrix, shape (n_users, n_factors)
        self.U = None
        # item latent factor matrix, shape (n_items, n_factors)
        self.V = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, matrix: NDArray[np.float64]) -> None:
        self.matrix = matrix
        n_users, n_items = matrix.shape

        # small random init prevents symmetry
        U = torch.normal(0, 0.1, (n_users, self.n_factors), device=self.device)
        V = torch.normal(0, 0.1, (n_items, self.n_factors), device=self.device)

        # get indices and values of known ratings, move to GPU
        matrix_tensor = torch.tensor(matrix, dtype=torch.float32, device=self.device)
        rows, cols = torch.where(matrix_tensor > 0)
        ratings = matrix_tensor[rows, cols]

        for epoch in range(self.epochs):
            preds = (U[rows] * V[cols]).sum(dim=1)
            errors = ratings - preds

            # snapshot U[rows] before updating so V's gradient uses the pre-update value
            u_old = U[rows].clone()
            U.index_add_(0, rows, self.learning_rate * (errors.unsqueeze(1) * V[cols] - self.alpha * U[rows]))
            V.index_add_(0, cols, self.learning_rate * (errors.unsqueeze(1) * u_old - self.beta * V[cols]))

        # move back to CPU for predict and recommend
        self.U = U.cpu().numpy()
        self.V = V.cpu().numpy()

    def predict(self, user_idx: int, item_idx: int) -> float:
        # clamp to [0, 10] to match Resona's rating scale
        return float(np.clip(self.U[user_idx] @ self.V[item_idx], 0, 10))

    def recommend(self, user_idx: int, item_index: dict[str, int], rated_items: set[int], top_n: int = 10) -> list[tuple[str, str, float]]:
        scores: list[tuple[str, str, float]] = []
        for namespaced_id, idx in item_index.items():
            if idx in rated_items:
                continue
            # strip collision-avoidance prefix (like "track:abc123" -> ("track", "abc123"))
            item_type, spotify_id = namespaced_id.split(":", 1)
            scores.append((item_type, spotify_id, self.predict(user_idx, idx)))
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_n]
