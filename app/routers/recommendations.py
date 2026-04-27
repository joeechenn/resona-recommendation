import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from app.db.fetch_ratings import fetch_ratings
from app.models.funk_svd import FunkSVD

router = APIRouter()

class RecommendationItem(BaseModel):
    item_type: str
    spotify_id: str
    score: float

class RecommendationsResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendationItem]


class SimilarUser(BaseModel):
    user_id: str
    # cosine similarity [-1, 1]
    similarity: float


class SimilarUsersResponse(BaseModel):
    user_id: str
    similar_users: list[SimilarUser]

class RetrainResponse(BaseModel):
    status: str
    n_users: int
    n_items: int

@router.get("/recommendations/{user_id}", response_model=RecommendationsResponse)
async def get_recommendations(user_id: str, request: Request, top_n: int = 10):
    model: FunkSVD = request.app.state.model
    user_index: dict[str, int] = request.app.state.user_index
    item_index: dict[str, int] = request.app.state.item_index
    matrix = request.app.state.matrix

    # user_index only contains users who have at least one rating, no ratings means no latent vector
    if user_id not in user_index:
        raise HTTPException(status_code=404, detail="User has no ratings")

    user_idx = user_index[user_id]

    # find column indices where this user has already rated something
    rated_items = set(np.where(matrix[user_idx] > 0)[0].tolist())
    results = model.recommend(user_idx, item_index, rated_items, top_n)

    return RecommendationsResponse(
        user_id=user_id,
        recommendations=[
            RecommendationItem(item_type=item_type, spotify_id=spotify_id, score=score)
            for item_type, spotify_id, score in results
        ],
    )

@router.get("/users/{user_id}/similar", response_model=SimilarUsersResponse)
async def get_similar_users(user_id: str, request: Request, top_n: int = 10):
    model: FunkSVD = request.app.state.model
    user_index: dict[str, int] = request.app.state.user_index

    # same check, user must have ratings
    if user_id not in user_index:
        raise HTTPException(status_code=404, detail="User has no ratings")

    user_idx = user_index[user_id]

    # normalize each row of U to unit length, then dot product = cosine similarity
    norms = np.linalg.norm(model.U, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero for users with no latent signal
    U_norm = model.U / norms

    # one matrix multiply gives cosine similarity between the target user and all others
    sims = U_norm @ U_norm[user_idx]
    # exclude self from results
    sims[user_idx] = -1

    idx_to_user = {idx: uid for uid, idx in user_index.items()}
    top_indices = np.argsort(sims)[::-1][:top_n]

    return SimilarUsersResponse(
        user_id=user_id,
        similar_users=[
            SimilarUser(user_id=idx_to_user[int(i)], similarity=float(sims[i]))
            for i in top_indices
        ],
    )

@router.post("/model/retrain", response_model=RetrainResponse)
async def retrain_model(request: Request):
    # re-fetch ratings and train a fresh model, then swap it into app.state atomically
    matrix, user_index, item_index = fetch_ratings()
    model = FunkSVD()
    model.fit(matrix)
    request.app.state.model = model
    request.app.state.user_index = user_index
    request.app.state.item_index = item_index
    request.app.state.matrix = matrix

    return RetrainResponse(
        status="retrained",
        n_users=len(user_index),
        n_items=len(item_index),
    )
