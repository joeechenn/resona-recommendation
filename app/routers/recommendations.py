import logging
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from app.db.fetch_ratings import fetch_ratings
from app.models.funk_svd import FunkSVD

logger = logging.getLogger(__name__)
MIN_RATINGS = 15
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

class UsersResponse(BaseModel):
    user_ids: list[str]

@router.get("/users", response_model=UsersResponse)
async def get_users(request: Request):
    user_index: dict[str, int] = request.app.state.user_index
    return UsersResponse(user_ids=list(user_index.keys()))

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
    rated_items = set(np.where(matrix[user_idx] > 0)[0].tolist())

    if len(rated_items) < MIN_RATINGS:
        logger.info({"event": "insufficient_data", "user_id": user_id, "n_ratings": len(rated_items), "threshold": MIN_RATINGS})
        return RecommendationsResponse(user_id=user_id, recommendations=[])

    results = model.recommend(user_idx, item_index, rated_items, top_n)
    logger.info({"event": "recommendations_generated", "user_id": user_id, "n_ratings": len(rated_items), "recommendations": [{"item_type": t, "spotify_id": s, "score": round(score, 3)} for t, s, score in results]})

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
