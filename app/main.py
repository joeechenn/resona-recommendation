from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.db.fetch_ratings import fetch_ratings
from app.models.funk_svd import FunkSVD
from app.routers import recommendations

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # train once on startup
    # POST /model/retrain replaces these in-place
    matrix, user_index, item_index = fetch_ratings()
    model = FunkSVD()
    model.fit(matrix)

    # store on app.state so all request handlers share the same trained model
    app.state.model = model
    app.state.user_index = user_index
    app.state.item_index = item_index
    # kept for rated-item lookup in recommendations
    app.state.matrix = matrix
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(recommendations.router)
