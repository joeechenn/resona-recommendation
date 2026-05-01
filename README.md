# resona-recommendation

ML microservice for [Resona](https://github.com/joeechenn/resona), a music social discovery app. Generates personalized music recommendations using Funk SVD collaborative filtering trained on user rating data from NeonDB.

---

## How it works

Users rate tracks, artists, and albums on a 0–10 scale in Resona. This service pulls those ratings, trains a matrix factorization model, and exposes endpoints for personalized recommendations and taste-compatible user discovery.

The model is trained in-process on startup (and on demand via `/model/retrain`). No model is persisted to disk — restarting the server retrains from the latest ratings.

A nightly script (`scripts/nightly_run.py`) runs as a dark launch: it retrains the model and fires recommendation requests for every known user. Predictions are logged server-side but not yet surfaced in the UI.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/recommendations/{user_id}` | Top-N recommendations for a user. Returns `[]` if user has fewer than 15 ratings. |
| `GET` | `/users/{user_id}/similar` | Top-N taste-compatible users via cosine similarity on latent vectors. |
| `GET` | `/users` | All user IDs with at least one rating. |
| `POST` | `/model/retrain` | Re-fetches ratings from DB and retrains the model in-place. |

---

## Tech stack

- **Python 3.11** — Conda environment `resona`
- **FastAPI + Uvicorn** — API framework and ASGI server
- **PyTorch** — GPU-accelerated batch gradient descent for SVD training
- **NumPy** — matrix construction and post-training inference
- **SQLAlchemy + psycopg2** — read-only access to NeonDB (PostgreSQL)
- **Pydantic** — request/response validation
- **python-dotenv** — environment variable loading

---

## Status

The recommendation pipeline is fully operational end-to-end. Deployment to production (Railway or Render) is deferred until the rating matrix is dense enough for meaningful predictions.
