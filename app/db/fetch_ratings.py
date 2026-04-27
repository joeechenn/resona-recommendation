import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import numpy as np
from numpy.typing import NDArray

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def fetch_ratings() -> tuple[NDArray[np.float64], dict[str, int], dict[str, int]]:
    query = text("""
    SELECT "userId", 'track:' || "trackId" AS "itemId", rating FROM "UserTrackStat"
    UNION ALL
    SELECT "userId", 'album:' || "albumId" AS "itemId", rating FROM "UserAlbumStat"
    UNION ALL
    SELECT "userId", 'artist:' || "artistId" AS "itemId", rating FROM "UserArtistStat"
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    user_ids = list(set(row[0] for row in rows))
    item_ids = list(set(row[1] for row in rows))

    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}

    matrix = np.zeros((len(user_ids), len(item_ids)))

    for userId, itemId, rating in rows:
        u = user_index[userId]
        i = item_index[itemId]
        matrix[u][i] = float(rating)

    return matrix, user_index, item_index