"""
nightly dark-launch script
retrains the model then fires recommendation requests for every known user
structured output is logged by the FastAPI server, this script only drives the requests

usage:
    python scripts/nightly_run.py [--base-url http://localhost:8000]
"""

import argparse
import logging
import sys
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(base_url: str) -> None:
    # retrain on latest ratings
    logger.info("Triggering model retrain...")
    resp = requests.post(f"{base_url}/model/retrain", timeout=120)
    resp.raise_for_status()
    retrain = resp.json()
    logger.info(f"Retrain complete: {retrain['n_users']} users, {retrain['n_items']} items")

    # fetch all known users
    resp = requests.get(f"{base_url}/users", timeout=30)
    resp.raise_for_status()
    user_ids: list[str] = resp.json()["user_ids"]
    logger.info(f"Found {len(user_ids)} users, requesting recommendations...")

    success = 0
    insufficient = 0
    errors = 0

    for user_id in user_ids:
        try:
            resp = requests.get(f"{base_url}/recommendations/{user_id}", timeout=30)
            resp.raise_for_status()
            if resp.json()["recommendations"]:
                success += 1
            else:
                insufficient += 1
        except requests.RequestException as e:
            logger.error(f"Request failed for user {user_id}: {e}")
            errors += 1

    logger.info(f"Nightly run complete — success: {success}, insufficient_data: {insufficient}, errors: {errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    try:
        run(args.base_url)
    except Exception as e:
        logger.error(f"Nightly run failed: {e}")
        sys.exit(1)
