from __future__ import annotations

"""
Simple smoke test for the Relation Radar Web API.

This script:
1. Ensures the database is initialized.
2. Optionally imports sample data.
3. Uses FastAPI TestClient to call the main endpoints:
   - GET /persons
   - GET /persons/{id}
   - GET /persons/{id}/events
The goal is to verify that the API wiring is correct.
"""

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.db import init_db  # noqa: E402
from backend.api.service import get_app  # noqa: E402
from scripts.import_sample_data import main as import_sample_data  # noqa: E402


def main() -> int:
    init_db()
    # Seed some sample data (idempotent enough for local testing)
    import_sample_data()

    app = get_app()
    client = TestClient(app)

    print("=== Testing GET /persons ===")
    resp = client.get("/persons")
    print("Status:", resp.status_code)
    if resp.status_code != 200:
        print("Body:", resp.text)
        return 1

    persons = resp.json()
    print(f"Persons count: {len(persons)}")
    if not persons:
        print("No persons in database, nothing more to test.")
        return 0

    person_id = persons[0]["id"]
    print(f"\n=== Testing GET /persons/{person_id} ===")
    resp = client.get(f"/persons/{person_id}")
    print("Status:", resp.status_code)
    print("Body:", resp.json())
    if resp.status_code != 200:
        return 1

    print(f"\n=== Testing GET /persons/{person_id}/events ===")
    resp = client.get(f"/persons/{person_id}/events", params={"limit": 5})
    print("Status:", resp.status_code)
    events = resp.json()
    print(f"Events count: {len(events)}")
    if resp.status_code != 200:
        print("Body:", resp.text)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
