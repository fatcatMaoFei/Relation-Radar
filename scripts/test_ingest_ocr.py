from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.db import init_db  # noqa: E402
from backend.core.ingest import ingest_ocr  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test OCR ingestion: image -> text -> Event"
    )
    parser.add_argument(
        "--person-id",
        type=int,
        required=True,
        help="ID of the person associated with this event",
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file (e.g., chat screenshot)",
    )
    args = parser.parse_args()

    init_db()

    event = ingest_ocr([args.person_id], args.image_path)
    print("Created event from OCR:")
    print(event)


if __name__ == "__main__":
    main()

