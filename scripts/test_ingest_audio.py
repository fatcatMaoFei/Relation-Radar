from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.db import init_db  # noqa: E402
from backend.core.ingest import ingest_audio  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test audio ingestion: audio -> text -> Event"
    )
    parser.add_argument(
        "--person-id",
        type=int,
        required=True,
        help="ID of the person associated with this event",
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file (e.g., wav/mp3/m4a)",
    )
    args = parser.parse_args()

    init_db()

    event = ingest_audio([args.person_id], args.audio_path)
    print("Created event from audio transcription:")
    print(event)


if __name__ == "__main__":
    main()

