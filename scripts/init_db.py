from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.db import get_db_path, init_db  # noqa: E402


def main() -> None:
    init_db()
    db_path = get_db_path()
    print(f"Database initialized at: {db_path}")


if __name__ == "__main__":
    main()
