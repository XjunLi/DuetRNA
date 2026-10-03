"""User-facing DuetRNA training entrypoint."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from duetrna.runtime.train import main


if __name__ == "__main__":
    main()
