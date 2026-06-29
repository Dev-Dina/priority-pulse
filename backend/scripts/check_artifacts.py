"""Fail fast if required model artifacts are missing.

Used as the container startup gate (see backend/Dockerfile) so a misconfigured
deployment aborts with a clear, actionable message instead of silently serving
UNAVAILABLE responses. Local `uvicorn` dev still degrades gracefully — this gate
runs only in the Docker image CMD.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


def main() -> None:
    required = [
        settings.MODEL_DIR / "priority_classifier.pkl",
        settings.MODEL_DIR / "feature_names.pkl",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        listed = "\n  ".join(str(p) for p in missing)
        sys.stderr.write(
            "\n[startup] Missing required model artifacts:\n  "
            f"{listed}\n\n"
            "The data/ML pipeline has not been run. From ./backend:\n"
            "  uv run python scripts/run_data_pipeline.py\n"
            "  uv run python scripts/init_db.py\n"
            "  uv run python scripts/populate_vectors.py\n"
            "  uv run python scripts/train_models.py\n\n"
            "Aborting so the API does not serve UNAVAILABLE responses.\n"
        )
        raise SystemExit(1)
    print("[startup] Model artifacts present.")


if __name__ == "__main__":
    main()
