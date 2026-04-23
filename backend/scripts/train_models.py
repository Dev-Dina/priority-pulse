"""
Model Training Runner

Thin entry point — all logic lives in app/ml/train_models.py.

Usage:
    cd backend
    python scripts/train_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.train_models import run_training

if __name__ == "__main__":
    run_training()
