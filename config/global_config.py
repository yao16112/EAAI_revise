import json
import os
from pathlib import Path

PATHS_FILE = Path(os.environ.get("MARS_PATHS_CONFIG", Path(__file__).resolve().parents[1] / "configs/paths.json"))
with PATHS_FILE.open() as f:
    PATHS = json.load(f)
DATASET_ROOTS = PATHS["datasets"]
DATA_ROOT = DATASET_ROOTS["mvtec"]
