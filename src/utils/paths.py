import os
from pathlib import Path

# Assumed structure:
#   PROJECT_ROOT/
#     configs/
#     data/
#     src/
#     ...

# This file is in: PROJECT_ROOT/src/utils/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "gwosc_cache"
CONFIG_DIR = PROJECT_ROOT / "configs"

def project_path(*parts: str) -> Path:
    """
    Returns an absolute path within the repo, regardless of cwd.

    Example:
      project_path("configs", "dataset.yaml")
      project_path("data", "gw_multi_events_sft.h5")
    """
    return PROJECT_ROOT.joinpath(*parts)

def ensure_dir(path: Path) -> None:
    """
    Creates a directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)