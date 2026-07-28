"""
Bridge from the ROS2 nodes to the repo's inference code.

The rclpy nodes are thin wrappers: all model/decision logic stays in the repo's src/
(detection.py, prediction.py, fusion.py, the model definitions). This module locates the
repo, puts src/ on sys.path, and exposes the model/sample paths so a node can simply do:

    from adr_pdm.pipeline import MODELS          # side effect: src/ importable
    from detection import MVTFlowDetectionInterface
"""

import os
import sys
from pathlib import Path


def _looks_like_repo(path: Path) -> bool:
    return (path / "src" / "detection.py").is_file() and (path / "models").is_dir()


def repo_root() -> Path:
    """
    Locate the repo holding src/ and models/.

    ADR_PDM_ROOT wins when set (the Docker image sets it). Otherwise walk up from this
    file looking for the repo layout: colcon installs this package to
    ros2_ws/install/adr_pdm/lib/pythonX/site-packages/adr_pdm/, so a fixed number of
    parents only works when running from source — walking works for both.
    """
    env = os.environ.get("ADR_PDM_ROOT")
    if env:
        return Path(env)

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if _looks_like_repo(candidate):
            return candidate

    raise RuntimeError(
        "Could not locate the repository (no src/detection.py + models/ above "
        f"{here}). Set ADR_PDM_ROOT to the repo root."
    )


ROOT = repo_root()
SRC = ROOT / "src"
MODELS = ROOT / "models"
SAMPLES = ROOT / "data" / "samples"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
