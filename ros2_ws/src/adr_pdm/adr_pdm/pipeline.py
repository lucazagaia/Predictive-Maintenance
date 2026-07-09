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


def repo_root() -> Path:
    env = os.environ.get("ADR_PDM_ROOT")
    if env:
        return Path(env)
    # this file: ros2_ws/src/adr_pdm/adr_pdm/pipeline.py  ->  repo root is parents[4]
    return Path(__file__).resolve().parents[4]


ROOT = repo_root()
SRC = ROOT / "src"
MODELS = ROOT / "models"
SAMPLES = ROOT / "data" / "samples"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
