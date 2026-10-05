import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SAMPLES = ROOT / "data" / "samples"

sys.path.insert(0, str(ROOT / "src"))

from cloud import CloudNode  # noqa: E402
from edge import EdgeNode    # noqa: E402


@pytest.fixture(scope="session")
def edge():
    return EdgeNode(
        model_path=str(MODELS / "mvt_flow_voraus_ad.pt"),
        scaler_path=str(MODELS / "scaler_voraus_ad.pkl"),
    )


@pytest.fixture(scope="session")
def cloud():
    node = CloudNode()
    assert node.predictor.model_loaded, "committed RUL weights failed to load"
    return node
