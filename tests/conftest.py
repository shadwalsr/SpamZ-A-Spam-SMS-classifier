"""Shared fixtures for SpamZ test suite."""
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def tokenizer():
    """Load the DistilBERT tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


@pytest.fixture(scope="session")
def onnx_session():
    """Load the quantised ONNX model; skip if the file is missing."""
    import onnxruntime as ort

    model_path = PROJECT_ROOT / "models" / "distilbert-spam" / "model_int8.onnx"
    if not model_path.exists():
        pytest.skip("ONNX model not found")
    return ort.InferenceSession(str(model_path))


@pytest.fixture(scope="session")
def client():
    """Create a FastAPI TestClient; skip if app cannot be imported."""
    from starlette.testclient import TestClient

    try:
        from modules.app import app  # noqa: WPS433
    except Exception as exc:
        pytest.skip(f"Could not import app: {exc}")
    return TestClient(app)
