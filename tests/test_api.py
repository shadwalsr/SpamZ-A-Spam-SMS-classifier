"""Integration tests — FastAPI /predict endpoint."""
import pytest


class TestPredictEndpoint:
    """POST /predict integration tests."""

    def test_status_200(self, client):
        """Valid request returns 200."""
        resp = client.post("/predict", json={"text": "Hello friend!"})
        assert resp.status_code == 200

    def test_response_schema(self, client):
        """Response must contain label, confidence, latency_ms."""
        resp = client.post("/predict", json={"text": "Hello friend!"})
        data = resp.json()
        assert "label" in data
        assert "confidence" in data
        assert "latency_ms" in data

    def test_label_values(self, client):
        """Label must be either 'spam' or 'ham'."""
        resp = client.post("/predict", json={"text": "Hello!"})
        assert resp.json()["label"] in ("spam", "ham")

    def test_confidence_range(self, client):
        """Confidence should be between 0 and 1."""
        resp = client.post("/predict", json={"text": "Win a free iPhone now!"})
        conf = resp.json()["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_latency_positive(self, client):
        """Latency must be a positive number."""
        resp = client.post("/predict", json={"text": "Test message"})
        assert resp.json()["latency_ms"] > 0

    def test_empty_text_returns_400(self, client):
        """Empty text body should return 400."""
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 400

    def test_missing_text_returns_422(self, client):
        """Missing 'text' field should return 422 (Pydantic validation)."""
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_response_types(self, client):
        """Verify response field types."""
        resp = client.post("/predict", json={"text": "Hello"})
        data = resp.json()
        assert isinstance(data["label"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["latency_ms"], float)


class TestLatencyRegression:
    """Latency ceiling tests — CI should fail if inference gets too slow."""

    LATENCY_CEILING_MS = 100  # Maximum acceptable inference latency

    def test_single_inference_under_ceiling(self, client):
        """A single prediction must complete within the latency ceiling."""
        resp = client.post("/predict", json={"text": "Check this message"})
        latency = resp.json()["latency_ms"]
        assert latency < self.LATENCY_CEILING_MS, (
            f"Inference latency {latency:.1f}ms exceeds ceiling of {self.LATENCY_CEILING_MS}ms"
        )

    def test_average_latency_under_ceiling(self, client):
        """Average latency over 10 requests must stay under the ceiling."""
        messages = [
            "Hello!",
            "Win a free prize now! Call 0800-WINNER",
            "Are you free for dinner tonight?",
            "URGENT: Your bank account needs verification",
            "Meeting at 3pm, don't forget!",
            "Congratulations, you've been selected!",
            "Can you send me the report?",
            "FREE unlimited texts! Reply YES to claim",
            "I'll pick you up at 6",
            "Your parcel is ready for collection",
        ]
        latencies = []
        for msg in messages:
            resp = client.post("/predict", json={"text": msg})
            latencies.append(resp.json()["latency_ms"])

        avg = sum(latencies) / len(latencies)
        assert avg < self.LATENCY_CEILING_MS, (
            f"Average latency {avg:.1f}ms exceeds ceiling of {self.LATENCY_CEILING_MS}ms"
        )

    def test_p95_latency_under_ceiling(self, client):
        """p95 latency over 20 requests must stay under the ceiling."""
        import numpy as np
        latencies = []
        for _ in range(20):
            resp = client.post("/predict", json={"text": "Sample text for latency test"})
            latencies.append(resp.json()["latency_ms"])

        p95 = float(np.percentile(latencies, 95))
        assert p95 < self.LATENCY_CEILING_MS, (
            f"p95 latency {p95:.1f}ms exceeds ceiling of {self.LATENCY_CEILING_MS}ms"
        )
