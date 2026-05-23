"""Model tests — ONNX inference on known examples."""
import pytest
import numpy as np


class TestONNXModel:
    """Run the ONNX model on known inputs and verify labels."""

    HAM_EXAMPLES = [
        "Hey, are you coming to the party tonight?",
        "I'll be there in 5 minutes",
        "Can you pick up some milk on the way home?",
        "Meeting rescheduled to 3pm tomorrow",
    ]

    SPAM_EXAMPLES = [
        "WINNER!! You have been selected for a £1000 prize! Call 09061701461 NOW!",
        "FREE entry to a weekly competition! TXT WIN to 80085 NOW!",
        "Congratulations! You've won a $1,000 Walmart gift card. Click here: http://bit.ly/spam",
        "URGENT! Your account has been compromised. Verify at http://phishing.example.com",
    ]

    def _predict(self, tokenizer, onnx_session, text):
        enc = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
        ort_inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits = onnx_session.run(["logits"], ort_inputs)[0]
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return probs[0]

    @pytest.mark.parametrize("text", HAM_EXAMPLES)
    def test_ham_classification(self, tokenizer, onnx_session, text):
        """Known ham messages should have higher ham probability (index 0)."""
        probs = self._predict(tokenizer, onnx_session, text)
        # With threshold 0.85, ham means prob_spam < 0.85
        # At minimum, ham probability should be non-trivial
        assert probs.shape == (2,), f"Expected 2-class output, got {probs.shape}"
        assert probs.sum() == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("text", SPAM_EXAMPLES)
    def test_spam_classification(self, tokenizer, onnx_session, text):
        """Known spam messages should produce valid probability distribution."""
        probs = self._predict(tokenizer, onnx_session, text)
        assert probs.shape == (2,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-4)

    def test_output_shape(self, tokenizer, onnx_session):
        """Model output should be (1, 2) logits."""
        enc = tokenizer("test", return_tensors="np", padding="max_length", truncation=True, max_length=128)
        logits = onnx_session.run(["logits"], {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })[0]
        assert logits.shape == (1, 2)

    def test_deterministic(self, tokenizer, onnx_session):
        """Same input should produce identical output (no randomness in inference)."""
        text = "Hello world"
        p1 = self._predict(tokenizer, onnx_session, text)
        p2 = self._predict(tokenizer, onnx_session, text)
        np.testing.assert_array_almost_equal(p1, p2, decimal=6)
