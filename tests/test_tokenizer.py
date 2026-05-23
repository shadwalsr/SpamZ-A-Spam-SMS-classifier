"""Unit tests — tokenizer output shape and token count."""
import pytest


class TestTokenizer:
    """Verify the DistilBERT tokenizer produces expected shapes."""

    def test_output_keys(self, tokenizer):
        """Tokenizer must return input_ids and attention_mask."""
        enc = tokenizer("hello", return_tensors="np", padding="max_length", truncation=True, max_length=128)
        assert "input_ids" in enc
        assert "attention_mask" in enc

    def test_shape_single(self, tokenizer):
        """Single input should produce shape (1, 128)."""
        enc = tokenizer("hello", return_tensors="np", padding="max_length", truncation=True, max_length=128)
        assert enc["input_ids"].shape == (1, 128)
        assert enc["attention_mask"].shape == (1, 128)

    def test_truncation(self, tokenizer):
        """A very long input must still be truncated to 128 tokens."""
        long_text = "word " * 500
        enc = tokenizer(long_text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
        assert enc["input_ids"].shape == (1, 128)

    def test_empty_input(self, tokenizer):
        """Empty string should still produce (1, 128) with padding."""
        enc = tokenizer("", return_tensors="np", padding="max_length", truncation=True, max_length=128)
        assert enc["input_ids"].shape == (1, 128)

    def test_special_tokens_present(self, tokenizer):
        """CLS and SEP tokens must be present in the encoded output."""
        enc = tokenizer("hello", return_tensors="np", padding="max_length", truncation=True, max_length=128)
        ids = enc["input_ids"][0].tolist()
        assert ids[0] == tokenizer.cls_token_id
        # SEP should appear somewhere after CLS
        assert tokenizer.sep_token_id in ids

    def test_dtype_int(self, tokenizer):
        """Token IDs must be integers."""
        import numpy as np
        enc = tokenizer("hello", return_tensors="np", padding="max_length", truncation=True, max_length=128)
        assert np.issubdtype(enc["input_ids"].dtype, np.integer)
