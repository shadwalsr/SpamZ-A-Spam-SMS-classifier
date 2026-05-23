# Final Summary Report: SpamZ

## Overview
SpamZ is a production-ready, real-time SMS classification system. It leverages a fine-tuned DistilBERT model compressed via ONNX Runtime to deliver high-precision spam detection with sub-25ms latency. The system is designed to prioritize precision (minimizing false positives) over raw accuracy, making it suitable for real-world filtering where legitimate messages must not be blocked.

## Architecture

1.  **Machine Learning Backend (`app.py`)**: 
    *   **Framework**: FastAPI (Async REST API).
    *   **Inference Engine**: ONNX Runtime.
    *   **Model**: DistilBERT (`distilbert-base-uncased`) with INT8 dynamic quantization.
    *   **Tokenization**: HuggingFace `AutoTokenizer` (max length 128).
2.  **Web Frontend (`frontend/`)**: 
    *   **Framework**: React built with Vite.
    *   **UI/UX**: Single-page application with a minimalistic, responsive design featuring real-time classification, confidence bars, and latency metrics.
3.  **Containerization**: 
    *   Multi-stage Dockerfiles for both frontend and backend.
    *   Orchestration via `docker-compose.yml` for unified deployment.

## Performance Metrics

Extensive benchmarking was performed (`benchmark.py`) against 1,000 hold-out test samples on an Intel i7 CPU.

### Latency
The transition from PyTorch (FP32) to ONNX Runtime (INT8) yielded a **7.1x speedup**.

| Metric | PyTorch (FP32) | ONNX Runtime (INT8) |
| :--- | :--- | :--- |
| **Mean Latency** | 135.33 ms | **19.04 ms** |
| **p95 Latency** | 143.05 ms | **24.70 ms** |

### Memory & Storage
Dynamic INT8 quantization via HuggingFace Optimum reduced the model footprint by **75%**.

| Format | File Size |
| :--- | :--- |
| PyTorch Checkpoint | ~268 MB |
| ONNX FP32 | ~268 MB |
| **ONNX INT8** | **~67 MB** |

## Quality Assurance & Testing

A comprehensive 3-tier `pytest` suite guarantees system stability:

1.  **Unit Tests (`test_tokenizer.py`)**: Validates HuggingFace tokenizer tensor shapes, truncation logic, and presence of special tokens (`[CLS]`, `[SEP]`).
2.  **Model Tests (`test_model.py`)**: Executes the ONNX graph against known deterministic spam/ham examples, verifying expected logit shapes and probability distributions.
3.  **Integration & Regression (`test_api.py`)**: 
    *   Validates the FastAPI `/predict` endpoint contract (HTTP status codes, Pydantic validation for missing fields).
    *   **Latency Regression**: Enforces a strict **100ms ceiling** for single, average, and p95 inference latencies. CI builds will fail if this threshold is breached.

## Deliverables

*   `app.py`: High-performance FastAPI server.
*   `frontend/`: Complete React source code and build tools.
*   `models/`: The INT8 quantized ONNX artifact (`model_int8.onnx`).
*   `tests/`: The full pytest suite.
*   `reports/`: Documentation including EDA findings, training strategies, and benchmarks.
*   `README.md`: Developer onboarding, API documentation, and Quick Start guides.

## Future Recommendations

1.  **Continuous Training Pipeline**: Implement a feedback loop where false positives/negatives reported by users are periodically aggregated to retrain the model.
2.  **GPU Acceleration**: If deployed on NVIDIA hardware, explore ONNX Runtime with TensorRT execution providers for further latency reduction.
3.  **Analytics Dashboard**: Integrate a telemetry sink (e.g., Prometheus/Grafana) to monitor production latency drift and confidence score distributions over time.
