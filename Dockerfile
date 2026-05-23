FROM python:3.11-slim AS backend

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir onnxruntime fastapi uvicorn

# Copy application code and model assets
COPY modules/ modules/
COPY models/ models/
COPY processed_data/optimal_threshold.json processed_data/optimal_threshold.json

EXPOSE 8000
CMD ["uvicorn", "modules.app:app", "--host", "0.0.0.0", "--port", "8000"]
