import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

app = FastAPI(title="SMS Spam Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

# Load ONNX model and tokenizer once at startup
MODEL_PATH = "./models/distilbert-spam/model_int8.onnx"
TOKENIZER_PATH = "./models/distilbert-spam"
ort = InferenceSession(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
with open("processed_data/optimal_threshold.json") as f:
    THRESH = json.load(f)["threshold"]

@app.post("/predict")
async def predict(msg: Message):
    if not msg.text:
        raise HTTPException(status_code=400, detail="Empty text")
    inputs = tokenizer(msg.text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
    ort_inputs = {"input_ids": inputs["input_ids"].astype(np.int64), "attention_mask": inputs["attention_mask"].astype(np.int64)}
    start = time.perf_counter()
    logits = ort.run(["logits"], ort_inputs)[0]
    latency_ms = (time.perf_counter() - start) * 1000
    prob_spam = (np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))[0, 1]
    label = "spam" if prob_spam >= THRESH else "ham"
    return {"label": label, "confidence": float(prob_spam), "latency_ms": latency_ms}
