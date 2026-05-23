import os, json, time
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from onnxruntime import InferenceSession
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Paths
MODEL_DIR = os.path.join('models', 'distilbert-spam', 'best')
ONNX_PATH = os.path.join('models', 'distilbert-spam', 'model_quantized.onnx')
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)

# Load models
pytorch_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
pytorch_model.eval()
ort_session = InferenceSession(ONNX_PATH)

# Load test data (expects list of dicts with input_ids, attention_mask, label)
test_path = os.path.join('processed_data', 'test.pt')
bundle = torch.load(test_path)
input_ids = bundle['encodings']['input_ids']
attention_mask = bundle['encodings']['attention_mask']
samples = [{
    'input_ids': ids.tolist(),
    'attention_mask': mask.tolist()
} for ids, mask in zip(input_ids, attention_mask)][:1000]

def bench_torch(samples):
    latencies = []
    for s in samples:
        ids = torch.tensor(s['input_ids']).unsqueeze(0)
        mask = torch.tensor(s['attention_mask']).unsqueeze(0)
        start = time.time()
        _ = pytorch_model(ids, mask).logits
        latencies.append((time.time() - start) * 1000)
    return np.mean(latencies), np.percentile(latencies, 95)

def bench_onnx(samples):
    latencies = []
    for s in samples:
        ids = np.array(s['input_ids'], dtype=np.int64).reshape(1, -1)
        mask = np.array(s['attention_mask'], dtype=np.int64).reshape(1, -1)
        start = time.time()
        _ = ort_session.run(['logits'], {'input_ids': ids, 'attention_mask': mask})[0]
        latencies.append((time.time() - start) * 1000)
    return np.mean(latencies), np.percentile(latencies, 95)

torch_mean, torch_p95 = bench_torch(samples)
onnx_mean, onnx_p95 = bench_onnx(samples)

report_path = os.path.join('reports', 'benchmark_report.md')
with open(report_path, 'w') as f:
    f.write('# Benchmark Report\n\n')
    f.write(f'**PyTorch** - mean: {torch_mean:.2f} ms, p95: {torch_p95:.2f} ms\n\n')
    f.write(f'**ONNX INT8** - mean: {onnx_mean:.2f} ms, p95: {onnx_p95:.2f} ms\n')

print(f"Benchmark report written to {report_path}")
