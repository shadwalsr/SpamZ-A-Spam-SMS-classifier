import os, json, torch, numpy as np
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

MODEL_DIR = os.path.join('models', 'distilbert-spam')
ONNX_MODEL = os.path.join('models', 'distilbert-spam', 'model_quantized.onnx')
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)

# Load test data from Torch bundle
test_path = os.path.join('processed_data', 'test.pt')
bundle = torch.load(test_path)
input_ids = bundle['encodings']['input_ids'].tolist()
attention_masks = bundle['encodings']['attention_mask'].tolist()
labels = bundle['labels'].tolist()

test_data = [{'input_ids': ids, 'attention_mask': mask, 'label': lbl} for ids, mask, lbl in zip(input_ids, attention_masks, labels)]

# Load ONNX model
ort_session = InferenceSession(ONNX_MODEL)

probs = []
true_labels = []
for item in test_data:
    ids = np.array(item['input_ids'], dtype=np.int64).reshape(1, -1)
    mask = np.array(item['attention_mask'], dtype=np.int64).reshape(1, -1)
    logits = ort_session.run(['logits'], {'input_ids': ids, 'attention_mask': mask})[0]
    prob_spam = torch.softmax(torch.from_numpy(logits), dim=1)[0,1].item()
    probs.append(prob_spam)
    true_labels.append(1 if item['label']==1 else 0)

# Load threshold
with open(os.path.join('processed_data', 'optimal_threshold.json')) as f:
    thresh = json.load(f)['threshold']

preds = [1 if p >= thresh else 0 for p in probs]

precision = precision_score(true_labels, preds, zero_division=0)
recall = recall_score(true_labels, preds, zero_division=0)
f1 = f1_score(true_labels, preds, zero_division=0)
roc = roc_auc_score(true_labels, probs)

report = ("# ONNX INT8 Accuracy Validation\n\n"
          f"Precision: {precision:.4f}\n"
          f"Recall: {recall:.4f}\n"
          f"F1: {f1:.4f}\n"
          f"ROC-AUC: {roc:.4f}\n")
report_path = os.path.join('reports', 'onnx_accuracy.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"Accuracy report written to {report_path}")
