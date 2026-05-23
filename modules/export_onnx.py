import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

MODEL_DIR = os.path.join('models', 'distilbert-spam', 'best')
OUTPUT_ONNX = os.path.join('models', 'distilbert-spam', 'model.onnx')

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Set model to eval mode
model.eval()

# Create dummy input
sample_text = "This is a sample SMS message for export."
inputs = tokenizer(sample_text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)

# Export to ONNX
torch.onnx.export(
    model,
    (inputs['input_ids'], inputs['attention_mask']),
    OUTPUT_ONNX,
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'seq'},
        'attention_mask': {0: 'batch', 1: 'seq'},
        'logits': {0: 'batch'}
    },
    opset_version=14,
    do_constant_folding=True
)
print(f"ONNX model saved to {OUTPUT_ONNX}")
