import os
from optimum.onnxruntime import ORTModelForSequenceClassification

# Adjust paths to match export location
MODEL_DIR = os.path.join('models', 'distilbert-spam')
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, 'model.onnx')
INT8_MODEL_PATH = os.path.join(MODEL_DIR, 'model_int8.onnx')

# Load the exported ONNX model
ort_model = ORTModelForSequenceClassification.from_pretrained(MODEL_DIR, file_name='model.onnx')

# Apply dynamic INT8 quantization (no explicit config needed)
ort_model.quantize(save_dir=MODEL_DIR, file_name='model_int8.onnx')
print(f"✅ INT8 quantized model saved to {INT8_MODEL_PATH}")
