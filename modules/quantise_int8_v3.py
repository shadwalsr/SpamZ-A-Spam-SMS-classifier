import os
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Paths
MODEL_DIR = os.path.join('models', 'distilbert-spam')
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, 'model.onnx')
INT8_MODEL_PATH = os.path.join(MODEL_DIR, 'model_int8.onnx')

# Load the exported ONNX model as an Optimum model
ort_model = ORTModelForSequenceClassification.from_pretrained(MODEL_DIR, file_name='model.onnx')

# Initialize quantizer
quantizer = ORTQuantizer.from_pretrained(ort_model)

# Define dynamic quantization configuration (using avx512_vnni as generic example)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

# Perform quantization
quantizer.quantize(save_dir=MODEL_DIR, quantization_config=qconfig)
print(f"INT8 quantized model saved to {INT8_MODEL_PATH}")
