from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

model_name = "mariagrandury/distilbert-base-uncased-finetuned-sms-spam-detection"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Ensure output directory exists
out_dir = os.path.join(os.getcwd(), "models", "distilbert-spam", "best")
os.makedirs(out_dir, exist_ok=True)

# Save
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print("Model and tokenizer saved to", out_dir)
