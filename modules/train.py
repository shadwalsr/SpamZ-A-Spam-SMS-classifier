import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
import evaluate
from datasets import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import os
import json

def load_data():
    train_dict = torch.load('processed_data/train.pt')
    val_dict = torch.load('processed_data/val.pt')
    test_dict = torch.load('processed_data/test.pt')
    
    # Convert to HuggingFace Dataset format for Trainer
    train_dataset = Dataset.from_dict({
        'input_ids': train_dict['encodings']['input_ids'],
        'attention_mask': train_dict['encodings']['attention_mask'],
        'labels': train_dict['labels']
    })
    val_dataset = Dataset.from_dict({
        'input_ids': val_dict['encodings']['input_ids'],
        'attention_mask': val_dict['encodings']['attention_mask'],
        'labels': val_dict['labels']
    })
    return train_dataset, val_dataset

# Custom Trainer to apply class weights
class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load weights computed in Phase 1
        weights = np.load('processed_data/class_weights.npy')
        self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.args.device)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = np.argmax(probs, axis=1)
    
    # Calculate metrics at standard 0.5 threshold
    clf_metrics = evaluate.load("evaluate-modules/metrics/evaluate-metric--clf_metrics/clf_metrics.py") # Custom fallback if needed, but let's use sklearn
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    auc_roc = roc_auc_score(labels, probs[:, 1])
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }

def train():
    print("Loading datasets...")
    train_dataset, val_dataset = load_data()
    
    print("Initializing DistilBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    
    training_args = TrainingArguments(
        output_dir='./models/distilbert-spam',
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        report_to="none"
    )
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving best model...")
    trainer.save_model('./models/distilbert-spam/best')
    
    # Step 4: Threshold Tuning
    print("Evaluating optimal decision threshold on validation set...")
    predictions = trainer.predict(val_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()[:, 1]
    labels = predictions.label_ids
    
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    
    # Plot PR vs Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='lower left')
    plt.title('Precision and Recall vs Decision Threshold')
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/precision_recall_thresholds.png')
    print("Saved PR threshold plot to plots/precision_recall_thresholds.png")
    
    # Find optimal threshold: highest precision while keeping recall > 0.85
    optimal_idx = np.where(recalls[:-1] >= 0.85)[0][-1] # Last index where recall >= 0.85
    optimal_threshold = thresholds[optimal_idx]
    opt_precision = precisions[optimal_idx]
    opt_recall = recalls[optimal_idx]
    
    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    print(f"Expected Precision: {opt_precision:.4f}")
    print(f"Expected Recall: {opt_recall:.4f}")
    
    with open('processed_data/optimal_threshold.json', 'w') as f:
        json.dump({'threshold': float(optimal_threshold)}, f)

if __name__ == '__main__':
    train()
