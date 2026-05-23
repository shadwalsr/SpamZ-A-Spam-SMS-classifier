import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from transformers import AutoTokenizer
import torch
import os

def prepare_pipeline():
    print("Starting Data Pipeline...")
    # Load and clean
    df = pd.read_csv('spam.csv', encoding='latin-1', usecols=[0, 1])
    df.columns = ['label', 'text']
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Map labels to integers
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Stratified Split (80/10/10)
    # First split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['text'].values, df['label'].values, 
        test_size=0.2, stratify=df['label'].values, random_state=42
    )
    
    # Second split: temp into 50% val, 50% test (each 10% of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"Train size: {len(X_train)} (Spam: {sum(y_train)/len(y_train):.2%})")
    print(f"Val size: {len(X_val)} (Spam: {sum(y_val)/len(y_val):.2%})")
    print(f"Test size: {len(X_test)} (Spam: {sum(y_test)/len(y_test):.2%})")
    
    # Compute Class Weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = {cls: weight for cls, weight in zip(classes, weights)}
    print(f"\nComputed Class Weights: {class_weights_dict}")
    
    # Save weights to a file for later use
    os.makedirs('processed_data', exist_ok=True)
    np.save('processed_data/class_weights.npy', weights)
    
    # Tokenization
    print("\nTokenizing with distilbert-base-uncased...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize_texts(texts):
        return tokenizer(
            texts.tolist(),
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
    
    train_encodings = tokenize_texts(X_train)
    val_encodings = tokenize_texts(X_val)
    test_encodings = tokenize_texts(X_test)
    
    # Save processed data
    print("Saving tokenized datasets...")
    torch.save({'encodings': train_encodings, 'labels': torch.tensor(y_train)}, 'processed_data/train.pt')
    torch.save({'encodings': val_encodings, 'labels': torch.tensor(y_val)}, 'processed_data/val.pt')
    torch.save({'encodings': test_encodings, 'labels': torch.tensor(y_test)}, 'processed_data/test.pt')
    
    print("Pipeline completed successfully! Tokens saved in 'processed_data/'")

if __name__ == '__main__':
    prepare_pipeline()
