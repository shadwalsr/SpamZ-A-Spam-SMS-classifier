# Project Report 1: SMS Spam Classification - Data & Pipeline Preparation

**Date**: May 2026  
**Project**: Real-Time SMS Spam Classification System  

## 1. Executive Summary
This report outlines the completion of Phase 1 in building a production-grade, real-time text classification system for SMS spam detection. The goal of this phase was to construct a robust data and tokenization pipeline that prepares raw text data for fine-tuning a DistilBERT model.

## 2. Environment Setup
A dedicated Python virtual environment (`venv`) was established in the project workspace to ensure reproducibility and isolate dependencies. The following core libraries were installed:
- `pandas`, `scikit-learn` (Data manipulation and splitting)
- `matplotlib`, `seaborn` (Data visualization)
- `transformers`, `torch` (Tokenization and tensor formatting)

## 3. Exploratory Data Analysis (EDA)
An EDA script (`eda.py`) was executed on the provided `spam.csv` dataset. Key findings and actions include:
- **Deduplication**: 403 exact duplicate messages were identified and removed, reducing the dataset size from 5,572 to 5,169 unique texts. This prevents data leakage across training and validation sets.
- **Class Distribution**: The dataset exhibits a severe class imbalance, heavily skewed towards legitimate messages ("ham"). Post-deduplication, the distribution stands at 87.3% ham and 12.6% spam.
- **Message Lengths**: Histograms were generated comparing the character lengths of spam vs. ham messages. It was observed that spam messages tend to be noticeably longer on average than legitimate texts, a common pattern due to the inclusion of URLs, promotional codes, and call-to-actions.

## 4. Data Processing & Tokenization Pipeline
The core data preparation pipeline (`pipeline.py`) was constructed and successfully executed:

### 4.1 Stratified Splitting
To ensure the model is evaluated honestly and learns from a representative sample, the dataset was split into an 80/10/10 ratio (Train/Validation/Test). The split was performed using stratification to preserve the 87/13 class imbalance across all sets:
- **Train Set**: 4,135 samples
- **Validation Set**: 517 samples
- **Test Set**: 517 samples

### 4.2 Class Weights Calculation
To combat the severe class imbalance, class weights were calculated for the training set (`ham: 0.57`, `spam: 3.96`). These weights have been saved (`processed_data/class_weights.npy`) and will be applied to the loss function during model training, ensuring the model heavily penalizes false negatives (misclassifying spam as ham).

### 4.3 Tokenization
The text messages were converted into numerical formats required by the DistilBERT model.
- Used HuggingFace's `distilbert-base-uncased` tokenizer.
- Messages were padded and truncated to a uniform length of 128 tokens.
- The resulting `input_ids` and `attention_mask` sequences were packaged alongside their labels into PyTorch dataset structures.
- The tokenized tensors were successfully exported and saved as `train.pt`, `val.pt`, and `test.pt` inside the `processed_data/` directory, ready for the next phase.

## 5. Next Steps (Phase 2)
With the data thoroughly cleaned, visualized, split, and tokenized, the system is fully prepared for Phase 2:
- Define the `DistilBERT` classification model in PyTorch.
- Construct the fine-tuning training loop utilizing the computed class weights.
- Implement early stopping on the validation set.
- Export the trained model to ONNX format with INT8 quantization for optimal inference speed.
