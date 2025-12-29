"""
FOUNDATION MODEL TRAINING - Google Colab / Kaggle Version

This version is optimized for cloud training platforms:
- Google Colab (free GPU)
- Kaggle (free GPU)
- Any cloud GPU instance

Usage in Colab:
    1. Upload this file to Colab
    2. Run: !pip install transformers datasets accelerate scikit-learn
    3. Run this script

Usage in Kaggle:
    1. Create new notebook
    2. Add this as code cell
    3. Enable GPU in settings
"""

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model choices
MODEL_OPTIONS = {
    'distilbert': 'distilbert-base-uncased',  # Fast, 66M params
    'roberta': 'roberta-base',                 # Better performance, 125M params
    'bert': 'bert-base-uncased',              # Standard, 110M params
}

BASE_MODEL = MODEL_OPTIONS['distilbert']  # Start with fastest
MAX_LENGTH = 256
BATCH_SIZE = 16  # Adjust based on GPU memory
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
RANDOM_SEED = 42
TASK = 'regression'  # 'regression' or 'classification'

# Output directory (Colab/Kaggle)
OUTPUT_DIR = f"/content/models_foundation"  # Colab
# OUTPUT_DIR = f"/kaggle/working/models_foundation"  # Kaggle

print("="*100)
print("FOUNDATION MODEL TRAINING - Cloud Version")
print("="*100)
print(f"\nConfiguration:")
print(f"  Base model: {BASE_MODEL}")
print(f"  Task: {TASK}")
print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  Output: {OUTPUT_DIR}")

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("\n" + "="*100)
print("LOADING AND PREPARING DATA")
print("="*100)

print("\nLoading dataset from Hugging Face...")
ds = load_dataset("benxh/tiktok-hooks-finetune")
df = ds['train'].to_pandas()

print(f"✓ Loaded {len(df):,} videos")

# Calculate virality score
df['engagement_rate'] = (df['likes'] + df['comments'] + df['shares']) / (df['views'] + 1)
df['virality_score'] = df['engagement_rate'] * np.log1p(df['views'])

# Create viral tiers for classification
if TASK == 'classification':
    low_threshold = df['virality_score'].quantile(0.20)
    mega_threshold = df['virality_score'].quantile(0.95)
    df['viral_tier'] = pd.cut(
        df['virality_score'],
        bins=[-float('inf'), low_threshold, mega_threshold, float('inf')],
        labels=['low_viral', 'viral', 'mega_viral']
    )

# Prepare text inputs
df['text_input'] = df['text_hook'].astype(str) + " [SEP] " + df['caption'].astype(str)

# Remove missing
before = len(df)
df = df.dropna(subset=['text_input', 'virality_score'])
if TASK == 'classification':
    df = df.dropna(subset=['viral_tier'])
print(f"✓ Final dataset: {len(df):,} videos (dropped {before - len(df):,})")

# Time-based split
df['uploaded_at'] = pd.to_datetime(df['uploaded_at'])
df = df.sort_values('uploaded_at')

train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)

train_df = df.iloc[:train_size].copy()
val_df = df.iloc[train_size:train_size+val_size].copy()
test_df = df.iloc[train_size+val_size:].copy()

print(f"\n✓ Data splits: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

# ============================================================================
# TOKENIZATION
# ============================================================================

print("\n" + "="*100)
print("TOKENIZATION")
print("="*100)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples['text_input'],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH
    )

print("Tokenizing datasets...")
train_dataset = Dataset.from_pandas(train_df[['text_input', 'virality_score', 'viral_tier'] if TASK == 'classification' else ['text_input', 'virality_score']])
val_dataset = Dataset.from_pandas(val_df[['text_input', 'virality_score', 'viral_tier'] if TASK == 'classification' else ['text_input', 'virality_score']])
test_dataset = Dataset.from_pandas(test_df[['text_input', 'virality_score', 'viral_tier'] if TASK == 'classification' else ['text_input', 'virality_score']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Prepare labels
if TASK == 'regression':
    train_dataset = train_dataset.map(lambda x: {'labels': float(x['virality_score'])}, remove_columns=['virality_score', 'viral_tier'] if 'viral_tier' in train_dataset.column_names else ['virality_score'])
    val_dataset = val_dataset.map(lambda x: {'labels': float(x['virality_score'])}, remove_columns=['virality_score', 'viral_tier'] if 'viral_tier' in val_dataset.column_names else ['virality_score'])
    test_dataset = test_dataset.map(lambda x: {'labels': float(x['virality_score'])}, remove_columns=['virality_score', 'viral_tier'] if 'viral_tier' in test_dataset.column_names else ['virality_score'])
    num_labels = 1
else:
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['viral_tier'].astype(str))
    val_labels = label_encoder.transform(val_df['viral_tier'].astype(str))
    test_labels = label_encoder.transform(test_df['viral_tier'].astype(str))
    
    train_dataset = train_dataset.map(lambda x, idx: {'labels': int(train_labels[idx])}, with_indices=True, remove_columns=['virality_score', 'viral_tier'])
    val_dataset = val_dataset.map(lambda x, idx: {'labels': int(val_labels[idx])}, with_indices=True, remove_columns=['virality_score', 'viral_tier'])
    test_dataset = test_dataset.map(lambda x, idx: {'labels': int(test_labels[idx])}, with_indices=True, remove_columns=['virality_score', 'viral_tier'])
    num_labels = len(label_encoder.classes_)

print("✓ Tokenization complete")

# ============================================================================
# MODEL SETUP
# ============================================================================

print("\n" + "="*100)
print("MODEL SETUP")
print("="*100)

if TASK == 'regression':
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=1,
        problem_type="regression"
    )
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels
    )

print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    if TASK == 'regression':
        mae = mean_absolute_error(labels, predictions)
        rmse = np.sqrt(mean_squared_error(labels, predictions))
        r2 = r2_score(labels, predictions)
        return {'mae': mae, 'rmse': rmse, 'r2': r2}
    else:
        predictions = np.argmax(predictions, axis=1)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        accuracy = (predictions == labels).mean()
        return {'accuracy': accuracy, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted}

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*100)
print("TRAINING")
print("="*100)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"{OUTPUT_DIR}/{BASE_MODEL.replace('/', '_')}_{TASK}_{timestamp}"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir=f'{output_dir}/logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="mae" if TASK == 'regression' else "f1_macro",
    greater_is_better=False if TASK == 'regression' else True,
    save_total_limit=2,
    seed=RANDOM_SEED,
    fp16=torch.cuda.is_available(),
    report_to="none",  # Disable wandb/tensorboard in Colab
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print(f"\nStarting training...")
trainer.train()

print("\n✓ Training complete!")

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "="*100)
print("EVALUATION ON TEST SET")
print("="*100)

test_results = trainer.evaluate(test_dataset)
print(f"\nTest set results:")
for key, value in test_results.items():
    if 'eval_' in key:
        print(f"  {key}: {value:.4f}")

# Save model
print(f"\n✓ Saving model to {output_dir}")
trainer.save_model()
tokenizer.save_pretrained(output_dir)

# Save metadata
metadata = {
    'base_model': BASE_MODEL,
    'task': TASK,
    'max_length': MAX_LENGTH,
    'timestamp': timestamp,
    'test_results': {k: float(v) for k, v in test_results.items()},
    'num_train': len(train_df),
    'num_val': len(val_df),
    'num_test': len(test_df),
}

if TASK == 'classification':
    metadata['label_classes'] = label_encoder.classes_.tolist()

with open(f'{output_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Model saved to {output_dir}")
print(f"\nTo download model from Colab:")
print(f"  from google.colab import files")
print(f"  files.download('{output_dir}/pytorch_model.bin')")
print(f"  files.download('{output_dir}/config.json')")
print(f"  files.download('{output_dir}/tokenizer_config.json')")
print(f"  files.download('{output_dir}/vocab.txt')")
print(f"  files.download('{output_dir}/metadata.json')")

print("\n" + "="*100)
print("TRAINING COMPLETE")
print("="*100)

