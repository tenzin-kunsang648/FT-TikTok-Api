"""
FOUNDATION MODEL TRAINING - TikTok Virality Prediction

Purpose:
    Fine-tune a foundational language model (BERT/RoBERTa) to predict virality
    using the actual text content (hooks + captions) instead of basic tabular features.
    
Architecture:
    - Base model: Pre-trained transformer (DistilBERT/RoBERTa)
    - Input: text_hook + caption + metadata
    - Output: Virality score (regression) or viral tier (classification)
    - Fine-tuning: Full model fine-tuning or LoRA for efficiency

Why This Should Work Better:
    - Current system uses only text lengths/counts (R² = 0.047)
    - Foundation models understand semantic meaning, patterns, hooks
    - Can learn what makes content viral from actual text
    - Leverages pre-trained knowledge about language patterns

Dataset:
    - Hugging Face: benxh/tiktok-hooks-finetune
    - ~46K videos with text_hook, caption, engagement metrics
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
from torch.utils.data import Dataset as TorchDataset
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model choices (start with smaller/faster models)
MODEL_OPTIONS = {
    'distilbert': 'distilbert-base-uncased',  # Fast, 66M params
    'roberta': 'roberta-base',                 # Better performance, 125M params
    'bert': 'bert-base-uncased',              # Standard, 110M params
}

BASE_MODEL = MODEL_OPTIONS['distilbert']  # Start with fastest
MAX_LENGTH = 256  # Token limit (hook + caption)
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
RANDOM_SEED = 42

# Task: 'regression' or 'classification'
TASK = 'regression'  # Start with regression, can switch to classification

print("="*100)
print("FOUNDATION MODEL TRAINING - TikTok Virality Prediction")
print("="*100)
print(f"\nConfiguration:")
print(f"  Base model: {BASE_MODEL}")
print(f"  Task: {TASK}")
print(f"  Max length: {MAX_LENGTH}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("\n" + "="*100)
print("LOADING AND PREPARING DATA")
print("="*100)

# Load dataset
print("\nLoading dataset from Hugging Face...")
ds = load_dataset("benxh/tiktok-hooks-finetune")
df = ds['train'].to_pandas()

print(f"✓ Loaded {len(df):,} videos")

# Calculate virality score (same as current system)
df['engagement_rate'] = (df['likes'] + df['comments'] + df['shares']) / (df['views'] + 1)
df['virality_score'] = df['engagement_rate'] * np.log1p(df['views'])

# Create viral tiers for classification
low_threshold = df['virality_score'].quantile(0.20)
mega_threshold = df['virality_score'].quantile(0.95)
df['viral_tier'] = pd.cut(
    df['virality_score'],
    bins=[-float('inf'), low_threshold, mega_threshold, float('inf')],
    labels=['low_viral', 'viral', 'mega_viral']
)

print(f"\n✓ Virality score calculated")
print(f"  Range: {df['virality_score'].min():.4f} to {df['virality_score'].max():.4f}")
print(f"  Mean: {df['virality_score'].mean():.4f}")

if TASK == 'classification':
    tier_dist = df['viral_tier'].value_counts()
    print(f"\n✓ Viral tiers:")
    for tier, count in tier_dist.items():
        print(f"  {tier}: {count:,} ({count/len(df)*100:.1f}%)")

# Prepare text inputs: Combine hook + caption
print("\nPreparing text inputs...")
df['text_input'] = df['text_hook'].astype(str) + " [SEP] " + df['caption'].astype(str)

# Optional: Add metadata as text
# df['text_input'] = df['text_hook'].astype(str) + " [SEP] " + df['caption'].astype(str) + \
#                    f" Category: {df['main_category']} Length: {df['length']}s"

# Remove rows with missing text or targets
before = len(df)
df = df.dropna(subset=['text_input', 'virality_score'])
if TASK == 'classification':
    df = df.dropna(subset=['viral_tier'])
dropped = before - len(df)
if dropped > 0:
    print(f"  Dropped {dropped:,} rows with missing data")

print(f"✓ Final dataset: {len(df):,} videos")

# Time-based split (same as current system)
df['uploaded_at'] = pd.to_datetime(df['uploaded_at'])
df = df.sort_values('uploaded_at')

train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)

train_df = df.iloc[:train_size].copy()
val_df = df.iloc[train_size:train_size+val_size].copy()
test_df = df.iloc[train_size+val_size:].copy()

print(f"\n✓ Data splits:")
print(f"  Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

# ============================================================================
# TOKENIZATION
# ============================================================================

print("\n" + "="*100)
print("TOKENIZATION")
print("="*100)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Add special token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✓ Loaded tokenizer: {BASE_MODEL}")

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(
        examples['text_input'],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH
    )

print("\nTokenizing datasets...")
train_dataset = Dataset.from_pandas(train_df[['text_input', 'virality_score', 'viral_tier']])
val_dataset = Dataset.from_pandas(val_df[['text_input', 'virality_score', 'viral_tier']])
test_dataset = Dataset.from_pandas(test_df[['text_input', 'virality_score', 'viral_tier']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

print("✓ Tokenization complete")

# Prepare labels
if TASK == 'regression':
    train_dataset = train_dataset.map(lambda x: {'labels': float(x['virality_score'])}, remove_columns=['virality_score', 'viral_tier'])
    val_dataset = val_dataset.map(lambda x: {'labels': float(x['virality_score'])}, remove_columns=['virality_score', 'viral_tier'])
    test_dataset = test_dataset.map(lambda x: {'labels': float(x['virality_score'])}, remove_columns=['virality_score', 'viral_tier'])
    num_labels = 1
else:
    # Classification: encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['viral_tier'].astype(str))
    val_labels = label_encoder.transform(val_df['viral_tier'].astype(str))
    test_labels = label_encoder.transform(test_df['viral_tier'].astype(str))
    
    train_dataset = train_dataset.map(lambda x, idx: {'labels': int(train_labels[idx])}, with_indices=True, remove_columns=['virality_score', 'viral_tier'])
    val_dataset = val_dataset.map(lambda x, idx: {'labels': int(val_labels[idx])}, with_indices=True, remove_columns=['virality_score', 'viral_tier'])
    test_dataset = test_dataset.map(lambda x, idx: {'labels': int(test_labels[idx])}, with_indices=True, remove_columns=['virality_score', 'viral_tier'])
    num_labels = len(label_encoder.classes_)
    
    print(f"\n✓ Label encoding: {label_encoder.classes_}")

# ============================================================================
# MODEL SETUP
# ============================================================================

print("\n" + "="*100)
print("MODEL SETUP")
print("="*100)

# Load model
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

print(f"✓ Loaded model: {BASE_MODEL}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    if TASK == 'regression':
        mae = mean_absolute_error(labels, predictions)
        rmse = np.sqrt(mean_squared_error(labels, predictions))
        r2 = r2_score(labels, predictions)
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    else:
        predictions = np.argmax(predictions, axis=1)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        accuracy = (predictions == labels).mean()
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*100)
print("TRAINING")
print("="*100)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"models_foundation/{BASE_MODEL.replace('/', '_')}_{TASK}_{timestamp}"

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
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
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
print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"  Output: {output_dir}")

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

# Save tokenizer
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

print(f"\n✓ Model and metadata saved to {output_dir}")
print("\n" + "="*100)
print("TRAINING COMPLETE")
print("="*100)

