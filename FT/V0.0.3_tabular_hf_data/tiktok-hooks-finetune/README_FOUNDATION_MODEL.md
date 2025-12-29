# Foundation Model Training - TikTok Virality Prediction

## Overview 

This approach fine-tunes a foundational language model (like DistilBERT or RoBERTa) to predict virality using the **actual text content** (hooks + captions) instead of basic tabular features.

## Why This Should Work Better

**Current System (Tabular ML):**
- Uses only text lengths, counts, basic features
- R² = 0.047 (4.7% variance explained) - Very poor
- No semantic understanding of content

**Foundation Model Approach:**
- Uses actual text content (hooks + captions)
- Leverages pre-trained language understanding
- Can learn patterns like:
  - What hooks are engaging?
  - What caption styles work?
  - Semantic patterns in viral content
- Should significantly outperform tabular approach

## Architecture

```
Input: text_hook + caption + metadata
    ↓
Tokenization (BERT/RoBERTa tokenizer)
    ↓
Pre-trained Transformer (DistilBERT/RoBERTa)
    ↓
Fine-tuned for virality prediction
    ↓
Output: Virality score (regression) or Viral tier (classification)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_foundation.txt
```

### 2. Train Model

```bash
python train_foundation_model.py
```

**Configuration options** (edit in script):
- `BASE_MODEL`: Choose model size
  - `'distilbert-base-uncased'` - Fastest, 66M params (recommended to start)
  - `'roberta-base'` - Better performance, 125M params
  - `'bert-base-uncased'` - Standard, 110M params
- `TASK`: `'regression'` or `'classification'`
- `BATCH_SIZE`: Adjust based on GPU memory (16 default)
- `NUM_EPOCHS`: Training epochs (3 default)

### 3. Expected Performance

**Regression:**
- Target: R² > 0.15 (3x improvement over current 0.047)
- Should see better understanding of text patterns

**Classification:**
- Target: Macro F1 > 0.50 (vs current 0.384)
- Better class separation using semantic features

## Model Options

### DistilBERT (Recommended to Start)
- **Pros:** Fast, small (66M params), good for experimentation
- **Cons:** Slightly lower performance than RoBERTa
- **Use when:** Testing approach, limited GPU memory

### RoBERTa
- **Pros:** Better performance, robust
- **Cons:** Larger (125M params), slower
- **Use when:** You want best performance

### BERT
- **Pros:** Standard baseline
- **Cons:** Older architecture
- **Use when:** Comparing with other BERT-based systems

## Training Details

- **Dataset:** ~46K videos from Hugging Face `benxh/tiktok-hooks-finetune`
- **Input:** `text_hook + [SEP] + caption`
- **Split:** 70% train, 15% val, 15% test (time-based)
- **Training:** Full fine-tuning (can switch to LoRA for efficiency)
- **Early stopping:** Stops if validation doesn't improve for 2 epochs

## Output

Model saved to:
```
models_foundation/{model_name}_{task}_{timestamp}/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.txt
└── metadata.json
```

## Memory Requirements

- **DistilBERT:** ~2-3 GB GPU memory (or CPU)
- **RoBERTa:** ~4-5 GB GPU memory
- **CPU training:** Possible but very slow (use DistilBERT)

## Next Steps

1. **Start with DistilBERT** - Fast iteration
2. **Compare with current system** - Should see significant improvement
3. **Experiment with:**
   - Adding metadata to text input
   - Different model sizes
   - LoRA fine-tuning (more efficient)
   - Ensemble with tabular features

## Comparison with Current System

| Aspect | Current (Tabular) | Foundation Model |
|--------|------------------|------------------|
| Features | Text lengths, counts | Actual text content |
| Understanding | None | Semantic understanding |
| R² (Regression) | 0.047 | Target: >0.15 |
| F1 (Classification) | 0.384 | Target: >0.50 |
| Training time | ~30 min | ~1-3 hours (GPU) |
| Inference | Fast (~50ms) | Slower (~200ms) |

## Troubleshooting

**Out of memory:**
- Reduce `BATCH_SIZE` to 8 or 4
- Use `distilbert-base-uncased` instead of RoBERTa
- Enable gradient checkpointing (add to TrainingArguments)

**Slow training:**
- Use GPU if available (automatically detected)
- Reduce `MAX_LENGTH` to 128
- Use smaller model (DistilBERT)

**Poor performance:**
- Try RoBERTa instead of DistilBERT
- Increase `NUM_EPOCHS` to 5
- Add metadata to text input
- Try classification instead of regression

