"""
INFERENCE WITH FOUNDATION MODEL - TikTok Virality Prediction
 
Usage:
    python predict_foundation.py --model models_foundation/distilbert-base-uncased_regression_20251210_120000 \
                                 --text_hook "When the interviewer asks..." \
                                 --caption "I fear this would work #corporatehumor"
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import argparse
import glob
import os

def load_model(model_path):
    """Load fine-tuned model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Load metadata
    metadata_path = os.path.join(model_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return model, tokenizer, metadata

def predict(model, tokenizer, text_hook, caption, metadata, device='cpu'):
    """Make prediction on new text"""
    # Combine hook and caption
    text_input = f"{text_hook} [SEP] {caption}"
    
    # Tokenize
    inputs = tokenizer(
        text_input,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        
        if metadata.get('task') == 'regression':
            prediction = outputs.logits.item()
            return {
                'virality_score': prediction,
                'prediction_type': 'regression'
            }
        else:
            # Classification
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            class_probs = probs[0].tolist()
            
            label_classes = metadata.get('label_classes', ['low_viral', 'viral', 'mega_viral'])
            
            return {
                'viral_tier': label_classes[predicted_class],
                'probabilities': {
                    label_classes[i]: class_probs[i] 
                    for i in range(len(label_classes))
                },
                'confidence': max(class_probs),
                'prediction_type': 'classification'
            }

def main():
    parser = argparse.ArgumentParser(description='Predict virality using foundation model')
    parser.add_argument('--model', type=str, help='Path to model directory')
    parser.add_argument('--text_hook', type=str, required=True, help='Video hook text')
    parser.add_argument('--caption', type=str, required=True, help='Video caption')
    parser.add_argument('--find_latest', action='store_true', help='Use latest model if --model not specified')
    
    args = parser.parse_args()
    
    # Find model if needed
    if args.find_latest and not args.model:
        model_dirs = glob.glob('models_foundation/*')
        if model_dirs:
            args.model = max(model_dirs, key=os.path.getmtime)
            print(f"Using latest model: {args.model}")
        else:
            print("Error: No models found. Train a model first with train_foundation_model.py")
            return
    
    if not args.model:
        print("Error: Specify --model or use --find_latest")
        return
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer, metadata = load_model(args.model)
    model.to(device)
    
    print(f"✓ Model loaded on {device}")
    print(f"  Task: {metadata.get('task', 'unknown')}")
    print(f"  Base model: {metadata.get('base_model', 'unknown')}")
    
    # Make prediction
    result = predict(model, tokenizer, args.text_hook, args.caption, metadata, device)
    
    # Print results
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80)

if __name__ == "__main__":
    main()

