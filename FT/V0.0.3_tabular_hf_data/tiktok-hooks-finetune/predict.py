"""
INFERENCE SCRIPT - TikTok Virality Prediction

Purpose:
    Makes predictions on new, unseen TikTok videos using trained models.
    Processes raw video metadata and returns virality predictions with confidence scores.

Usage:
    python predict.py --input video_metadata.json
    python predict.py --interactive
    
Input Format (JSON):
    {
        "text_hook": "When the interviewer asks if you have a background in finance",
        "caption": "I fear this would work on me as an interviewer #corporatehumor",
        "main_category": "Business",
        "subcategory": "HR & Payroll",
        "length": 7.0,
        "upload_dayofweek": 2,  # Optional: 0=Monday, 6=Sunday (default: today)
        "upload_month": 12,     # Optional: 1-12 (default: current month)
        "is_weekend": 0         # Optional: 0 or 1 (default: auto-detect from dayofweek)
    }

Output Format (JSON):
    {
        "virality_score": 0.4235,
        "viral_tier": "viral",
        "tier_confidence": 0.68,
        "predictions": {
            "regression": {
                "score": 0.4235,
                "interpretation": "moderate viral potential"
            },
            "classification": {
                "tier": "viral",
                "probabilities": {
                    "low_viral": 0.15,
                    "viral": 0.68,
                    "mega_viral": 0.17
                }
            }
        },
        "feature_contributions": {
            "top_5_features": [...]
        }
    }

Dependencies:
    - Trained models in models/regression/ and models/classification/
    - Preprocessing artifacts in artifacts/
    - Feature metadata in artifacts/
"""

import pandas as pd
import numpy as np
import pickle
import json
import argparse
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model paths (will use best models from training)
DEFAULT_REGRESSION_MODEL = 'models/regression/xgboost/model.pkl'
DEFAULT_CLASSIFICATION_MODEL = 'models/classification/xgboost/model.pkl'

# ============================================================================
# PREDICTOR CLASS
# ============================================================================

class ViralityPredictor:
    """
    Main predictor class that handles:
    - Loading trained models and preprocessing artifacts
    - Feature engineering for new data
    - Making predictions
    - Formatting output with confidence scores
    """
    
    def __init__(self, regression_model_path=None, classification_model_path=None, 
                 artifacts_dir='artifacts'):
        """
        Initialize predictor by loading models and preprocessing objects
        
        Args:
            regression_model_path: Path to regression model .pkl file
            classification_model_path: Path to classification model .pkl file
            artifacts_dir: Directory containing scaler, encoders, metadata
        """
        print("="*80)
        print("INITIALIZING VIRALITY PREDICTOR")
        print("="*80)
        
        # Find latest artifacts if not specified
        self.artifacts_dir = artifacts_dir
        metadata_files = glob.glob(f'{artifacts_dir}/feature_metadata_*.json')
        
        if not metadata_files:
            raise FileNotFoundError(f"No metadata found in {artifacts_dir}/")
        
        latest_metadata = max(metadata_files)
        # Extract timestamp: artifacts/feature_metadata_YYYYMMDD_HHMMSS.json -> YYYYMMDD_HHMMSS
        timestamp = latest_metadata.replace(f'{artifacts_dir}/feature_metadata_', '').replace('.json', '')
        
        print(f"\nLoading artifacts with timestamp: {timestamp}")
        
        # Load metadata
        with open(f'{artifacts_dir}/feature_metadata_{timestamp}.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load scaler
        with open(f'{artifacts_dir}/scaler_{timestamp}.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load encoders
        with open(f'{artifacts_dir}/encoders_{timestamp}.pkl', 'rb') as f:
            self.encoders = pickle.load(f)
        
        self.label_encoder = self.encoders['label_encoder']
        self.subcategory_encoder = self.encoders['subcategory_encoder']
        self.main_category_columns = self.encoders['main_category_columns']
        
        print(f"✓ Loaded preprocessing artifacts")
        print(f"  Features: {len(self.metadata['features']['all_features'])}")
        print(f"  Transform: {self.metadata['transformations']['target_transform']}")
        
        # Load models
        reg_path = regression_model_path or DEFAULT_REGRESSION_MODEL
        clf_path = classification_model_path or DEFAULT_CLASSIFICATION_MODEL
        
        with open(reg_path, 'rb') as f:
            self.regression_model = pickle.load(f)
        print(f"✓ Loaded regression model: {reg_path}")
        
        with open(clf_path, 'rb') as f:
            self.classification_model = pickle.load(f)
        print(f"✓ Loaded classification model: {clf_path}")
        
        # Extract thresholds
        self.thresholds = self.metadata['viral_tier_thresholds']
        
        print(f"\n✓ Predictor ready!")
        print(f"  Viral tier thresholds:")
        print(f"    low_viral:  < {self.thresholds['low_threshold']:.4f}")
        print(f"    mega_viral: ≥ {self.thresholds['mega_threshold']:.4f}")
    
    def engineer_features(self, video_data):
        """
        Engineer features from raw video metadata
        
        Args:
            video_data: Dict with video metadata
            
        Returns:
            DataFrame with engineered features matching training format
        """
        # Create DataFrame from input
        df = pd.DataFrame([video_data])
        
        # Text features
        df['hook_length_chars'] = df['text_hook'].astype(str).str.len()
        df['hook_length_words'] = df['text_hook'].astype(str).str.split().str.len()
        df['caption_length_chars'] = df['caption'].astype(str).str.len()
        df['caption_length_words'] = df['caption'].astype(str).str.split().str.len()
        df['hashtag_count'] = df['caption'].astype(str).str.count(r'#')
        df['mention_count'] = df['caption'].astype(str).str.count(r'@')
        
        # Temporal features (use defaults if not provided)
        if 'upload_dayofweek' not in df.columns:
            df['upload_dayofweek'] = datetime.now().weekday()
        if 'upload_month' not in df.columns:
            df['upload_month'] = datetime.now().month
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = int(df['upload_dayofweek'].iloc[0] in [5, 6])
        
        # Target encode subcategory
        subcategory = df['subcategory'].iloc[0]
        global_mean = np.mean(list(self.subcategory_encoder.values()))
        df['subcategory_encoded'] = self.subcategory_encoder.get(subcategory, global_mean)
        
        # One-hot encode main_category
        # Create all category columns, set to 0
        for col in self.main_category_columns:
            df[col] = 0
        
        # Set the appropriate category to 1
        main_cat = df['main_category'].iloc[0]
        cat_col = f'cat_{main_cat}'
        if cat_col in self.main_category_columns:
            df[cat_col] = 1
        
        # Get feature list from metadata
        feature_names = self.metadata['features']['all_features']
        
        # Ensure all features exist
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0  # Default to 0 for missing features
        
        # Select only required features in correct order
        df_features = df[feature_names]
        
        # Apply scaling to numeric features
        numeric_features = self.metadata['features']['numeric_features']
        df_features[numeric_features] = self.scaler.transform(df_features[numeric_features])
        
        return df_features
    
    def predict(self, video_data, return_probabilities=True):
        """
        Make predictions for a new video
        
        Args:
            video_data: Dict with video metadata
            return_probabilities: Include class probabilities in output
            
        Returns:
            Dict with predictions and confidence scores
        """
        # Engineer features
        X = self.engineer_features(video_data)
        
        # Regression prediction (on log scale)
        virality_score_log = self.regression_model.predict(X)[0]
        
        # Inverse transform to original scale
        virality_score = np.expm1(virality_score_log)
        
        # Classification prediction
        viral_tier_encoded = self.classification_model.predict(X)[0]
        viral_tier = self.label_encoder.inverse_transform([viral_tier_encoded])[0]
        
        # Get class probabilities if supported
        tier_probs = {}
        tier_confidence = 0.0
        
        if hasattr(self.classification_model, 'predict_proba'):
            proba = self.classification_model.predict_proba(X)[0]
            tier_confidence = float(proba.max())
            
            if return_probabilities:
                for idx, label in enumerate(self.label_encoder.classes_):
                    tier_probs[label] = float(proba[idx])
        
        # Determine interpretation
        if virality_score < self.thresholds['low_threshold']:
            interpretation = "low viral potential"
        elif virality_score >= self.thresholds['mega_threshold']:
            interpretation = "mega viral potential"
        else:
            interpretation = "moderate viral potential"
        
        # Get feature importance if available
        feature_contributions = {}
        if hasattr(self.regression_model, 'feature_importances_'):
            feature_names = self.metadata['features']['all_features']
            importances = self.regression_model.feature_importances_
            
            # Get top 5 features with their values
            top_indices = np.argsort(importances)[-5:][::-1]
            feature_contributions['top_5_features'] = [
                {
                    'feature': feature_names[i],
                    'importance': float(importances[i]),
                    'value': float(X.iloc[0, i])
                }
                for i in top_indices
            ]
        
        # Build response
        result = {
            'virality_score': float(virality_score),
            'viral_tier': viral_tier,
            'tier_confidence': tier_confidence,
            'predictions': {
                'regression': {
                    'score': float(virality_score),
                    'log_score': float(virality_score_log),
                    'interpretation': interpretation
                },
                'classification': {
                    'tier': viral_tier,
                    'tier_encoded': int(viral_tier_encoded),
                    'confidence': tier_confidence,
                }
            },
            'thresholds': {
                'low_threshold': self.thresholds['low_threshold'],
                'mega_threshold': self.thresholds['mega_threshold']
            }
        }
        
        if tier_probs:
            result['predictions']['classification']['probabilities'] = tier_probs
        
        if feature_contributions:
            result['feature_contributions'] = feature_contributions
        
        return result
    
    def predict_batch(self, video_list):
        """
        Make predictions for multiple videos
        
        Args:
            video_list: List of video metadata dicts
            
        Returns:
            List of prediction results
        """
        return [self.predict(video) for video in video_list]

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def interactive_mode():
    """Interactive mode for testing predictions"""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nEnter video metadata (or 'quit' to exit)")
    
    predictor = ViralityPredictor()
    
    while True:
        print("\n" + "-"*80)
        
        # Get input
        text_hook = input("Text Hook: ").strip()
        if text_hook.lower() == 'quit':
            break
        
        caption = input("Caption: ").strip()
        main_category = input("Main Category: ").strip()
        subcategory = input("Subcategory: ").strip()
        length = float(input("Video Length (seconds): ").strip())
        
        # Optional temporal features
        use_defaults = input("Use default temporal features? (y/n): ").strip().lower()
        
        video_data = {
            'text_hook': text_hook,
            'caption': caption,
            'main_category': main_category,
            'subcategory': subcategory,
            'length': length,
        }
        
        if use_defaults != 'y':
            video_data['upload_dayofweek'] = int(input("Upload Day of Week (0=Mon, 6=Sun): ").strip())
            video_data['upload_month'] = int(input("Upload Month (1-12): ").strip())
            video_data['is_weekend'] = int(input("Is Weekend? (0/1): ").strip())
        
        # Make prediction
        print("\n🔮 Predicting...")
        result = predictor.predict(video_data)
        
        # Display results
        print("\n" + "="*80)
        print("PREDICTION RESULTS")
        print("="*80)
        
        print(f"\n📊 Virality Score: {result['virality_score']:.4f}")
        print(f"   Interpretation: {result['predictions']['regression']['interpretation']}")
        
        print(f"\n🎯 Viral Tier: {result['viral_tier']}")
        print(f"   Confidence: {result['tier_confidence']*100:.1f}%")
        
        if 'probabilities' in result['predictions']['classification']:
            print(f"\n📈 Tier Probabilities:")
            for tier, prob in result['predictions']['classification']['probabilities'].items():
                print(f"   {tier:12s}: {prob*100:5.1f}%")
        
        if 'top_5_features' in result.get('feature_contributions', {}):
            print(f"\n🔑 Top Contributing Features:")
            for feat_info in result['feature_contributions']['top_5_features']:
                print(f"   {feat_info['feature']:30s}: importance={feat_info['importance']:.4f}")

def batch_mode(input_file, output_file=None):
    """Process multiple videos from JSON file"""
    print("\n" + "="*80)
    print("BATCH MODE")
    print("="*80)
    
    # Load input
    with open(input_file, 'r') as f:
        video_list = json.load(f)
    
    if not isinstance(video_list, list):
        video_list = [video_list]
    
    print(f"\nProcessing {len(video_list)} videos from {input_file}")
    
    # Initialize predictor
    predictor = ViralityPredictor()
    
    # Make predictions
    results = predictor.predict_batch(video_list)
    
    # Save or print results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved predictions to {output_file}")
    else:
        print("\n" + "="*80)
        print("PREDICTIONS")
        print("="*80)
        for i, result in enumerate(results, 1):
            print(f"\nVideo {i}:")
            print(f"  Virality Score: {result['virality_score']:.4f}")
            print(f"  Viral Tier: {result['viral_tier']} (confidence: {result['tier_confidence']*100:.1f}%)")
    
    return results

def single_prediction(video_data_str):
    """Make prediction for a single video from JSON string"""
    video_data = json.loads(video_data_str)
    
    predictor = ViralityPredictor()
    result = predictor.predict(video_data)
    
    print("\n" + "="*80)
    print("PREDICTION RESULT")
    print("="*80)
    print(json.dumps(result, indent=2))
    
    return result

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def run_example():
    """Run example prediction with sample data"""
    print("\n" + "="*80)
    print("EXAMPLE PREDICTION")
    print("="*80)
    
    # Sample video
    sample_video = {
        'text_hook': 'When the interviewer asks if you have a background in finance',
        'caption': 'I fear this would work on me as an interviewer #corporatehumor',
        'main_category': 'Business',
        'subcategory': 'HR & Payroll',
        'length': 7.0,
        'upload_dayofweek': 2,  # Wednesday
        'upload_month': 12,      # December
        'is_weekend': 0
    }
    
    print("\nSample Video Metadata:")
    print(json.dumps(sample_video, indent=2))
    
    predictor = ViralityPredictor()
    result = predictor.predict(sample_video)
    
    print("\n" + "="*80)
    print("PREDICTION")
    print("="*80)
    print(json.dumps(result, indent=2))
    
    return result

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='TikTok Virality Prediction')
    parser.add_argument('--input', type=str, help='Input JSON file with video metadata')
    parser.add_argument('--output', type=str, help='Output JSON file for predictions')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--example', action='store_true', help='Run example prediction')
    parser.add_argument('--json', type=str, help='Single video JSON string')
    parser.add_argument('--regression-model', type=str, help='Path to regression model')
    parser.add_argument('--classification-model', type=str, help='Path to classification model')
    
    args = parser.parse_args()
    
    if args.example:
        run_example()
    elif args.interactive:
        interactive_mode()
    elif args.json:
        single_prediction(args.json)
    elif args.input:
        batch_mode(args.input, args.output)
    else:
        # Default: run example
        print("No arguments provided. Running example prediction...")
        print("Use --help for usage options")
        run_example()

if __name__ == "__main__":
    main()