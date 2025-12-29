"""
AWS Lambda Handler - TikTok Virality Prediction

This is the ENTRY POINT for AWS Lambda. When Lambda is invoked, AWS calls the
lambda_handler() function and passes two arguments:

1. event: Input data (from API Gateway, direct invoke, S3 trigger, etc.)
2. context: Runtime information (request ID, memory limit, time remaining)

Key Lambda Concepts Demonstrated Here:
- Global scope initialization (loaded once per container, reused across invocations)
- Proper error handling and response formatting
- JSON parsing and validation
- CloudWatch logging
- Cold start optimization
"""

import json
import numpy as np
import pandas as pd
import pickle
import os
import traceback
from datetime import datetime

# ============================================================================
# GLOBAL SCOPE - Initialized Once Per Container (Cold Start Optimization)
# ============================================================================

"""
Why initialize here instead of inside lambda_handler?

Lambda containers are reused across invocations. By loading models in
global scope, they're loaded ONCE when the container starts, then reused for
subsequent requests. This makes the first request slower (cold start ~2s) but
all following requests fast (~50ms).

This is THE most important Lambda optimization technique!
"""

print("Lambda container initializing...")
print(f"Initialization timestamp: {datetime.now()}")

# Find artifacts (Lambda uploads everything to /var/task/)
LAMBDA_TASK_ROOT = os.environ.get('LAMBDA_TASK_ROOT', '.')
ARTIFACTS_DIR = os.path.join(LAMBDA_TASK_ROOT, 'artifacts')
MODELS_DIR = os.path.join(LAMBDA_TASK_ROOT, 'models')

# Load preprocessing artifacts
print(f"Loading artifacts from {ARTIFACTS_DIR}")

# Find latest metadata file
import glob
metadata_files = glob.glob(os.path.join(ARTIFACTS_DIR, 'feature_metadata_*.json'))
if not metadata_files:
    raise FileNotFoundError(f"No metadata found in {ARTIFACTS_DIR}")

latest_metadata = max(metadata_files)
timestamp = latest_metadata.replace(f'{ARTIFACTS_DIR}/feature_metadata_', '').replace('.json', '')

print(f"Using timestamp: {timestamp}")

# Load metadata
with open(os.path.join(ARTIFACTS_DIR, f'feature_metadata_{timestamp}.json'), 'r') as f:
    METADATA = json.load(f)

# Load scaler
with open(os.path.join(ARTIFACTS_DIR, f'scaler_{timestamp}.pkl'), 'rb') as f:
    SCALER = pickle.load(f)

# Load encoders
with open(os.path.join(ARTIFACTS_DIR, f'encoders_{timestamp}.pkl'), 'rb') as f:
    ENCODERS = pickle.load(f)

LABEL_ENCODER = ENCODERS['label_encoder']
SUBCATEGORY_ENCODER = ENCODERS['subcategory_encoder']
MAIN_CATEGORY_COLUMNS = ENCODERS['main_category_columns']

print(f"✓ Loaded preprocessing artifacts")

# Load trained models
REGRESSION_MODEL_PATH = os.path.join(MODELS_DIR, 'regression/xgboost/model.pkl')
CLASSIFICATION_MODEL_PATH = os.path.join(MODELS_DIR, 'classification/xgboost/model.pkl')

with open(REGRESSION_MODEL_PATH, 'rb') as f:
    REGRESSION_MODEL = pickle.load(f)
print(f"✓ Loaded regression model")

with open(CLASSIFICATION_MODEL_PATH, 'rb') as f:
    CLASSIFICATION_MODEL = pickle.load(f)
print(f"✓ Loaded classification model")

# Extract configuration
FEATURE_NAMES = METADATA['features']['all_features']
NUMERIC_FEATURES = METADATA['features']['numeric_features']
THRESHOLDS = METADATA['viral_tier_thresholds']

print(f"✓ Lambda initialization complete! Ready for requests.")
print(f"  Features: {len(FEATURE_NAMES)}")
print(f"  Models loaded and cached in memory")

# ============================================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================================

def engineer_features(video_data):
    """
    Transform raw video metadata into ML-ready features
    
    This replicates the feature engineering from training pipeline.
    Must match EXACTLY or predictions will be wrong!
    
    Args:
        video_data: Dict with video metadata
        
    Returns:
        DataFrame with engineered features ready for model input
    """
    df = pd.DataFrame([video_data])
    
    # Text features - extract from hook and caption
    df['hook_length_chars'] = df['text_hook'].astype(str).str.len()
    df['hook_length_words'] = df['text_hook'].astype(str).str.split().str.len()
    df['caption_length_chars'] = df['caption'].astype(str).str.len()
    df['caption_length_words'] = df['caption'].astype(str).str.split().str.len()
    df['hashtag_count'] = df['caption'].astype(str).str.count(r'#')
    df['mention_count'] = df['caption'].astype(str).str.count(r'@')
    
    # Temporal features - use defaults if not provided
    if 'upload_dayofweek' not in df.columns:
        df['upload_dayofweek'] = datetime.now().weekday()  # 0=Monday, 6=Sunday
    if 'upload_month' not in df.columns:
        df['upload_month'] = datetime.now().month
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = int(df['upload_dayofweek'].iloc[0] in [5, 6])
    
    # Encode subcategory using saved mapping
    subcategory = df['subcategory'].iloc[0] if 'subcategory' in df.columns else None
    global_mean = np.mean(list(SUBCATEGORY_ENCODER.values()))
    df['subcategory_encoded'] = SUBCATEGORY_ENCODER.get(subcategory, global_mean)
    
    # One-hot encode main_category
    for col in MAIN_CATEGORY_COLUMNS:
        df[col] = 0
    
    main_cat = df['main_category'].iloc[0] if 'main_category' in df.columns else None
    cat_col = f'cat_{main_cat}'
    if cat_col in MAIN_CATEGORY_COLUMNS:
        df[cat_col] = 1
    
    # Ensure all features exist
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            df[feat] = 0
    
    # Select features in correct order
    df_features = df[FEATURE_NAMES].copy()
    
    # Apply feature scaling (same scaler from training)
    df_features[NUMERIC_FEATURES] = SCALER.transform(df_features[NUMERIC_FEATURES])
    
    return df_features

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_prediction(video_data):
    """
    Make virality predictions for a video
    
    Args:
        video_data: Dict with video metadata
        
    Returns:
        Dict with predictions and confidence scores
    """
    # Engineer features
    X = engineer_features(video_data)
    
    # Regression: Predict virality score (on log scale)
    virality_score_log = REGRESSION_MODEL.predict(X)[0]
    
    # Inverse transform to original scale
    # We trained on log(1 + score), so inverse is exp(pred) - 1
    virality_score = float(np.expm1(virality_score_log))
    
    # Classification: Predict viral tier
    viral_tier_encoded = CLASSIFICATION_MODEL.predict(X)[0]
    viral_tier = LABEL_ENCODER.inverse_transform([viral_tier_encoded])[0]
    
    # Get class probabilities
    tier_probs = {}
    tier_confidence = 0.0
    
    if hasattr(CLASSIFICATION_MODEL, 'predict_proba'):
        proba = CLASSIFICATION_MODEL.predict_proba(X)[0]
        tier_confidence = float(proba.max())
        
        for idx, label in enumerate(LABEL_ENCODER.classes_):
            tier_probs[label] = float(proba[idx])
    
    # Determine interpretation based on thresholds
    if virality_score < THRESHOLDS['low_threshold']:
        interpretation = "low viral potential"
    elif virality_score >= THRESHOLDS['mega_threshold']:
        interpretation = "mega viral potential"
    else:
        interpretation = "moderate viral potential"
    
    # Build response
    return {
        'virality_score': virality_score,
        'viral_tier': viral_tier,
        'tier_confidence': tier_confidence,
        'predictions': {
            'regression': {
                'score': virality_score,
                'interpretation': interpretation
            },
            'classification': {
                'tier': viral_tier,
                'confidence': tier_confidence,
                'probabilities': tier_probs
            }
        },
        'metadata': {
            'model_timestamp': timestamp,
            'prediction_timestamp': datetime.now().isoformat()
        }
    }

# ============================================================================
# LAMBDA HANDLER - AWS Entry Point
# ============================================================================

def lambda_handler(event, context):
    """
    AWS Lambda handler function - Entry point for all invocations
    
    This function is called by AWS Lambda runtime. The signature MUST be:
    lambda_handler(event, context)
    
    Args:
        event: Dict containing request data. Format depends on trigger:
            - API Gateway: event['body'] contains JSON string
            - Direct invoke: event is the payload directly
            - S3 trigger: event['Records'] contains S3 event info
            
        context: Lambda context object with runtime info:
            - context.function_name: Name of this Lambda function
            - context.memory_limit_in_mb: Allocated memory
            - context.request_id: Unique ID for this invocation
            - context.get_remaining_time_in_millis(): Time left before timeout
    
    Returns:
        Dict with 'statusCode' and 'body' (API Gateway format)
        
    Why return statusCode and body instead of just the result?
    
    API Gateway expects this format. statusCode tells the HTTP response
    code (200=success, 400=bad request, 500=error). body contains the actual
    response data as a JSON string. This separates HTTP concerns from business logic.
    """
    
    # Log request info (appears in CloudWatch)
    print(f"Request ID: {context.aws_request_id}")
    print(f"Function name: {context.function_name}")
    print(f"Memory limit: {context.memory_limit_in_mb} MB")
    print(f"Time remaining: {context.get_remaining_time_in_millis()} ms")
    
    try:
        # Parse input based on invocation source
        # API Gateway wraps payload in event['body'] as a string
        # Direct invoke passes payload directly
        
        if isinstance(event, dict) and 'body' in event:
            # API Gateway invocation
            print("Invocation source: API Gateway")
            video_data = json.loads(event['body'])
        elif isinstance(event, dict):
            # Direct invocation
            print("Invocation source: Direct invoke")
            video_data = event
        else:
            raise ValueError(f"Unexpected event format: {type(event)}")
        
        print(f"Input data: {json.dumps(video_data, indent=2)}")
        
        # Validate required fields
        required_fields = ['text_hook', 'caption', 'main_category', 'length']
        missing_fields = [f for f in required_fields if f not in video_data]
        
        if missing_fields:
            return {
                'statusCode': 400,  # Bad Request
                'body': json.dumps({
                    'error': 'Missing required fields',
                    'missing': missing_fields,
                    'required': required_fields
                })
            }
        
        # Make prediction
        print("Making prediction...")
        start_time = datetime.now()
        
        result = make_prediction(video_data)
        
        end_time = datetime.now()
        prediction_time_ms = (end_time - start_time).total_seconds() * 1000
        
        print(f"Prediction completed in {prediction_time_ms:.2f}ms")
        print(f"Result: virality_score={result['virality_score']:.4f}, tier={result['viral_tier']}")
        
        # Add performance metrics
        result['performance'] = {
            'prediction_time_ms': prediction_time_ms,
            'memory_used_mb': context.memory_limit_in_mb,
        }
        
        # Return success response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'  # Enable CORS for web apps
            },
            'body': json.dumps(result)
        }
        
    except json.JSONDecodeError as e:
        # Invalid JSON in request
        print(f"JSON decode error: {str(e)}")
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': 'Invalid JSON format',
                'message': str(e)
            })
        }
    
    except Exception as e:
        # Unexpected error - log full traceback for debugging
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
        
        return {
            'statusCode': 500,  # Internal Server Error
            'body': json.dumps({
                'error': 'Prediction failed',
                'message': str(e),
                'type': type(e).__name__
            })
        }

# ============================================================================
# For local testing (not used by Lambda)
# ============================================================================

if __name__ == "__main__":
    """
    Test handler locally before deploying to AWS
    Simulates Lambda invocation environment
    """
    
    # Mock context object
    class MockContext:
        function_name = "tiktok-virality-predictor-local"
        memory_limit_in_mb = 2048
        request_id = "local-test-123"
        
        def get_remaining_time_in_millis(self):
            return 30000  # 30 seconds
    
    # Test event (simulates API Gateway format)
    test_event = {
        'body': json.dumps({
            'text_hook': 'When the interviewer asks if you have a background in finance',
            'caption': 'I fear this would work on me as an interviewer #corporatehumor',
            'main_category': 'Business',
            'subcategory': 'HR & Payroll',
            'length': 7.0
        })
    }
    
    print("\n" + "="*80)
    print("LOCAL TEST")
    print("="*80)
    
    response = lambda_handler(test_event, MockContext())
    
    print("\n" + "="*80)
    print("RESPONSE")
    print("="*80)
    print(json.dumps(json.loads(response['body']), indent=2))