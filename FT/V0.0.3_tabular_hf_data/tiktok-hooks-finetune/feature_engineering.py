"""
FEATURE ENGINEERING PIPELINE - TikTok Virality Prediction

Purpose:
    Transforms raw TikTok video data into ML-ready features for predicting video virality.
    Creates both regression (virality score) and classification (viral tier) targets.
    
Pipeline Steps:
    1. Load raw data from Hugging Face dataset
    2. Engineer temporal, text, and video features from metadata
    3. Create composite virality metrics from engagement data
    4. Handle missing values and remove statistical outliers
    5. Encode categorical variables (one-hot and target encoding)
    6. Apply appropriate transformations to normalize distributions
    7. Create time-based train/validation/test splits
    8. Save processed datasets and preprocessing artifacts for model training

Input:
    - Hugging Face dataset: benxh/tiktok-hooks-finetune
    - ~46K TikTok videos with metadata and engagement metrics
    
Output:
    - data_train_[timestamp].csv: Training set (70% - oldest data)
    - data_val_[timestamp].csv: Validation set (15% - middle period)
    - data_test_[timestamp].csv: Test set (15% - most recent)
    - scaler_[timestamp].pkl: StandardScaler for feature normalization
    - encoders_[timestamp].pkl: Category and label encoders
    - feature_metadata_[timestamp].json: Feature names and configuration
    
Key Transformations:
    - Virality score: log1p transform to handle skewness and compress outlier range
    - Text features: Character/word counts, hashtag/mention extraction
    - Temporal features: Day of week, month, weekend flags
    - Categories: One-hot encoding for main categories, target encoding for subcategories
    - Classification labels: Integer encoding (0, 1, 2) for compatibility with tree models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datasets import load_dataset
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Feature definitions based on EDA findings
NUMERIC_FEATURES = [
    'length',                    # Video duration in seconds
    'hook_length_chars',         # Opening hook character count
    'hook_length_words',         # Opening hook word count
    'caption_length_chars',      # Caption character count
    'caption_length_words',      # Caption word count
    'hashtag_count',             # Number of hashtags in caption
    'mention_count',             # Number of @ mentions in caption
    'upload_dayofweek',          # Day posted (0=Monday, 6=Sunday)
    'upload_month',              # Month posted (1-12)
    'is_weekend',                # Weekend flag (0=weekday, 1=weekend)
]

CATEGORICAL_FEATURES = [
    'main_category',             # Primary app category (e.g., Entertainment, Education)
    'subcategory',               # Specific subcategory
]

# Target variables
TARGET_REGRESSION = 'virality_score'           # Continuous engagement metric
TARGET_CLASSIFICATION = 'viral_tier'           # Categorical tier (low/viral/mega)

# Data cleaning parameters
OUTLIER_METHOD = 'IQR'        # Use Interquartile Range method for outlier detection
IQR_MULTIPLIER = 1.5          # Standard IQR threshold

# Dataset split ratios (time-based to prevent data leakage)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

print("="*100)
print("FEATURE ENGINEERING PIPELINE - TikTok Virality Prediction")
print("="*100)

# ----------------------------------------------------------------------------
# STEP 1: Load Raw Data
# ----------------------------------------------------------------------------

print("\n" + "="*100)
print("STEP 1: LOADING RAW DATA")
print("="*100)

print("\nLoading dataset from Hugging Face (benxh/tiktok-hooks-finetune)...")
ds = load_dataset("benxh/tiktok-hooks-finetune")
df = ds['train'].to_pandas()

print(f"✓ Loaded {len(df):,} videos with {len(df.columns)} raw features")
print(f"✓ Date range: {df['uploaded_at'].min()} to {df['uploaded_at'].max()}")

# ----------------------------------------------------------------------------
# STEP 2: Calculate Target Variables
# ----------------------------------------------------------------------------

print("\n" + "="*100)
print("STEP 2: CALCULATING TARGET VARIABLES")
print("="*100)

print("\nComputing engagement metrics...")

# Calculate engagement rate (quality metric - engagement per view)
df['engagement_rate'] = (df['likes'] + df['comments'] + df['shares']) / df['views']

# Calculate weighted engagement (prioritizes shares > comments > likes)
df['weighted_engagement'] = (df['shares'] * 3 + df['comments'] * 2 + df['likes'] * 1)

# Virality score combines engagement quality with reach scale
# Formula: engagement_rate × log(views + 1)
# - High engagement rate = quality content
# - log(views) = reach at scale (compresses outliers)
df['virality_score'] = df['engagement_rate'] * np.log1p(df['views'])

print(f"✓ Virality score calculated")
print(f"  Range: {df['virality_score'].min():.4f} to {df['virality_score'].max():.4f}")
print(f"  Mean: {df['virality_score'].mean():.4f}, Median: {df['virality_score'].median():.4f}")
print(f"  Skewness: {df['virality_score'].skew():.2f}")

# Apply log transformation to virality score for better model performance
# Log transform handles skewness and compresses outlier range
df['virality_score_log'] = np.log1p(df['virality_score'])

print(f"\n✓ Log-transformed virality score for modeling")
print(f"  Log range: {df['virality_score_log'].min():.4f} to {df['virality_score_log'].max():.4f}")
print(f"  Log mean: {df['virality_score_log'].mean():.4f}, Log skewness: {df['virality_score_log'].skew():.2f}")

# Create viral tier classifications based on percentiles
print("\nCreating viral tier classifications...")
low_threshold = df['virality_score'].quantile(0.20)    # Bottom 20%
mega_threshold = df['virality_score'].quantile(0.95)   # Top 5%

df['viral_tier'] = pd.cut(
    df['virality_score'],
    bins=[-float('inf'), low_threshold, mega_threshold, float('inf')],
    labels=['low_viral', 'viral', 'mega_viral']
)

tier_dist = df['viral_tier'].value_counts().sort_index()
print(f"✓ Viral tiers created:")
print(f"  low_viral (< {low_threshold:.4f}): {tier_dist.get('low_viral', 0):,} ({tier_dist.get('low_viral', 0)/len(df)*100:.1f}%)")
print(f"  viral ({low_threshold:.4f} - {mega_threshold:.4f}): {tier_dist.get('viral', 0):,} ({tier_dist.get('viral', 0)/len(df)*100:.1f}%)")
print(f"  mega_viral (≥ {mega_threshold:.4f}): {tier_dist.get('mega_viral', 0):,} ({tier_dist.get('mega_viral', 0)/len(df)*100:.1f}%)")

# ----------------------------------------------------------------------------
# STEP 3: Engineer Features
# ----------------------------------------------------------------------------

print("\n" + "="*100)
print("STEP 3: FEATURE ENGINEERING")
print("="*100)

# Temporal features from upload timestamp
print("\nExtracting temporal features from upload_at...")
df['uploaded_at'] = pd.to_datetime(df['uploaded_at'])
df['upload_dayofweek'] = df['uploaded_at'].dt.dayofweek  # 0=Monday, 6=Sunday
df['upload_month'] = df['uploaded_at'].dt.month           # 1-12
df['is_weekend'] = df['upload_dayofweek'].isin([5, 6]).astype(int)  # Weekend flag

print(f"✓ Temporal features: upload_dayofweek, upload_month, is_weekend")

# Text features from hooks and captions
print("\nExtracting text features...")
df['hook_length_chars'] = df['text_hook'].astype(str).str.len()
df['hook_length_words'] = df['text_hook'].astype(str).str.split().str.len()
df['caption_length_chars'] = df['caption'].astype(str).str.len()
df['caption_length_words'] = df['caption'].astype(str).str.split().str.len()
df['hashtag_count'] = df['caption'].astype(str).str.count(r'#')
df['mention_count'] = df['caption'].astype(str).str.count(r'@')

print(f"✓ Text features: hook/caption lengths, hashtag/mention counts")

# Video features (length already exists in raw data)
print(f"✓ Video features: length (duration in seconds)")

# Verify all required features exist
all_required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['virality_score_log', 'viral_tier']
missing = [f for f in all_required if f not in df.columns]
if missing:
    raise ValueError(f"Missing required features: {missing}")

print(f"\n✓ All {len(NUMERIC_FEATURES + CATEGORICAL_FEATURES)} input features verified")

# ----------------------------------------------------------------------------
# STEP 4: Handle Missing Values
# ----------------------------------------------------------------------------

print("\n" + "="*100)
print("STEP 4: HANDLING MISSING VALUES")
print("="*100)

missing_counts = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isnull().sum()
total_missing = missing_counts.sum()

if total_missing > 0:
    print(f"\n⚠ Found {total_missing:,} missing values across features")
    
    # Impute numeric features with median (robust to outliers)
    for col in NUMERIC_FEATURES:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  ✓ {col}: filled {df[col].isnull().sum()} missing with median ({median_val:.2f})")
    
    # Impute categorical features with mode (most frequent value)
    for col in CATEGORICAL_FEATURES:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  ✓ {col}: filled {df[col].isnull().sum()} missing with mode ('{mode_val}')")
    
    # Remove rows with missing targets (cannot train on these)
    before = len(df)
    df = df.dropna(subset=['virality_score_log', 'viral_tier'])
    dropped = before - len(df)
    if dropped > 0:
        print(f"  ✓ Dropped {dropped:,} rows with missing target variables")
else:
    print("✓ No missing values detected in feature columns")

print(f"\nDataset size after handling missing values: {len(df):,} rows")

# ----------------------------------------------------------------------------
# STEP 5: Remove Statistical Outliers
# ----------------------------------------------------------------------------

print("\n" + "="*100)
print("STEP 5: REMOVING STATISTICAL OUTLIERS")
print("="*100)

print(f"\nApplying IQR method (multiplier: {IQR_MULTIPLIER}) to engagement metrics...")
print("Purpose: Remove extreme outliers that could destabilize model training")

initial_size = len(df)
outliers_removed_per_metric = {}

for col in ['views', 'likes', 'comments', 'shares']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - IQR_MULTIPLIER * IQR
    upper_bound = Q3 + IQR_MULTIPLIER * IQR
    
    before = len(df)
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    removed = before - len(df)
    outliers_removed_per_metric[col] = removed
    
    print(f"  {col:10s}: removed {removed:5,} outliers (kept values in [{lower_bound:,.0f}, {upper_bound:,.0f}])")

total_outliers = initial_size - len(df)
print(f"\n✓ Total outliers removed: {total_outliers:,} ({total_outliers/initial_size*100:.2f}%)")
print(f"✓ Remaining dataset: {len(df):,} rows")

# ----------------------------------------------------------------------------
# STEP 6: Encode Categorical Features
# ----------------------------------------------------------------------------

print("\n" + "="*100)
print("STEP 6: ENCODING CATEGORICAL FEATURES")
print("="*100)

# One-hot encode main_category (manageable number of categories)
print("\nOne-hot encoding main_category...")
main_cat_dummies = pd.get_dummies(df['main_category'], prefix='cat', drop_first=False)
print(f"✓ Created {len(main_cat_dummies.columns)} binary category features")

# Target encode subcategory (too many unique values for one-hot)
# Target encoding maps each subcategory to its mean virality score with smoothing
print("\nTarget encoding subcategory (with smoothing to prevent overfitting)...")

global_mean = df['virality_score'].mean()
subcategory_means = df.groupby('subcategory')['virality_score'].mean()
subcategory_counts = df['subcategory'].value_counts()

# Smoothing factor: higher = more conservative, relies more on global mean
smoothing_factor = 100

subcategory_encoded_map = {}
for subcat in df['subcategory'].unique():
    if pd.isna(subcat):
        subcategory_encoded_map[subcat] = global_mean
    else:
        count = subcategory_counts.get(subcat, 0)
        mean = subcategory_means.get(subcat, global_mean)
        # Smoothed mean = weighted average of category mean and global mean
        smoothed = (count * mean + smoothing_factor * global_mean) / (count + smoothing_factor)
        subcategory_encoded_map[subcat] = smoothed

df['subcategory_encoded'] = df['subcategory'].map(subcategory_encoded_map)

print(f"✓ Encoded {len(subcategory_encoded_map)} unique subcategories")
print(f"  Encoded value range: [{df['subcategory_encoded'].min():.4f}, {df['subcategory_encoded'].max():.4f}]")

# Encode classification labels as integers for XGBoost compatibility
print("\nEncoding viral_tier labels as integers...")
label_encoder = LabelEncoder()
df['viral_tier_encoded'] = label_encoder.fit_transform(df['viral_tier'])

# Create mapping for later decoding predictions
class_mapping = {str(k): int(v) for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
print(f"✓ Label encoding mapping:")
for original_label, encoded_value in sorted(class_mapping.items(), key=lambda x: x[1]):
    count = (df['viral_tier'] == original_label).sum()
    print(f"  {original_label:12s} → {encoded_value}  ({count:,} videos)")

# Combine all features
df_features = pd.concat([df, main_cat_dummies], axis=1)

# Update feature lists
NUMERIC_FEATURES_ENCODED = NUMERIC_FEATURES + ['subcategory_encoded']
CATEGORY_DUMMIES = list(main_cat_dummies.columns)
ALL_FEATURES = NUMERIC_FEATURES_ENCODED + CATEGORY_DUMMIES

print(f"\n✓ Total model input features: {len(ALL_FEATURES)}")
print(f"  - Numeric features: {len(NUMERIC_FEATURES_ENCODED)}")
print(f"  - Category dummies: {len(CATEGORY_DUMMIES)}")

# ----------------------------------------------------------------------------
# STEP 7: Scale Numeric Features
# ----------------------------------------------------------------------------

print("\n" + "="*100)
print("STEP 7: FEATURE NORMALIZATION")
print("="*100)

print("\nApplying StandardScaler to numeric features...")
print("Purpose: Normalize feature scales for improved model convergence")
print("Note: Target variables are NOT scaled - only input features")

scaler = StandardScaler()
df_features[NUMERIC_FEATURES_ENCODED] = scaler.fit_transform(df_features[NUMERIC_FEATURES_ENCODED])

print(f"✓ Standardized {len(NUMERIC_FEATURES_ENCODED)} numeric features (mean≈0, std≈1)")

# Verify targets were NOT scaled
print(f"\n✓ Target variable verification:")
print(f"  virality_score_log: mean={df_features['virality_score_log'].mean():.4f}, std={df_features['virality_score_log'].std():.4f}")
print(f"  (Should NOT be mean≈0, std≈1 - targets remain in original scale)")

# ----------------------------------------------------------------------------
# STEP 8: Create Train/Validation/Test Splits
# ----------------------------------------------------------------------------

print("\n" + "="*100)
print("STEP 8: CREATING TRAIN/VALIDATION/TEST SPLITS")
print("="*100)

print(f"\nSplit strategy: Time-based (prevents data leakage)")
print(f"  - Train: {TRAIN_RATIO:.0%} (oldest videos)")
print(f"  - Validation: {VAL_RATIO:.0%} (middle period)")  
print(f"  - Test: {TEST_RATIO:.0%} (most recent videos)")
print(f"\nRationale: Validates model performance on future, unseen time periods")

# Sort by upload date (oldest to newest)
df_features = df_features.sort_values('uploaded_at').reset_index(drop=True)

# Calculate split indices
train_size = int(TRAIN_RATIO * len(df_features))
val_size = int(VAL_RATIO * len(df_features))

# Split datasets
train_df = df_features.iloc[:train_size].copy()
val_df = df_features.iloc[train_size:train_size + val_size].copy()
test_df = df_features.iloc[train_size + val_size:].copy()

print(f"\n✓ Dataset splits:")
print(f"  Train: {len(train_df):6,} rows ({len(train_df)/len(df_features)*100:5.2f}%)")
print(f"  Val:   {len(val_df):6,} rows ({len(val_df)/len(df_features)*100:5.2f}%)")
print(f"  Test:  {len(test_df):6,} rows ({len(test_df)/len(df_features)*100:5.2f}%)")

print(f"\n✓ Temporal coverage:")
print(f"  Train: {train_df['uploaded_at'].min().date()} to {train_df['uploaded_at'].max().date()}")
print(f"  Val:   {val_df['uploaded_at'].min().date()} to {val_df['uploaded_at'].max().date()}")
print(f"  Test:  {test_df['uploaded_at'].min().date()} to {test_df['uploaded_at'].max().date()}")

# Verify viral tier distributions are balanced across splits
print(f"\n✓ Viral tier distribution across splits:")
for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    dist = split_df['viral_tier'].value_counts(normalize=True).sort_index()
    print(f"  {split_name:5s}: ", end='')
    for tier in ['low_viral', 'viral', 'mega_viral']:
        pct = dist.get(tier, 0) * 100
        print(f"{tier}={pct:5.2f}%  ", end='')
    print()

# ----------------------------------------------------------------------------
# STEP 9: Save Processed Data and Artifacts
# ----------------------------------------------------------------------------

print("\n" + "="*100)
print("STEP 9: SAVING PROCESSED DATA AND ARTIFACTS")
print("="*100)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"\nTimestamp for this pipeline run: {timestamp}")

# Save train/validation/test datasets
print("\nSaving processed datasets...")
train_df.to_csv(f'data_train_{timestamp}.csv', index=False)
val_df.to_csv(f'data_val_{timestamp}.csv', index=False)
test_df.to_csv(f'data_test_{timestamp}.csv', index=False)

print(f"  ✓ data_train_{timestamp}.csv ({len(train_df):,} rows)")
print(f"  ✓ data_val_{timestamp}.csv ({len(val_df):,} rows)")
print(f"  ✓ data_test_{timestamp}.csv ({len(test_df):,} rows)")

# Save StandardScaler for inference-time feature normalization
print("\nSaving preprocessing artifacts...")
with open(f'scaler_{timestamp}.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ scaler_{timestamp}.pkl (for feature normalization during inference)")

# Save encoders for categorical features and label decoding
encoders = {
    'subcategory_encoder': subcategory_encoded_map,     # Subcategory → virality mean mapping
    'main_category_columns': CATEGORY_DUMMIES,          # List of one-hot encoded columns
    'label_encoder': label_encoder,                      # viral_tier string ↔ integer mapping
    'class_mapping': class_mapping,                      # Explicit mapping for clarity
}

with open(f'encoders_{timestamp}.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print(f"  ✓ encoders_{timestamp}.pkl (category and label encoders)")

# Save metadata for model training and inference configuration
metadata = {
    'pipeline_timestamp': timestamp,
    'dataset_info': {
        'total_rows_processed': len(df_features),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'outliers_removed': total_outliers,
    },
    'features': {
        'numeric_features': NUMERIC_FEATURES_ENCODED,
        'category_features': CATEGORY_DUMMIES,
        'all_features': ALL_FEATURES,
        'total_count': len(ALL_FEATURES),
    },
    'targets': {
        'regression': 'virality_score_log',              # Use log-transformed target for training
        'regression_original': 'virality_score',         # Original scale for evaluation
        'classification': 'viral_tier_encoded',          # Integer-encoded labels for training
        'classification_original': 'viral_tier',         # String labels for reporting
    },
    'transformations': {
        'target_transform': 'log1p',                     # Log transformation applied
        'inverse_transform': 'expm1',                    # Use this to get predictions back to original scale
        'feature_scaling': 'StandardScaler',
    },
    'viral_tier_thresholds': {
        'low_threshold': float(low_threshold),
        'mega_threshold': float(mega_threshold),
    },
    'class_encoding': class_mapping,
    'config': {
        'random_seed': RANDOM_SEED,
        'outlier_method': OUTLIER_METHOD,
        'iqr_multiplier': IQR_MULTIPLIER,
    },
}

with open(f'feature_metadata_{timestamp}.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ feature_metadata_{timestamp}.json (pipeline configuration and feature info)")

# ----------------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------------

print("\n" + "="*100)
print("PIPELINE SUMMARY")
print("="*100)

print(f"\n✓ Feature engineering completed successfully")
print(f"\nDataset Statistics:")
print(f"  Original size: {len(ds['train']):,} rows")
print(f"  After cleaning: {len(df_features):,} rows ({len(df_features)/len(ds['train'])*100:.1f}%)")
print(f"  Outliers removed: {total_outliers:,}")

print(f"\nFeature Summary:")
print(f"  Input features: {len(ALL_FEATURES)}")
print(f"    - Temporal: upload_dayofweek, upload_month, is_weekend")
print(f"    - Text: hook/caption lengths, hashtag/mention counts")
print(f"    - Video: duration (length)")
print(f"    - Category: {len(CATEGORY_DUMMIES)} one-hot encoded categories + 1 target-encoded subcategory")

print(f"\nTarget Variables:")
print(f"  Regression: virality_score_log (log-transformed, range: {df_features['virality_score_log'].min():.2f} to {df_features['virality_score_log'].max():.2f})")
print(f"  Classification: viral_tier_encoded (3 classes: 0=low_viral, 1=mega_viral, 2=viral)")

print(f"\nDataset Splits:")
print(f"  Train: {len(train_df):,} rows (oldest: {train_df['uploaded_at'].min().date()} to {train_df['uploaded_at'].max().date()})")
print(f"  Val:   {len(val_df):,} rows (middle: {val_df['uploaded_at'].min().date()} to {val_df['uploaded_at'].max().date()})")
print(f"  Test:  {len(test_df):,} rows (newest: {test_df['uploaded_at'].min().date()} to {test_df['uploaded_at'].max().date()})")

print(f"\nGenerated Files:")
print(f"  • data_train_{timestamp}.csv")
print(f"  • data_val_{timestamp}.csv")
print(f"  • data_test_{timestamp}.csv")
print(f"  • scaler_{timestamp}.pkl")
print(f"  • encoders_{timestamp}.pkl")
print(f"  • feature_metadata_{timestamp}.json")

print("\n" + "="*100)
print("✓ READY FOR MODEL TRAINING")
print("="*100)
print(f"\nNext step: python train_models.py")
print(f"Timestamp to use: {timestamp}")