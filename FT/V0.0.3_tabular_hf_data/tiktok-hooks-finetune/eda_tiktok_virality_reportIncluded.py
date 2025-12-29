"""
EDA: TikTok Virality Prediction
Run this locally with the full 46K dataset from Hugging Face
"""
"""
🔍 COMPREHENSIVE EDA: TikTok Virality Prediction
Saves all output to a report file + processed data
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Create report file
report_file = f'EDA_REPORT_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
report = open(report_file, 'w', encoding='utf-8')

def print_both(text=""):
    """Print to both console and report file"""
    print(text)
    report.write(text + '\n')

print_both("="*100)
print_both("📊 COMPREHENSIVE EDA REPORT - TikTok Virality Prediction")
print_both(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_both("="*100)

print_both("\n📥 LOADING DATA FROM HUGGING FACE")
print_both("="*100)

# Load data
from datasets import load_dataset
ds = load_dataset("benxh/tiktok-hooks-finetune")
df = ds['train'].to_pandas()

print_both(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================================
# 1. BASIC DATA OVERVIEW
# ============================================================================
print_both("\n" + "="*100)
print_both("1️⃣ BASIC DATASET OVERVIEW")
print_both("="*100)

print_both(f"\n📋 Dataset Shape: {df.shape}")
print_both(f"\n📝 Column Names:")
for i, col in enumerate(df.columns, 1):
    print_both(f"  {i:2d}. {col:30s} | Type: {df[col].dtype}")

print_both("\n🔍 Data Types Summary:")
type_counts = df.dtypes.value_counts()
for dtype, count in type_counts.items():
    print_both(f"  {dtype}: {count}")

print_both("\n❓ Missing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Percentage': missing_pct.values
}).sort_values('Missing_Count', ascending=False)

if missing_df['Missing_Count'].sum() > 0:
    print_both(missing_df[missing_df['Missing_Count'] > 0].to_string(index=False))
else:
    print_both("  ✅ No missing values found!")

# ============================================================================
# 2. TARGET VARIABLES - ENGAGEMENT METRICS
# ============================================================================
print_both("\n" + "="*100)
print_both("2️⃣ ENGAGEMENT METRICS ANALYSIS")
print_both("="*100)

engagement_cols = ['views', 'likes', 'comments', 'shares']

print_both("\n📊 Basic Statistics:")
stats_df = df[engagement_cols].describe()
print_both(stats_df.round(2).to_string())

print_both("\n📈 Distribution Characteristics:")
for col in engagement_cols:
    print_both(f"\n{col.upper()}:")
    print_both(f"  Mean:     {df[col].mean():,.2f}")
    print_both(f"  Median:   {df[col].median():,.2f}")
    print_both(f"  Std Dev:  {df[col].std():,.2f}")
    print_both(f"  Min:      {df[col].min():,}")
    print_both(f"  Max:      {df[col].max():,}")
    print_both(f"  Skewness: {df[col].skew():.2f}")
    print_both(f"  Kurtosis: {df[col].kurtosis():.2f}")

# Calculate engagement rate
df['engagement_rate'] = (df['likes'] + df['comments'] + df['shares']) / df['views']
df['weighted_engagement'] = (df['shares'] * 3 + df['comments'] * 2 + df['likes'] * 1)
df['virality_score'] = df['engagement_rate'] * np.log1p(df['views'])

print_both("\n🎯 VIRALITY SCORE CALCULATION:")
print_both(f"  Formula: engagement_rate × log(views + 1)")
print_both(f"  Where: engagement_rate = (likes + comments + shares) / views")
print_both(f"  Where: weighted_engagement = (shares×3 + comments×2 + likes×1)")
print_both("\n  Virality Score Statistics:")
virality_stats = df['virality_score'].describe()
for stat, val in virality_stats.items():
    print_both(f"    {stat:10s}: {val:.4f}")

# ============================================================================
# 3. CORRELATION ANALYSIS
# ============================================================================
print_both("\n" + "="*100)
print_both("3️⃣ CORRELATION ANALYSIS")
print_both("="*100)

corr_cols = ['views', 'likes', 'comments', 'shares', 'engagement_rate', 
             'weighted_engagement', 'virality_score', 'length', 'outlier_multiplier']
corr_matrix = df[corr_cols].corr()

print_both("\n🔗 Correlation with Virality Score:")
virality_corr = corr_matrix['virality_score'].sort_values(ascending=False)
for feat, corr_val in virality_corr.items():
    print_both(f"  {feat:25s}: {corr_val:+.4f}")

print_both("\n📊 Full Correlation Matrix:")
print_both(corr_matrix.round(3).to_string())

# ============================================================================
# 4. CATEGORICAL FEATURES ANALYSIS
# ============================================================================
print_both("\n" + "="*100)
print_both("4️⃣ CATEGORICAL FEATURES ANALYSIS")
print_both("="*100)

print_both("\n👤 Unique Usernames:")
print_both(f"  Total: {df['username'].nunique():,}")
print_both(f"\n  Top 10 Most Active:")
top_creators = df['username'].value_counts().head(10)
for creator, count in top_creators.items():
    avg_views = df[df['username'] == creator]['views'].mean()
    print_both(f"    {creator:30s}: {count:3d} videos | Avg Views: {avg_views:,.0f}")

print_both("\n📂 Main Categories:")
cat_counts = df['main_category'].value_counts()
for cat, count in cat_counts.items():
    pct = count / len(df) * 100
    print_both(f"  {cat:30s}: {count:5d} ({pct:5.2f}%)")

print_both(f"\n  📊 Category Performance (Avg Views):")
cat_performance = df.groupby('main_category')['views'].mean().sort_values(ascending=False)
for cat, views in cat_performance.items():
    print_both(f"    {cat:30s}: {views:10,.0f}")

print_both("\n📁 Subcategories:")
print_both(f"  Total unique: {df['subcategory'].nunique()}")
print_both(f"\n  Top 15:")
subcat_counts = df['subcategory'].value_counts().head(15)
for subcat, count in subcat_counts.items():
    print_both(f"    {subcat:40s}: {count:5d}")

print_both("\n🌍 Languages:")
print_both(f"  Hook Languages:")
hook_langs = df['text_hook_lang'].value_counts()
for lang, count in hook_langs.items():
    print_both(f"    {lang}: {count:,}")

print_both(f"\n  Caption Languages:")
caption_langs = df['caption_lang'].value_counts()
for lang, count in caption_langs.items():
    print_both(f"    {lang}: {count:,}")

# ============================================================================
# 5. TEXT FEATURES ANALYSIS
# ============================================================================
print_both("\n" + "="*100)
print_both("5️⃣ TEXT FEATURES ANALYSIS")
print_both("="*100)

# Hook analysis
df['hook_length_chars'] = df['text_hook'].astype(str).str.len()
df['hook_length_words'] = df['text_hook'].astype(str).str.split().str.len()
df['hook_has_question'] = df['text_hook'].astype(str).str.contains(r'\?', na=False)
df['hook_has_numbers'] = df['text_hook'].astype(str).str.contains(r'\d', na=False)

# Caption analysis
df['caption_length_chars'] = df['caption'].astype(str).str.len()
df['caption_length_words'] = df['caption'].astype(str).str.split().str.len()
df['hashtag_count'] = df['caption'].astype(str).str.count(r'#')
df['mention_count'] = df['caption'].astype(str).str.count(r'@')

print_both("\n📝 HOOK ANALYSIS:")
print_both(f"  Avg Length (chars): {df['hook_length_chars'].mean():.1f}")
print_both(f"  Median Length (chars): {df['hook_length_chars'].median():.1f}")
print_both(f"  Avg Length (words): {df['hook_length_words'].mean():.1f}")
print_both(f"  % with Questions:   {df['hook_has_question'].mean()*100:.1f}%")
print_both(f"  % with Numbers:     {df['hook_has_numbers'].mean()*100:.1f}%")

print_both("\n💬 CAPTION ANALYSIS:")
print_both(f"  Avg Length (chars): {df['caption_length_chars'].mean():.1f}")
print_both(f"  Median Length (chars): {df['caption_length_chars'].median():.1f}")
print_both(f"  Avg Length (words): {df['caption_length_words'].mean():.1f}")
print_both(f"  Avg Hashtags:       {df['hashtag_count'].mean():.1f}")
print_both(f"  Median Hashtags:    {df['hashtag_count'].median():.1f}")
print_both(f"  Avg Mentions:       {df['mention_count'].mean():.1f}")

# Text features vs virality
print_both("\n🎯 TEXT FEATURES vs VIRALITY SCORE:")
text_features = ['hook_length_chars', 'hook_length_words', 'caption_length_chars', 
                 'caption_length_words', 'hashtag_count', 'mention_count']
for feat in text_features:
    corr = df[feat].corr(df['virality_score'])
    print_both(f"  {feat:25s}: {corr:+.4f}")

# ============================================================================
# 6. TEMPORAL ANALYSIS
# ============================================================================
print_both("\n" + "="*100)
print_both("6️⃣ TEMPORAL ANALYSIS")
print_both("="*100)

df['uploaded_at'] = pd.to_datetime(df['uploaded_at'])
df['upload_date'] = df['uploaded_at'].dt.date
df['upload_hour'] = df['uploaded_at'].dt.hour
df['upload_dayofweek'] = df['uploaded_at'].dt.dayofweek
df['upload_month'] = df['uploaded_at'].dt.month
df['is_weekend'] = df['upload_dayofweek'].isin([5, 6])

print_both("\n📅 Upload Date Range:")
print_both(f"  First: {df['uploaded_at'].min()}")
print_both(f"  Last:  {df['uploaded_at'].max()}")
print_both(f"  Span:  {(df['uploaded_at'].max() - df['uploaded_at'].min()).days} days")

print_both("\n⏰ Upload Time Distribution:")
print_both(f"  By Hour (All 24 hours):")
hour_dist = df['upload_hour'].value_counts().sort_index()
for hour, count in hour_dist.items():
    avg_views = df[df['upload_hour'] == hour]['views'].mean()
    print_both(f"    Hour {hour:2d}: {count:5d} videos | Avg Views: {avg_views:,.0f}")

print_both(f"\n  By Day of Week:")
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_dist = df['upload_dayofweek'].value_counts().sort_index()
for dow, count in dow_dist.items():
    avg_views = df[df['upload_dayofweek'] == dow]['views'].mean()
    avg_virality = df[df['upload_dayofweek'] == dow]['virality_score'].mean()
    print_both(f"    {dow_names[dow]}: {count:5d} videos | Avg Views: {avg_views:,.0f} | Avg Virality: {avg_virality:.4f}")

print_both(f"\n  By Month:")
month_dist = df['upload_month'].value_counts().sort_index()
for month, count in month_dist.items():
    avg_views = df[df['upload_month'] == month]['views'].mean()
    print_both(f"    Month {month:2d}: {count:5d} videos | Avg Views: {avg_views:,.0f}")

print_both(f"\n  Weekend vs Weekday:")
weekend_views = df[df['is_weekend']]['views'].mean()
weekday_views = df[~df['is_weekend']]['views'].mean()
weekend_virality = df[df['is_weekend']]['virality_score'].mean()
weekday_virality = df[~df['is_weekend']]['virality_score'].mean()
print_both(f"    Weekend: {weekend_views:,.0f} avg views | {weekend_virality:.4f} avg virality")
print_both(f"    Weekday: {weekday_views:,.0f} avg views | {weekday_virality:.4f} avg virality")
diff_pct = ((weekend_views - weekday_views) / weekday_views) * 100
print_both(f"    Difference: {diff_pct:+.1f}%")

# ============================================================================
# 7. VIDEO LENGTH ANALYSIS
# ============================================================================
print_both("\n" + "="*100)
print_both("7️⃣ VIDEO LENGTH ANALYSIS")
print_both("="*100)

print_both(f"\n📏 Video Length (seconds):")
length_stats = df['length'].describe()
for stat, val in length_stats.items():
    print_both(f"  {stat:10s}: {val:.2f}")

# Bin video lengths
df['length_category'] = pd.cut(df['length'], 
                                bins=[0, 10, 20, 30, 60, float('inf')],
                                labels=['0-10s', '10-20s', '20-30s', '30-60s', '60s+'])

print_both(f"\n  Length Categories:")
for cat in ['0-10s', '10-20s', '20-30s', '30-60s', '60s+']:
    if cat in df['length_category'].values:
        cat_df = df[df['length_category'] == cat]
        count = len(cat_df)
        avg_views = cat_df['views'].mean()
        avg_virality = cat_df['virality_score'].mean()
        print_both(f"    {cat:10s}: {count:5d} videos | Avg Views: {avg_views:8,.0f} | Avg Virality: {avg_virality:.4f}")

# ============================================================================
# 8. OUTLIER ANALYSIS
# ============================================================================
print_both("\n" + "="*100)
print_both("8️⃣ OUTLIER ANALYSIS")
print_both("="*100)

print_both("\n🔥 Top 10 Most Viral Videos:")
top_viral = df.nlargest(10, 'virality_score')[['username', 'main_category', 'views', 
                                                  'likes', 'shares', 'comments', 'virality_score']]
print_both(top_viral.to_string(index=False))

print_both("\n👀 Top 10 Most Viewed Videos:")
top_views = df.nlargest(10, 'views')[['username', 'main_category', 'views', 
                                        'likes', 'shares', 'comments', 'engagement_rate']]
print_both(top_views.to_string(index=False))

# Outlier detection using IQR
print_both("\n📊 Outlier Detection (IQR Method):")
for col in engagement_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print_both(f"  {col.upper():10s}: {len(outliers):6d} outliers ({len(outliers)/len(df)*100:5.2f}%) | Bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")

# ============================================================================
# 9. VIRALITY THRESHOLDS
# ============================================================================
print_both("\n" + "="*100)
print_both("9️⃣ VIRALITY CLASSIFICATION THRESHOLDS")
print_both("="*100)

# Define thresholds based on percentiles
percentiles = [5, 10, 20, 25, 50, 75, 80, 90, 95, 99]
print_both("\n🎯 Virality Score Percentiles:")
for p in percentiles:
    threshold = df['virality_score'].quantile(p/100)
    print_both(f"  {p:2d}th percentile: {threshold:.4f}")

# Proposed classification
low_threshold = df['virality_score'].quantile(0.20)
mega_threshold = df['virality_score'].quantile(0.95)

df['viral_tier'] = pd.cut(df['virality_score'],
                           bins=[-float('inf'), low_threshold, mega_threshold, float('inf')],
                           labels=['low_viral', 'viral', 'mega_viral'])

print_both(f"\n📊 Proposed Classification:")
print_both(f"  low_viral:  virality_score < {low_threshold:.4f}  (bottom 20%)")
print_both(f"  viral:      {low_threshold:.4f} ≤ virality_score < {mega_threshold:.4f}  (20-95%)")
print_both(f"  mega_viral: virality_score ≥ {mega_threshold:.4f}  (top 5%)")

print_both(f"\n  Distribution:")
tier_dist = df['viral_tier'].value_counts().sort_index()
for tier, count in tier_dist.items():
    pct = count / len(df) * 100
    print_both(f"    {tier:12s}: {count:6d} ({pct:5.2f}%)")

print_both(f"\n  Avg Metrics by Tier:")
tier_stats = df.groupby('viral_tier')[['views', 'likes', 'comments', 'shares', 'engagement_rate']].mean()
print_both(tier_stats.round(2).to_string())

# ============================================================================
# 10. FEATURE IMPORTANCE PREVIEW
# ============================================================================
print_both("\n" + "="*100)
print_both("🔟 FEATURE CORRELATION WITH VIRALITY SCORE")
print_both("="*100)

# Collect all numeric features
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove target variables
exclude_cols = ['views', 'likes', 'comments', 'shares', 'engagement_rate', 
                'weighted_engagement', 'virality_score']
feature_cols = [col for col in numeric_features if col not in exclude_cols]

print_both(f"\n🔗 All Features Correlated with Virality Score:")
feature_corr = df[feature_cols + ['virality_score']].corr()['virality_score'].drop('virality_score')
feature_corr_sorted = feature_corr.abs().sort_values(ascending=False)

for feat in feature_corr_sorted.index:
    actual_corr = feature_corr[feat]
    print_both(f"  {feat:30s}: {actual_corr:+.4f}")

# ============================================================================
# 11. KEY INSIGHTS & RECOMMENDATIONS
# ============================================================================
print_both("\n" + "="*100)
print_both("💡 KEY INSIGHTS & RECOMMENDATIONS")
print_both("="*100)

print_both("""
DATASET CHARACTERISTICS:
✓ Highly skewed engagement metrics (use log transform)
✓ Large range in views (3 to millions) - need normalization
✓ Sparse comments/shares (many zeros) - handle carefully
✓ Clear viral tiers exist (20% low, 75% viral, 5% mega)
✓ Multiple categories with different performance levels

MODELING RECOMMENDATIONS:
1. Remove outliers using IQR method before training
2. Log-transform engagement metrics (views, likes, shares, comments)
3. Use engagement_rate as quality metric (not just raw views)
4. Temporal features (hour, day, month) are important
5. Text length features show weak but consistent correlation
6. Video duration appears important based on research

FEATURE ENGINEERING PRIORITIES:
HIGH:   Temporal features (hour, day, weekend), video duration, hashtag counts
MEDIUM: Text length, category encoding, language features  
LOW:    Username (use category aggregates instead for generalization)

TARGET VARIABLES:
• Regression: virality_score (continuous)
• Classification: viral_tier (3 classes: low_viral, viral, mega_viral)

MODEL CHOICE:
→ XGBoost (proven best in research papers)
→ Handles mixed feature types
→ Fast inference for Lambda
→ Interpretable feature importance

NEXT STEPS:
→ Build feature engineering pipeline
→ Train XGBoost regression + classification models
→ Validate on time-based split (train on old, test on recent)
→ Deploy to AWS Lambda
""")

print_both("\n" + "="*100)
print_both("✅ EDA COMPLETE!")
print_both("="*100)

# Save processed dataset
csv_filename = f'tiktok_data_processed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
print_both(f"\n💾 Saving processed dataset...")
df.to_csv(csv_filename, index=False)
print_both(f"✅ Saved to: {csv_filename}")

# Save feature list for modeling
feature_list_file = 'feature_list.txt'
with open(feature_list_file, 'w') as f:
    f.write("ENGINEERED FEATURES FOR MODELING\n")
    f.write("="*50 + "\n\n")
    f.write("Numeric Features:\n")
    for feat in sorted(feature_cols):
        f.write(f"  - {feat}\n")
    f.write("\nCategorical Features:\n")
    f.write("  - main_category\n")
    f.write("  - subcategory\n")
    f.write("  - text_hook_lang\n")
    f.write("  - caption_lang\n")
    f.write("  - platform\n")
    f.write("\nTarget Variables:\n")
    f.write("  - virality_score (regression)\n")
    f.write("  - viral_tier (classification)\n")

print_both(f"\n📝 Feature list saved to: {feature_list_file}")

print_both("\n🎯 FILES GENERATED:")
print_both(f"  1. {report_file} - This comprehensive report")
print_both(f"  2. {csv_filename} - Processed data with engineered features")
print_both(f"  3. {feature_list_file} - List of features for modeling")

print_both("\n🚀 Ready to proceed with model training!")

# Close report file
report.close()
print(f"\n✅ Report saved to: {report_file}")
print(f"📖 Open this file to review all EDA findings!")