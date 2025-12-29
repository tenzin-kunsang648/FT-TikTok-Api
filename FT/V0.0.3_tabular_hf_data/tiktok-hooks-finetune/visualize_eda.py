"""
📊 VISUALIZATION SCRIPT: TikTok Virality EDA
Creates all necessary plots for understanding the data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load processed data
df = pd.read_csv('tiktok_data_processed.csv')
df['uploaded_at'] = pd.to_datetime(df['uploaded_at'])

print("Creating visualizations...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# ============================================================================
# 1. ENGAGEMENT METRICS DISTRIBUTIONS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Engagement Metrics Distributions (Log Scale)', fontsize=16, fontweight='bold')

metrics = ['views', 'likes', 'comments', 'shares']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    # Plot histogram on log scale
    log_data = np.log10(df[metric] + 1)
    ax.hist(log_data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel(f'Log10({metric.capitalize()})', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{metric.capitalize()} Distribution', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add statistics
    mean_val = df[metric].mean()
    median_val = df[metric].median()
    ax.axvline(np.log10(mean_val + 1), color='red', linestyle='--', label=f'Mean: {mean_val:,.0f}')
    ax.axvline(np.log10(median_val + 1), color='green', linestyle='--', label=f'Median: {median_val:,.0f}')
    ax.legend()

plt.tight_layout()
plt.savefig('01_engagement_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_engagement_distributions.png")
plt.close()

# ============================================================================
# 2. VIRALITY SCORE DISTRIBUTION
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Virality Score Analysis', fontsize=16, fontweight='bold')

# Histogram
axes[0].hist(df['virality_score'], bins=100, color='purple', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Virality Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Virality Score Distribution', fontsize=14, fontweight='bold')
axes[0].axvline(df['virality_score'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["virality_score"].mean():.4f}')
axes[0].axvline(df['virality_score'].median(), color='green', linestyle='--',
                label=f'Median: {df["virality_score"].median():.4f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot by viral tier
if 'viral_tier' in df.columns:
    df.boxplot(column='virality_score', by='viral_tier', ax=axes[1])
    axes[1].set_xlabel('Viral Tier', fontsize=12)
    axes[1].set_ylabel('Virality Score', fontsize=12)
    axes[1].set_title('Virality Score by Tier', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove auto title

plt.tight_layout()
plt.savefig('02_virality_score.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_virality_score.png")
plt.close()

# ============================================================================
# 3. CORRELATION HEATMAP
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 10))

corr_cols = ['views', 'likes', 'comments', 'shares', 'engagement_rate', 
             'virality_score', 'length', 'hook_length_chars', 'caption_length_chars',
             'hashtag_count', 'mention_count', 'upload_hour', 'upload_dayofweek']
corr_matrix = df[corr_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_correlation_heatmap.png")
plt.close()

# ============================================================================
# 4. CATEGORY PERFORMANCE
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Category Analysis', fontsize=16, fontweight='bold')

# Main category counts
cat_counts = df['main_category'].value_counts().head(15)
axes[0, 0].barh(range(len(cat_counts)), cat_counts.values, color='steelblue')
axes[0, 0].set_yticks(range(len(cat_counts)))
axes[0, 0].set_yticklabels(cat_counts.index, fontsize=10)
axes[0, 0].set_xlabel('Number of Videos', fontsize=12)
axes[0, 0].set_title('Top 15 Categories by Video Count', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)

# Category avg views
cat_views = df.groupby('main_category')['views'].mean().sort_values(ascending=False).head(15)
axes[0, 1].barh(range(len(cat_views)), cat_views.values, color='coral')
axes[0, 1].set_yticks(range(len(cat_views)))
axes[0, 1].set_yticklabels(cat_views.index, fontsize=10)
axes[0, 1].set_xlabel('Average Views', fontsize=12)
axes[0, 1].set_title('Top 15 Categories by Avg Views', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# Category avg virality score
cat_virality = df.groupby('main_category')['virality_score'].mean().sort_values(ascending=False).head(15)
axes[1, 0].barh(range(len(cat_virality)), cat_virality.values, color='mediumseagreen')
axes[1, 0].set_yticks(range(len(cat_virality)))
axes[1, 0].set_yticklabels(cat_virality.index, fontsize=10)
axes[1, 0].set_xlabel('Average Virality Score', fontsize=12)
axes[1, 0].set_title('Top 15 Categories by Avg Virality', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# Engagement rate by category
cat_engagement = df.groupby('main_category')['engagement_rate'].mean().sort_values(ascending=False).head(15)
axes[1, 1].barh(range(len(cat_engagement)), cat_engagement.values, color='mediumpurple')
axes[1, 1].set_yticks(range(len(cat_engagement)))
axes[1, 1].set_yticklabels(cat_engagement.index, fontsize=10)
axes[1, 1].set_xlabel('Average Engagement Rate', fontsize=12)
axes[1, 1].set_title('Top 15 Categories by Engagement Rate', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('04_category_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_category_analysis.png")
plt.close()

# ============================================================================
# 5. TEMPORAL PATTERNS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Temporal Patterns Analysis', fontsize=16, fontweight='bold')

# Hour of day
hour_views = df.groupby('upload_hour')['views'].mean()
axes[0, 0].plot(hour_views.index, hour_views.values, marker='o', linewidth=2, markersize=6, color='blue')
axes[0, 0].fill_between(hour_views.index, hour_views.values, alpha=0.3)
axes[0, 0].set_xlabel('Hour of Day', fontsize=12)
axes[0, 0].set_ylabel('Average Views', fontsize=12)
axes[0, 0].set_title('Avg Views by Hour of Day', fontsize=14, fontweight='bold')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_xticks(range(0, 24, 2))

# Day of week
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_views = df.groupby('upload_dayofweek')['views'].mean()
axes[0, 1].bar(range(7), dow_views.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', 
                                                   '#96CEB4', '#FFEAA7', '#DFE6E9', '#74B9FF'])
axes[0, 1].set_xticks(range(7))
axes[0, 1].set_xticklabels(dow_names)
axes[0, 1].set_xlabel('Day of Week', fontsize=12)
axes[0, 1].set_ylabel('Average Views', fontsize=12)
axes[0, 1].set_title('Avg Views by Day of Week', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# Month
month_views = df.groupby('upload_month')['views'].mean()
axes[1, 0].plot(month_views.index, month_views.values, marker='s', linewidth=2, markersize=8, color='green')
axes[1, 0].fill_between(month_views.index, month_views.values, alpha=0.3)
axes[1, 0].set_xlabel('Month', fontsize=12)
axes[1, 0].set_ylabel('Average Views', fontsize=12)
axes[1, 0].set_title('Avg Views by Month', fontsize=14, fontweight='bold')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xticks(range(1, 13))

# Weekend vs Weekday
weekend_data = df.groupby('is_weekend')['views'].mean()
axes[1, 1].bar(['Weekday', 'Weekend'], weekend_data.values, color=['#3498db', '#e74c3c'])
axes[1, 1].set_ylabel('Average Views', fontsize=12)
axes[1, 1].set_title('Weekday vs Weekend Performance', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('05_temporal_patterns.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_temporal_patterns.png")
plt.close()

# ============================================================================
# 6. TEXT FEATURES
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Text Features Analysis', fontsize=16, fontweight='bold')

# Hook length distribution
axes[0, 0].hist(df['hook_length_chars'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Hook Length (characters)', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Hook Length Distribution', fontsize=12, fontweight='bold')
axes[0, 0].axvline(df['hook_length_chars'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Caption length distribution
axes[0, 1].hist(df['caption_length_chars'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Caption Length (characters)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Caption Length Distribution', fontsize=12, fontweight='bold')
axes[0, 1].axvline(df['caption_length_chars'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Hashtag count distribution
axes[0, 2].hist(df['hashtag_count'], bins=range(0, min(20, df['hashtag_count'].max()+1)), 
                color='lightgreen', edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel('Hashtag Count', fontsize=11)
axes[0, 2].set_ylabel('Frequency', fontsize=11)
axes[0, 2].set_title('Hashtag Count Distribution', fontsize=12, fontweight='bold')
axes[0, 2].grid(alpha=0.3)

# Hook length vs virality
axes[1, 0].scatter(df['hook_length_chars'], df['virality_score'], alpha=0.3, s=10)
axes[1, 0].set_xlabel('Hook Length (characters)', fontsize=11)
axes[1, 0].set_ylabel('Virality Score', fontsize=11)
axes[1, 0].set_title('Hook Length vs Virality', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Caption length vs virality
axes[1, 1].scatter(df['caption_length_chars'], df['virality_score'], alpha=0.3, s=10, color='coral')
axes[1, 1].set_xlabel('Caption Length (characters)', fontsize=11)
axes[1, 1].set_ylabel('Virality Score', fontsize=11)
axes[1, 1].set_title('Caption Length vs Virality', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

# Hashtag count vs views
hashtag_views = df.groupby('hashtag_count')['views'].mean().head(15)
axes[1, 2].plot(hashtag_views.index, hashtag_views.values, marker='o', linewidth=2, color='green')
axes[1, 2].set_xlabel('Hashtag Count', fontsize=11)
axes[1, 2].set_ylabel('Average Views', fontsize=11)
axes[1, 2].set_title('Hashtag Count vs Avg Views', fontsize=12, fontweight='bold')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('06_text_features.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_text_features.png")
plt.close()

# ============================================================================
# 7. VIDEO LENGTH ANALYSIS
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Video Length Analysis', fontsize=16, fontweight='bold')

# Length distribution
axes[0].hist(df['length'], bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Video Length (seconds)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Video Length Distribution', fontsize=14, fontweight='bold')
axes[0].axvline(df['length'].mean(), color='red', linestyle='--', label=f'Mean: {df["length"].mean():.1f}s')
axes[0].axvline(df['length'].median(), color='green', linestyle='--', label=f'Median: {df["length"].median():.1f}s')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Length category vs virality
if 'length_category' in df.columns:
    length_virality = df.groupby('length_category')['virality_score'].mean()
    axes[1].bar(range(len(length_virality)), length_virality.values, 
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(length_virality)])
    axes[1].set_xticks(range(len(length_virality)))
    axes[1].set_xticklabels(length_virality.index, rotation=45)
    axes[1].set_ylabel('Average Virality Score', fontsize=12)
    axes[1].set_title('Virality by Video Length Category', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('07_video_length.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_video_length.png")
plt.close()

print("\n" + "="*80)
print("✅ ALL VISUALIZATIONS CREATED!")
print("="*80)
print("\nGenerated files:")
print("  01_engagement_distributions.png")
print("  02_virality_score.png")
print("  03_correlation_heatmap.png")
print("  04_category_analysis.png")
print("  05_temporal_patterns.png")
print("  06_text_features.png")
print("  07_video_length.png")