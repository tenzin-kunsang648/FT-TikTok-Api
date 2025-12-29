"""
🔍 DIAGNOSTIC: Understanding why MAPE is 264-338%
"""
import pandas as pd
import numpy as np

print("="*80)
print("🔍 DIAGNOSING HIGH MAPE SCORES")
print("="*80)

# Check what data you currently have
import glob

train_files = glob.glob('data/train/data_train_*.csv')
if not train_files:
    print("❌ No train files in data/train/")
    train_files = glob.glob('data_train_*.csv')
    if train_files:
        print(f"✅ Found in root: {train_files[0]}")

if train_files:
    df = pd.read_csv(train_files[0])
    
    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"\n📋 Columns: {list(df.columns[:15])}")
    
    if 'virality_score' in df.columns:
        print("\n🎯 VIRALITY_SCORE ANALYSIS:")
        vs = df['virality_score']
        print(f"   Count:  {vs.count():,}")
        print(f"   Mean:   {vs.mean():.6f}")
        print(f"   Std:    {vs.std():.6f}")
        print(f"   Min:    {vs.min():.6f}")
        print(f"   25%:    {vs.quantile(0.25):.6f}")
        print(f"   Median: {vs.median():.6f}")
        print(f"   75%:    {vs.quantile(0.75):.6f}")
        print(f"   Max:    {vs.max():.6f}")
        
        print("\n❓ WAS IT SCALED?")
        if abs(vs.mean()) < 0.2 and 0.8 < vs.std() < 1.2:
            print("   🚨 YES! Virality score was StandardScaled!")
            print("   This is WRONG - targets should NEVER be scaled!")
            print("   This explains the terrible MAPE scores!")
        else:
            print("   ✅ No, looks normal")
        
        print("\n📊 Sample values:")
        print(vs.head(20).values)
    
    # Check if features were scaled
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n🔍 CHECKING ALL NUMERIC COLUMNS:")
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        print(f"   {col:30s}: mean={mean:8.4f}, std={std:8.4f}", end='')
        if abs(mean) < 0.1 and 0.8 < std < 1.2:
            print(" ← SCALED")
        else:
            print()
    
    print("\n💡 SOLUTION:")
    print("   Re-run feature_engineering.py with corrected script")
    print("   Make sure virality_score is NOT in the scaled features!")
    
else:
    print("❌ No data files found at all!")
    print("   You need to run feature_engineering.py first")