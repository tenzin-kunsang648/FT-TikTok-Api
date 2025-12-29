"""
MODEL TRAINING PIPELINE - TikTok Virality Prediction

Purpose:
    Trains and evaluates multiple machine learning models to predict TikTok video virality.
    Compares model performance and selects best candidates for production deployment.
    
Tasks:
    1. Regression: Predict continuous virality scores for content optimization
    2. Classification: Categorize videos into viral tiers for decision-making
    
Models Trained:
    - XGBoost: Gradient boosting, typically best for tabular data
    - Random Forest: Ensemble of decision trees, robust and interpretable
    - LightGBM: Fast gradient boosting alternative, handles large datasets efficiently
    
Hyperparameter Optimization:
    - Uses Optuna for automated hyperparameter search
    - 50 trials per model to find optimal configuration
    - Validates on held-out validation set to prevent overfitting
    
Evaluation Strategy:
    - Regression: MAE (Mean Absolute Error) - interpretable error in original units
    - Classification: Macro F1-Score - balanced performance across imbalanced classes
    - All standard metrics tracked: RMSE, R², Accuracy, Precision, Recall
    
Output Organization:
    data/               → Train/val/test splits (moved from root)
    models/             → Trained model files (.pkl) + hyperparameters + metrics
    artifacts/          → Preprocessing objects (scaler, encoders, metadata)
    results/            → Model comparisons, visualizations, evaluation reports
    
Deployment Artifacts:
    - Best regression model for virality score prediction
    - Best classification model for viral tier categorization  
    - Preprocessing pipeline for new data
    - Feature importance analysis for model interpretation
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

N_TRIALS = 50                    # Optuna trials per model
OPTUNA_TIMEOUT = 1800            # 30 minutes max per model
RANDOM_SEED = 42
EARLY_STOPPING_ROUNDS = 50

# Primary optimization metrics (what Optuna minimizes/maximizes)
REGRESSION_PRIMARY_METRIC = 'MAE'           # Mean Absolute Error (interpretable)
CLASSIFICATION_PRIMARY_METRIC = 'Macro_F1'  # Balanced F1 across classes

np.random.seed(RANDOM_SEED)

print("="*100)
print("MODEL TRAINING PIPELINE - TikTok Virality Prediction")
print("="*100)
print(f"\nConfiguration:")
print(f"  Optuna trials per model: {N_TRIALS}")
print(f"  Regression metric: {REGRESSION_PRIMARY_METRIC}")
print(f"  Classification metric: {CLASSIFICATION_PRIMARY_METRIC}")
print(f"  Random seed: {RANDOM_SEED}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_folder_structure():
    """Create organized directory structure for all outputs"""
    folders = [
        'data/train', 'data/val', 'data/test',
        'models/regression/xgboost', 'models/regression/random_forest', 'models/regression/lightgbm',
        'models/classification/xgboost', 'models/classification/random_forest', 'models/classification/lightgbm',
        'artifacts', 'results/feature_importance', 'results/confusion_matrices', 'logs',
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    return True

def evaluate_regression_model(y_true, y_pred, model_name, dataset_name):
    """
    Calculate comprehensive regression metrics
    
    Metrics:
        MAE: Average absolute error (same units as target)
        RMSE: Root mean squared error (penalizes large errors more)
        R²: Proportion of variance explained (0-1, higher is better)
    """
    return {
        'Model': model_name,
        'Dataset': dataset_name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
    }

def evaluate_classification_model(y_true, y_pred, model_name, dataset_name):
    """
    Calculate comprehensive classification metrics
    
    Metrics:
        Accuracy: Overall correctness
        Macro F1: Balanced F1 score (treats all classes equally)
        Weighted F1: F1 weighted by class frequency
        Precision/Recall: Per-class performance indicators
    """
    return {
        'Model': model_name,
        'Dataset': dataset_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Macro_F1': f1_score(y_true, y_pred, average='macro'),
        'Weighted_F1': f1_score(y_true, y_pred, average='weighted'),
        'Macro_Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro_Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }

def plot_feature_importance(model, feature_names, model_name, task_type, top_n=20):
    """Generate and save feature importance visualization"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title(f'Top {top_n} Important Features - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    filename = f'results/feature_importance/{task_type}_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

def plot_confusion_matrix(y_true, y_pred, model_name, dataset_name, class_labels):
    """Generate and save confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix - {model_name} ({dataset_name})', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    filename = f'results/confusion_matrices/{model_name.lower().replace(" ", "_")}_{dataset_name.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# ============================================================================

def optimize_xgboost_regression(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for XGBoost regression hyperparameter tuning"""
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'random_state': RANDOM_SEED,
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    
    return mean_absolute_error(y_val, y_pred)

def optimize_xgboost_classification(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for XGBoost classification hyperparameter tuning"""
    params = {
        'objective': 'multi:softprob',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'random_state': RANDOM_SEED,
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    
    return f1_score(y_val, y_pred, average='macro')

def optimize_random_forest_regression(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for Random Forest regression hyperparameter tuning"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
    }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    return mean_absolute_error(y_val, y_pred)

def optimize_random_forest_classification(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for Random Forest classification hyperparameter tuning"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    return f1_score(y_val, y_pred, average='macro')

def optimize_lightgbm_regression(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for LightGBM regression hyperparameter tuning"""
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'random_state': RANDOM_SEED,
        'verbose': -1,
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    y_pred = model.predict(X_val)
    
    return mean_absolute_error(y_val, y_pred)

def optimize_lightgbm_classification(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for LightGBM classification hyperparameter tuning"""
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'random_state': RANDOM_SEED,
        'verbose': -1,
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    y_pred = model.predict(X_val)
    
    return f1_score(y_val, y_pred, average='macro')

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ------------------------------------------------------------------------
    # Setup: Create Directory Structure
    # ------------------------------------------------------------------------
    
    print("\n" + "="*100)
    print("SETUP: CREATING DIRECTORY STRUCTURE")
    print("="*100)
    
    create_folder_structure()
    print("✓ Created organized folder structure for outputs")
    
    # ------------------------------------------------------------------------
    # Load Processed Data
    # ------------------------------------------------------------------------
    
    print("\n" + "="*100)
    print("LOADING PROCESSED DATA")
    print("="*100)
    
    # Find most recent processed datasets
    train_files = glob.glob('data_train_*.csv')
    if not train_files:
        print("\n✗ ERROR: No processed data files found!")
        print("  Please run feature_engineering.py first to generate datasets")
        return
    
    latest_file = max(train_files)
    data_timestamp = latest_file.replace('data_train_', '').replace('.csv', '')
    
    print(f"\nLoading datasets with timestamp: {data_timestamp}")
    
    # Load train/val/test splits
    train_df = pd.read_csv(f'data_train_{data_timestamp}.csv')
    val_df = pd.read_csv(f'data_val_{data_timestamp}.csv')
    test_df = pd.read_csv(f'data_test_{data_timestamp}.csv')
    
    print(f"  Train: {len(train_df):6,} rows")
    print(f"  Val:   {len(val_df):6,} rows")
    print(f"  Test:  {len(test_df):6,} rows")
    
    # Load feature metadata and configuration
    with open(f'feature_metadata_{data_timestamp}.json', 'r') as f:
        metadata = json.load(f)
    
    # Load encoders for later use
    with open(f'encoders_{data_timestamp}.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    feature_names = metadata['features']['all_features']
    target_reg = metadata['targets']['regression']
    target_clf = metadata['targets']['classification']
    label_encoder = encoders['label_encoder']
    
    print(f"\n✓ Configuration loaded:")
    print(f"  Input features: {len(feature_names)}")
    print(f"  Regression target: {target_reg}")
    print(f"  Classification target: {target_clf}")
    print(f"  Target transformation: {metadata['transformations']['target_transform']}")
    
    # Prepare feature matrices and targets
    X_train = train_df[feature_names]
    y_train_reg = train_df[target_reg]
    y_train_clf = train_df[target_clf]
    
    X_val = val_df[feature_names]
    y_val_reg = val_df[target_reg]
    y_val_clf = val_df[target_clf]
    
    X_test = test_df[feature_names]
    y_test_reg = test_df[target_reg]
    y_test_clf = test_df[target_clf]
    
    # Keep original scale targets for final evaluation
    y_test_reg_original = test_df[metadata['targets']['regression_original']]
    
    # Move processed files to organized folders
    print("\n✓ Organizing files into folder structure...")
    os.rename(f'data_train_{data_timestamp}.csv', f'data/train/data_train_{data_timestamp}.csv')
    os.rename(f'data_val_{data_timestamp}.csv', f'data/val/data_val_{data_timestamp}.csv')
    os.rename(f'data_test_{data_timestamp}.csv', f'data/test/data_test_{data_timestamp}.csv')
    
    for artifact_file in [f'scaler_{data_timestamp}.pkl', f'encoders_{data_timestamp}.pkl', 
                          f'feature_metadata_{data_timestamp}.json']:
        if os.path.exists(artifact_file):
            os.rename(artifact_file, f'artifacts/{artifact_file}')
    
    print("  ✓ Moved data files to data/train/, data/val/, data/test/")
    print("  ✓ Moved artifacts to artifacts/")
    
    # ------------------------------------------------------------------------
    # Train Regression Models
    # ------------------------------------------------------------------------
    
    print("\n" + "="*100)
    print("TRAINING REGRESSION MODELS")
    print("Task: Predict continuous virality scores")
    print("="*100)
    
    regression_results = []
    
    # XGBoost Regression
    print("\n" + "-"*100)
    print("MODEL 1/3: XGBoost Regression")
    print("-"*100)
    print(f"Optimizing hyperparameters with Optuna ({N_TRIALS} trials)...")
    print(f"Objective: Minimize {REGRESSION_PRIMARY_METRIC} on validation set")
    
    study_xgb_reg = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_SEED))
    study_xgb_reg.optimize(
        lambda trial: optimize_xgboost_regression(trial, X_train, y_train_reg, X_val, y_val_reg),
        n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True
    )
    
    best_params_xgb_reg = study_xgb_reg.best_params
    best_params_xgb_reg.update({'objective': 'reg:squarederror', 'random_state': RANDOM_SEED})
    
    print(f"\n✓ Hyperparameter optimization complete")
    print(f"  Best {REGRESSION_PRIMARY_METRIC}: {study_xgb_reg.best_value:.4f}")
    
    print("\nTraining final model with optimized hyperparameters...")
    model_xgb_reg = xgb.XGBRegressor(**best_params_xgb_reg)
    model_xgb_reg.fit(X_train, y_train_reg)
    
    # Evaluate on validation and test sets
    # Note: Predictions are on log scale, inverse transform for original scale evaluation
    y_pred_val_log = model_xgb_reg.predict(X_val)
    y_pred_test_log = model_xgb_reg.predict(X_test)
    
    # Inverse transform: log scale → original scale
    y_pred_val = np.expm1(y_pred_val_log)
    y_pred_test = np.expm1(y_pred_test_log)
    
    # Evaluate on original scale for interpretability
    y_val_original = val_df[metadata['targets']['regression_original']]
    
    metrics_val = evaluate_regression_model(y_val_original, y_pred_val, 'XGBoost', 'Validation')
    metrics_test = evaluate_regression_model(y_test_reg_original, y_pred_test, 'XGBoost', 'Test')
    
    regression_results.append(metrics_val)
    regression_results.append(metrics_test)
    
    print(f"\n✓ Model evaluation on test set (original scale):")
    print(f"  MAE:  {metrics_test['MAE']:.4f} (average error)")
    print(f"  RMSE: {metrics_test['RMSE']:.4f}")
    print(f"  R²:   {metrics_test['R2']:.4f} (variance explained)")
    
    # Save model and artifacts
    with open('models/regression/xgboost/model.pkl', 'wb') as f:
        pickle.dump(model_xgb_reg, f)
    
    with open('models/regression/xgboost/best_params.json', 'w') as f:
        json.dump(best_params_xgb_reg, f, indent=2)
    
    with open('models/regression/xgboost/metrics.json', 'w') as f:
        json.dump({'validation': metrics_val, 'test': metrics_test}, f, indent=2)
    
    # Feature importance analysis
    importance_df = plot_feature_importance(model_xgb_reg, feature_names, 'XGBoost', 'regression')
    if importance_df is not None:
        print(f"\n✓ Top 5 most important features:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")
    
    print("\n✓ XGBoost regression model saved to models/regression/xgboost/")
    
    # Random Forest Regression
    print("\n" + "-"*100)
    print("MODEL 2/3: Random Forest Regression")
    print("-"*100)
    print(f"Optimizing hyperparameters with Optuna ({N_TRIALS} trials)...")
    
    study_rf_reg = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_SEED))
    study_rf_reg.optimize(
        lambda trial: optimize_random_forest_regression(trial, X_train, y_train_reg, X_val, y_val_reg),
        n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True
    )
    
    best_params_rf_reg = study_rf_reg.best_params
    best_params_rf_reg.update({'random_state': RANDOM_SEED, 'n_jobs': -1})
    
    print(f"\n✓ Hyperparameter optimization complete")
    print(f"  Best {REGRESSION_PRIMARY_METRIC}: {study_rf_reg.best_value:.4f}")
    
    model_rf_reg = RandomForestRegressor(**best_params_rf_reg)
    model_rf_reg.fit(X_train, y_train_reg)
    
    y_pred_val = np.expm1(model_rf_reg.predict(X_val))
    y_pred_test = np.expm1(model_rf_reg.predict(X_test))
    
    metrics_val = evaluate_regression_model(y_val_original, y_pred_val, 'Random Forest', 'Validation')
    metrics_test = evaluate_regression_model(y_test_reg_original, y_pred_test, 'Random Forest', 'Test')
    
    regression_results.append(metrics_val)
    regression_results.append(metrics_test)
    
    print(f"\n✓ Test set performance: MAE={metrics_test['MAE']:.4f}, RMSE={metrics_test['RMSE']:.4f}, R²={metrics_test['R2']:.4f}")
    
    with open('models/regression/random_forest/model.pkl', 'wb') as f:
        pickle.dump(model_rf_reg, f)
    with open('models/regression/random_forest/best_params.json', 'w') as f:
        json.dump(best_params_rf_reg, f, indent=2)
    with open('models/regression/random_forest/metrics.json', 'w') as f:
        json.dump({'validation': metrics_val, 'test': metrics_test}, f, indent=2)
    
    plot_feature_importance(model_rf_reg, feature_names, 'Random Forest', 'regression')
    print("\n✓ Random Forest regression model saved")
    
    # LightGBM Regression
    print("\n" + "-"*100)
    print("MODEL 3/3: LightGBM Regression")
    print("-"*100)
    print(f"Optimizing hyperparameters with Optuna ({N_TRIALS} trials)...")
    
    study_lgb_reg = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_SEED))
    study_lgb_reg.optimize(
        lambda trial: optimize_lightgbm_regression(trial, X_train, y_train_reg, X_val, y_val_reg),
        n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True
    )
    
    best_params_lgb_reg = study_lgb_reg.best_params
    best_params_lgb_reg.update({'objective': 'regression', 'random_state': RANDOM_SEED, 'verbose': -1})
    
    print(f"\n✓ Hyperparameter optimization complete")
    print(f"  Best {REGRESSION_PRIMARY_METRIC}: {study_lgb_reg.best_value:.4f}")
    
    model_lgb_reg = lgb.LGBMRegressor(**best_params_lgb_reg)
    model_lgb_reg.fit(X_train, y_train_reg)
    
    y_pred_val = np.expm1(model_lgb_reg.predict(X_val))
    y_pred_test = np.expm1(model_lgb_reg.predict(X_test))
    
    metrics_val = evaluate_regression_model(y_val_original, y_pred_val, 'LightGBM', 'Validation')
    metrics_test = evaluate_regression_model(y_test_reg_original, y_pred_test, 'LightGBM', 'Test')
    
    regression_results.append(metrics_val)
    regression_results.append(metrics_test)
    
    print(f"\n✓ Test set performance: MAE={metrics_test['MAE']:.4f}, RMSE={metrics_test['RMSE']:.4f}, R²={metrics_test['R2']:.4f}")
    
    with open('models/regression/lightgbm/model.pkl', 'wb') as f:
        pickle.dump(model_lgb_reg, f)
    with open('models/regression/lightgbm/best_params.json', 'w') as f:
        json.dump(best_params_lgb_reg, f, indent=2)
    with open('models/regression/lightgbm/metrics.json', 'w') as f:
        json.dump({'validation': metrics_val, 'test': metrics_test}, f, indent=2)
    
    plot_feature_importance(model_lgb_reg, feature_names, 'LightGBM', 'regression')
    print("\n✓ LightGBM regression model saved")
    
    # ------------------------------------------------------------------------
    # Train Classification Models
    # ------------------------------------------------------------------------
    
    print("\n" + "="*100)
    print("TRAINING CLASSIFICATION MODELS")
    print("Task: Categorize videos into viral tiers (low/viral/mega)")
    print("="*100)
    
    classification_results = []
    class_labels = label_encoder.classes_
    
    # XGBoost Classification
    print("\n" + "-"*100)
    print("MODEL 1/3: XGBoost Classification")
    print("-"*100)
    print(f"Optimizing hyperparameters with Optuna ({N_TRIALS} trials)...")
    print(f"Objective: Maximize {CLASSIFICATION_PRIMARY_METRIC} on validation set")
    
    study_xgb_clf = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_SEED))
    study_xgb_clf.optimize(
        lambda trial: optimize_xgboost_classification(trial, X_train, y_train_clf, X_val, y_val_clf),
        n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True
    )
    
    best_params_xgb_clf = study_xgb_clf.best_params
    best_params_xgb_clf.update({'objective': 'multi:softprob', 'random_state': RANDOM_SEED})
    
    print(f"\n✓ Hyperparameter optimization complete")
    print(f"  Best {CLASSIFICATION_PRIMARY_METRIC}: {study_xgb_clf.best_value:.4f}")
    
    model_xgb_clf = xgb.XGBClassifier(**best_params_xgb_clf)
    model_xgb_clf.fit(X_train, y_train_clf)
    
    y_pred_val = model_xgb_clf.predict(X_val)
    y_pred_test = model_xgb_clf.predict(X_test)
    
    metrics_val = evaluate_classification_model(y_val_clf, y_pred_val, 'XGBoost', 'Validation')
    metrics_test = evaluate_classification_model(y_test_clf, y_pred_test, 'XGBoost', 'Test')
    
    classification_results.append(metrics_val)
    classification_results.append(metrics_test)
    
    print(f"\n✓ Test set performance:")
    print(f"  Accuracy:  {metrics_test['Accuracy']:.4f}")
    print(f"  Macro F1:  {metrics_test['Macro_F1']:.4f}")
    
    with open('models/classification/xgboost/model.pkl', 'wb') as f:
        pickle.dump(model_xgb_clf, f)
    with open('models/classification/xgboost/best_params.json', 'w') as f:
        json.dump(best_params_xgb_clf, f, indent=2)
    with open('models/classification/xgboost/metrics.json', 'w') as f:
        json.dump({'validation': metrics_val, 'test': metrics_test}, f, indent=2)
    
    plot_feature_importance(model_xgb_clf, feature_names, 'XGBoost', 'classification')
    plot_confusion_matrix(y_test_clf, y_pred_test, 'XGBoost', 'Test', class_labels)
    
    print("\n✓ XGBoost classification model saved")
    
    # Random Forest Classification
    print("\n" + "-"*100)
    print("MODEL 2/3: Random Forest Classification")
    print("-"*100)
    print(f"Optimizing hyperparameters with Optuna ({N_TRIALS} trials)...")
    
    study_rf_clf = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_SEED))
    study_rf_clf.optimize(
        lambda trial: optimize_random_forest_classification(trial, X_train, y_train_clf, X_val, y_val_clf),
        n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True
    )
    
    best_params_rf_clf = study_rf_clf.best_params
    best_params_rf_clf.update({'random_state': RANDOM_SEED, 'n_jobs': -1})
    
    print(f"\n✓ Hyperparameter optimization complete")
    print(f"  Best {CLASSIFICATION_PRIMARY_METRIC}: {study_rf_clf.best_value:.4f}")
    
    model_rf_clf = RandomForestClassifier(**best_params_rf_clf)
    model_rf_clf.fit(X_train, y_train_clf)
    
    y_pred_val = model_rf_clf.predict(X_val)
    y_pred_test = model_rf_clf.predict(X_test)
    
    metrics_val = evaluate_classification_model(y_val_clf, y_pred_val, 'Random Forest', 'Validation')
    metrics_test = evaluate_classification_model(y_test_clf, y_pred_test, 'Random Forest', 'Test')
    
    classification_results.append(metrics_val)
    classification_results.append(metrics_test)
    
    print(f"\n✓ Test set performance: Accuracy={metrics_test['Accuracy']:.4f}, Macro F1={metrics_test['Macro_F1']:.4f}")
    
    with open('models/classification/random_forest/model.pkl', 'wb') as f:
        pickle.dump(model_rf_clf, f)
    with open('models/classification/random_forest/best_params.json', 'w') as f:
        json.dump(best_params_rf_clf, f, indent=2)
    with open('models/classification/random_forest/metrics.json', 'w') as f:
        json.dump({'validation': metrics_val, 'test': metrics_test}, f, indent=2)
    
    plot_feature_importance(model_rf_clf, feature_names, 'Random Forest', 'classification')
    plot_confusion_matrix(y_test_clf, y_pred_test, 'Random Forest', 'Test', class_labels)
    
    print("\n✓ Random Forest classification model saved")
    
    # LightGBM Classification
    print("\n" + "-"*100)
    print("MODEL 3/3: LightGBM Classification")
    print("-"*100)
    print(f"Optimizing hyperparameters with Optuna ({N_TRIALS} trials)...")
    
    study_lgb_clf = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_SEED))
    study_lgb_clf.optimize(
        lambda trial: optimize_lightgbm_classification(trial, X_train, y_train_clf, X_val, y_val_clf),
        n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True
    )
    
    best_params_lgb_clf = study_lgb_clf.best_params
    best_params_lgb_clf.update({'objective': 'multiclass', 'random_state': RANDOM_SEED, 'verbose': -1})
    
    print(f"\n✓ Hyperparameter optimization complete")
    print(f"  Best {CLASSIFICATION_PRIMARY_METRIC}: {study_lgb_clf.best_value:.4f}")
    
    model_lgb_clf = lgb.LGBMClassifier(**best_params_lgb_clf)
    model_lgb_clf.fit(X_train, y_train_clf)
    
    y_pred_val = model_lgb_clf.predict(X_val)
    y_pred_test = model_lgb_clf.predict(X_test)
    
    metrics_val = evaluate_classification_model(y_val_clf, y_pred_val, 'LightGBM', 'Validation')
    metrics_test = evaluate_classification_model(y_test_clf, y_pred_test, 'LightGBM', 'Test')
    
    classification_results.append(metrics_val)
    classification_results.append(metrics_test)
    
    print(f"\n✓ Test set performance: Accuracy={metrics_test['Accuracy']:.4f}, Macro F1={metrics_test['Macro_F1']:.4f}")
    
    with open('models/classification/lightgbm/model.pkl', 'wb') as f:
        pickle.dump(model_lgb_clf, f)
    with open('models/classification/lightgbm/best_params.json', 'w') as f:
        json.dump(best_params_lgb_clf, f, indent=2)
    with open('models/classification/lightgbm/metrics.json', 'w') as f:
        json.dump({'validation': metrics_val, 'test': metrics_test}, f, indent=2)
    
    plot_feature_importance(model_lgb_clf, feature_names, 'LightGBM', 'classification')
    plot_confusion_matrix(y_test_clf, y_pred_test, 'LightGBM', 'Test', class_labels)
    
    print("\n✓ LightGBM classification model saved")
    
    # ------------------------------------------------------------------------
    # Model Comparison and Selection
    # ------------------------------------------------------------------------
    
    print("\n" + "="*100)
    print("MODEL COMPARISON AND SELECTION")
    print("="*100)
    
    # Regression model comparison
    print("\n" + "-"*100)
    print("REGRESSION MODELS (Test Set Performance)")
    print("-"*100)
    
    regression_df = pd.DataFrame(regression_results)
    regression_test = regression_df[regression_df['Dataset'] == 'Test'].copy()
    regression_test = regression_test.sort_values(REGRESSION_PRIMARY_METRIC)
    
    print(f"\nRanked by {REGRESSION_PRIMARY_METRIC}:")
    print(regression_test.to_string(index=False))
    
    regression_df.to_csv('results/model_comparison_regression.csv', index=False)
    
    best_reg_model = regression_test.iloc[0]
    print(f"\n🏆 BEST REGRESSION MODEL: {best_reg_model['Model']}")
    print(f"   Test MAE: {best_reg_model['MAE']:.4f}")
    print(f"   Test R²:  {best_reg_model['R2']:.4f}")
    
    # Classification model comparison
    print("\n" + "-"*100)
    print("CLASSIFICATION MODELS (Test Set Performance)")
    print("-"*100)
    
    classification_df = pd.DataFrame(classification_results)
    classification_test = classification_df[classification_df['Dataset'] == 'Test'].copy()
    classification_test = classification_test.sort_values(CLASSIFICATION_PRIMARY_METRIC, ascending=False)
    
    print(f"\nRanked by {CLASSIFICATION_PRIMARY_METRIC}:")
    print(classification_test.to_string(index=False))
    
    classification_df.to_csv('results/model_comparison_classification.csv', index=False)
    
    best_clf_model = classification_test.iloc[0]
    print(f"\n🏆 BEST CLASSIFICATION MODEL: {best_clf_model['Model']}")
    print(f"   Test Macro F1: {best_clf_model['Macro_F1']:.4f}")
    print(f"   Test Accuracy: {best_clf_model['Accuracy']:.4f}")
    
    # ------------------------------------------------------------------------
    # Generate Evaluation Report
    # ------------------------------------------------------------------------
    
    print("\n" + "="*100)
    print("GENERATING EVALUATION REPORT")
    print("="*100)
    
    report_file = f'results/EVALUATION_REPORT_{run_timestamp}.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("MODEL EVALUATION REPORT - TikTok Virality Prediction\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data timestamp: {data_timestamp}\n")
        f.write("="*100 + "\n\n")
        
        f.write("REGRESSION TASK: Predicting Virality Scores\n")
        f.write("-"*100 + "\n")
        f.write(f"Purpose: Predict continuous engagement metric for content optimization\n")
        f.write(f"Primary Metric: {REGRESSION_PRIMARY_METRIC}\n\n")
        f.write(regression_test.to_string(index=False) + "\n\n")
        f.write(f"WINNER: {best_reg_model['Model']}\n")
        f.write(f"  Test MAE:  {best_reg_model['MAE']:.4f}\n")
        f.write(f"  Test RMSE: {best_reg_model['RMSE']:.4f}\n")
        f.write(f"  Test R²:   {best_reg_model['R2']:.4f}\n\n")
        
        f.write("="*100 + "\n\n")
        
        f.write("CLASSIFICATION TASK: Predicting Viral Tiers\n")
        f.write("-"*100 + "\n")
        f.write(f"Purpose: Categorize content into actionable viral tiers\n")
        f.write(f"Classes: low_viral (bottom 20%), viral (20-95%), mega_viral (top 5%)\n")
        f.write(f"Primary Metric: {CLASSIFICATION_PRIMARY_METRIC}\n\n")
        f.write(classification_test.to_string(index=False) + "\n\n")
        f.write(f"WINNER: {best_clf_model['Model']}\n")
        f.write(f"  Test Macro F1: {best_clf_model['Macro_F1']:.4f}\n")
        f.write(f"  Test Accuracy: {best_clf_model['Accuracy']:.4f}\n\n")
        
        f.write("="*100 + "\n\n")
        
        f.write("DEPLOYMENT RECOMMENDATIONS\n")
        f.write("-"*100 + "\n\n")
        f.write("For AWS Lambda deployment, package:\n\n")
        
        reg_model_path = f"models/regression/{best_reg_model['Model'].lower().replace(' ', '_')}/model.pkl"
        clf_model_path = f"models/classification/{best_clf_model['Model'].lower().replace(' ', '_')}/model.pkl"
        
        f.write(f"Models:\n")
        f.write(f"  - {reg_model_path}\n")
        f.write(f"  - {clf_model_path}\n\n")
        
        f.write(f"Preprocessing Artifacts:\n")
        f.write(f"  - artifacts/scaler_{data_timestamp}.pkl\n")
        f.write(f"  - artifacts/encoders_{data_timestamp}.pkl\n")
        f.write(f"  - artifacts/feature_metadata_{data_timestamp}.json\n\n")
        
        f.write("Inference Steps:\n")
        f.write("  1. Load raw video metadata\n")
        f.write("  2. Engineer features (use feature_engineering.py logic)\n")
        f.write("  3. Apply scaler to normalize features\n")
        f.write("  4. Encode categories using saved encoders\n")
        f.write("  5. Predict on log scale\n")
        f.write("  6. Inverse transform predictions: np.expm1(pred_log)\n")
        f.write("  7. Decode classification labels\n\n")
        
        f.write("="*100 + "\n")
    
    print(f"✓ Comprehensive evaluation report saved: {report_file}")
    
    # ------------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------------
    
    print("\n" + "="*100)
    print("TRAINING PIPELINE COMPLETE")
    print("="*100)
    
    print(f"\n✓ Successfully trained and evaluated 6 models:")
    print(f"  - 3 regression models (XGBoost, Random Forest, LightGBM)")
    print(f"  - 3 classification models (XGBoost, Random Forest, LightGBM)")
    
    print(f"\n✓ Best performing models:")
    print(f"  Regression:     {best_reg_model['Model']:15s} (MAE: {best_reg_model['MAE']:.4f})")
    print(f"  Classification: {best_clf_model['Model']:15s} (Macro F1: {best_clf_model['Macro_F1']:.4f})")
    
    print(f"\n✓ All artifacts organized in folder structure:")
    print(f"  data/          → Train/validation/test datasets")
    print(f"  models/        → Trained models with hyperparameters and metrics")
    print(f"  artifacts/     → Preprocessing objects (scaler, encoders, metadata)")
    print(f"  results/       → Model comparisons, feature importance plots, confusion matrices")
    
    print(f"\n✓ Ready for deployment:")
    print(f"  Next step: Create inference script (predict.py)")
    print(f"  Then: Package for AWS Lambda deployment")
    
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    main()