"""
Phase 3: Hyperparameter Tuning for Selected Models
Performs grid search to find optimal hyperparameters for top-performing models
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import config
from utils.data_loader import load_raw_data, encode_target, prepare_structured_features
from utils.evaluator import evaluate_model

def get_param_grids():
    """
    Define hyperparameter grids for tuning
    """
    # Simplified parameter grids for faster execution
    param_grids = {
        'rf': {
            'n_estimators': [100, 200],
            'max_depth': [20, 30],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt']
        },
        'gb': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5],
            'subsample': [0.9, 1.0]
        },
        'xgb': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'min_child_weight': [1, 3],
            'subsample': [0.9, 1.0]
        }
    }
    
    return param_grids

def tune_model(model_name, model_class, param_grid, X_train, y_train, X_test, y_test):
    """
    Perform grid search for a single model
    """
    print(f"\n{'='*80}")
    print(f"Tuning {model_name.upper()} Model")
    print(f"{'='*80}")
    
    # Initialize model
    if model_name == 'xgb':
        base_model = model_class(random_state=config.RANDOM_SEED, eval_metric='logloss')
    else:
        base_model = model_class(random_state=config.RANDOM_SEED)
    
    # Perform grid search
    print(f"\nPerforming grid search with {len(param_grid)} parameters...")
    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    print(f"\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV ROC-AUC: {best_cv_score:.4f}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_proba)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return best_model, best_params, best_cv_score, metrics

def main():
    print(f"{'='*80}")
    print("Phase 3: Hyperparameter Tuning")
    print(f"{'='*80}")
    
    # Load data
    print("\nLoading data...")
    
    # Load train/test indices
    train_idx = np.load(config.SPLITS_DIR / "train_indices.npy")
    test_idx = np.load(config.SPLITS_DIR / "test_indices.npy")
    
    # Load and prepare data
    df = load_raw_data()
    df = encode_target(df)
    df, feature_cols = prepare_structured_features(df)
    
    X = df[feature_cols]
    y = df['target']
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Get parameter grids
    param_grids = get_param_grids()
    
    # Define models to tune
    models_to_tune = {
        'rf': RandomForestClassifier,
        'gb': GradientBoostingClassifier,
        'xgb': XGBClassifier
    }
    
    # Store results
    results = []
    tuned_models = {}
    
    # Tune each model
    for model_name, model_class in models_to_tune.items():
        best_model, best_params, best_cv_score, test_metrics = tune_model(
            model_name, model_class, param_grids[model_name],
            X_train, y_train, X_test, y_test
        )
        
        # Save tuned model
        config.ensure_dir(config.MODELS_DIR / "phase3")
        model_path = config.MODELS_DIR / "phase3" / f"{model_name}_tuned_model.joblib"
        joblib.dump(best_model, model_path)
        print(f"\nTuned model saved to: {model_path}")
        
        # Save best parameters
        params_path = config.MODELS_DIR / "phase3" / f"{model_name}_best_params.pkl"
        with open(params_path, 'wb') as f:
            pickle.dump(best_params, f)
        
        tuned_models[model_name] = best_model
        
        # Store results
        results.append({
            'model': model_name.upper(),
            'cv_roc_auc': best_cv_score,
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1_score'],
            'test_roc_auc': test_metrics['roc_auc']
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Compare with baseline (Phase 0)
    print(f"\n{'='*80}")
    print("Comparison with Baseline (Phase 0)")
    print(f"{'='*80}")
    
    baseline_models = ['rf', 'gb', 'xgb']
    for model_name in baseline_models:
        baseline_path = config.MODELS_DIR / "phase0" / f"{model_name}_model.joblib"
        if baseline_path.exists():
            baseline_model = joblib.load(baseline_path)
            y_pred_baseline = baseline_model.predict(X_test)
            y_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]
            baseline_metrics = evaluate_model(y_test, y_pred_baseline, y_proba_baseline)
            
            tuned_metrics = results_df[results_df['model'] == model_name.upper()].iloc[0]
            
            print(f"\n{model_name.upper()}:")
            print(f"  Baseline ROC-AUC: {baseline_metrics['roc_auc']:.4f}")
            print(f"  Tuned ROC-AUC:    {tuned_metrics['test_roc_auc']:.4f}")
            print(f"  Improvement:      {(tuned_metrics['test_roc_auc'] - baseline_metrics['roc_auc']) / baseline_metrics['roc_auc'] * 100:+.2f}%")
    
    # Save results
    config.ensure_dir(config.TABLES_DIR)
    output_file = config.TABLES_DIR / "table_4_3_hyperparameter_tuning.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_file}")
    
    print(f"\n{'='*80}")
    print("Phase 3 completed successfully!")
    print(f"{'='*80}")
    
    return results_df, tuned_models

if __name__ == "__main__":
    results_df, tuned_models = main()
