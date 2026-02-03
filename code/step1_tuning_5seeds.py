"""
Step 1: Hyperparameter Tuning with 5 Seeds
Run tuning 5 times and collect best parameters for RF, GB, XGB
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

import config
from experiment_50iterations import split_data

def tune_rf(X_train, y_train, seed):
    """Tune RandomForest"""
    print(f"\n  Tuning RF (seed={seed})...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print(f"    Best CV score: {grid_search.best_score_:.4f}")
    print(f"    Best params: {grid_search.best_params_}")
    
    return grid_search.best_params_, grid_search.best_score_

def tune_gb(X_train, y_train, seed):
    """Tune GradientBoosting"""
    print(f"\n  Tuning GB (seed={seed})...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }
    
    gb = GradientBoostingClassifier(random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    grid_search = GridSearchCV(
        gb, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print(f"    Best CV score: {grid_search.best_score_:.4f}")
    print(f"    Best params: {grid_search.best_params_}")
    
    return grid_search.best_params_, grid_search.best_score_

def tune_xgb(X_train, y_train, seed):
    """Tune XGBoost"""
    print(f"\n  Tuning XGB (seed={seed})...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_child_weight': [1, 3]
    }
    
    xgb = XGBClassifier(random_state=seed, eval_metric='logloss')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    grid_search = GridSearchCV(
        xgb, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print(f"    Best CV score: {grid_search.best_score_:.4f}")
    print(f"    Best params: {grid_search.best_params_}")
    
    return grid_search.best_params_, grid_search.best_score_

def run_tuning_one_seed(seed):
    """Run tuning for one seed"""
    print(f"\n{'='*80}")
    print(f"Tuning Seed: {seed}")
    print(f"{'='*80}")
    
    # Split data
    X_train, y_train, X_test, y_test, feature_names = split_data(seed)
    
    # Tune models
    rf_params, rf_score = tune_rf(X_train, y_train, seed)
    gb_params, gb_score = tune_gb(X_train, y_train, seed)
    xgb_params, xgb_score = tune_xgb(X_train, y_train, seed)
    
    return {
        'seed': seed,
        'rf': {'params': rf_params, 'cv_score': rf_score},
        'gb': {'params': gb_params, 'cv_score': gb_score},
        'xgb': {'params': xgb_params, 'cv_score': xgb_score}
    }

def main():
    print("="*80)
    print("Step 1: Hyperparameter Tuning (5 Seeds)")
    print("="*80)
    
    tuning_seeds = [1, 2, 3, 4, 5]
    all_tuning_results = []
    
    for seed in tuning_seeds:
        result = run_tuning_one_seed(seed)
        all_tuning_results.append(result)
    
    # Save results
    output_path = config.RESULTS_DIR / "tuning_5seeds_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_tuning_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Tuning results saved to: {output_path}")
    print(f"{'='*80}")
    
    # Print summary
    print("\nTuning Summary:")
    for model_name in ['rf', 'gb', 'xgb']:
        print(f"\n{model_name.upper()}:")
        for result in all_tuning_results:
            print(f"  Seed {result['seed']}: CV={result[model_name]['cv_score']:.4f}, Params={result[model_name]['params']}")

if __name__ == '__main__':
    main()
