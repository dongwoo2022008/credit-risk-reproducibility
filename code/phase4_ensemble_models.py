"""
Phase 4: Ensemble Models (Voting, Blending, Stacking)
Combines multiple models to improve prediction performance
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

import config
from utils.data_loader import load_raw_data, encode_target, prepare_structured_features
from utils.evaluator import evaluate_model

def load_base_models():
    """
    Load trained base models from Phase 0 and Phase 3
    """
    print("\nLoading base models...")
    
    models = {}
    
    # Try to load tuned models first (Phase 3)
    phase3_dir = config.MODELS_DIR / "phase3"
    if phase3_dir.exists():
        for model_name in ['rf', 'gb', 'xgb']:
            tuned_path = phase3_dir / f"{model_name}_tuned_model.joblib"
            if tuned_path.exists():
                models[model_name] = joblib.load(tuned_path)
                print(f"  Loaded tuned {model_name.upper()} model")
            else:
                # Fall back to Phase 0 baseline
                baseline_path = config.MODELS_DIR / "phase0" / f"{model_name}_model.joblib"
                if baseline_path.exists():
                    models[model_name] = joblib.load(baseline_path)
                    print(f"  Loaded baseline {model_name.upper()} model")
    else:
        # Load Phase 0 baseline models
        for model_name in ['rf', 'gb', 'xgb']:
            baseline_path = config.MODELS_DIR / "phase0" / f"{model_name}_model.joblib"
            if baseline_path.exists():
                models[model_name] = joblib.load(baseline_path)
                print(f"  Loaded baseline {model_name.upper()} model")
    
    return models

def voting_ensemble(base_models, X_train, y_train, X_test, y_test):
    """
    Create and evaluate voting ensemble
    """
    print(f"\n{'='*80}")
    print("Voting Ensemble")
    print(f"{'='*80}")
    
    # Create voting classifier (soft voting for probability averaging)
    estimators = [(name, model) for name, model in base_models.items()]
    
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )
    
    print("\nTraining voting ensemble...")
    voting_clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = voting_clf.predict(X_test)
    y_proba = voting_clf.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_proba)
    
    print(f"\nVoting Ensemble Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Save model
    config.ensure_dir(config.MODELS_DIR / "phase4")
    model_path = config.MODELS_DIR / "phase4" / "voting_ensemble.joblib"
    joblib.dump(voting_clf, model_path)
    print(f"\nVoting ensemble saved to: {model_path}")
    
    return voting_clf, metrics

def blending_ensemble(base_models, X_train, y_train, X_test, y_test):
    """
    Create and evaluate blending ensemble (simple averaging)
    """
    print(f"\n{'='*80}")
    print("Blending Ensemble (Simple Averaging)")
    print(f"{'='*80}")
    
    # Get predictions from all base models
    print("\nGetting predictions from base models...")
    predictions = []
    
    for name, model in base_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        predictions.append(y_proba)
        print(f"  {name.upper()} predictions obtained")
    
    # Average predictions
    blended_proba = np.mean(predictions, axis=0)
    blended_pred = (blended_proba >= 0.5).astype(int)
    
    # Evaluate
    metrics = evaluate_model(y_test, blended_pred, blended_proba)
    
    print(f"\nBlending Ensemble Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return blended_proba, metrics

def stacking_ensemble(base_models, X_train, y_train, X_test, y_test):
    """
    Create and evaluate stacking ensemble
    """
    print(f"\n{'='*80}")
    print("Stacking Ensemble")
    print(f"{'='*80}")
    
    # Create stacking classifier with logistic regression as meta-learner
    estimators = [(name, model) for name, model in base_models.items()]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000),
        cv=5,
        n_jobs=-1
    )
    
    print("\nTraining stacking ensemble...")
    print("  Using 5-fold CV for meta-features...")
    stacking_clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = stacking_clf.predict(X_test)
    y_proba = stacking_clf.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_proba)
    
    print(f"\nStacking Ensemble Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Save model
    model_path = config.MODELS_DIR / "phase4" / "stacking_ensemble.joblib"
    joblib.dump(stacking_clf, model_path)
    print(f"\nStacking ensemble saved to: {model_path}")
    
    return stacking_clf, metrics

def compare_with_base_models(base_models, ensemble_results, X_test, y_test):
    """
    Compare ensemble models with base models
    """
    print(f"\n{'='*80}")
    print("Comparison: Base Models vs Ensemble Models")
    print(f"{'='*80}")
    
    results = []
    
    # Base models
    for name, model in base_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_proba)
        
        results.append({
            'model': f'{name.upper()} (base)',
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc']
        })
    
    # Ensemble models
    for ensemble_name, metrics in ensemble_results.items():
        results.append({
            'model': ensemble_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc']
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roc_auc', ascending=False)
    
    print("\nAll Models Ranked by ROC-AUC:")
    print(results_df.to_string(index=False))
    
    # Find best model
    best_model = results_df.iloc[0]
    print(f"\nBest Model: {best_model['model']}")
    print(f"  ROC-AUC: {best_model['roc_auc']:.4f}")
    
    return results_df

def main():
    print(f"{'='*80}")
    print("Phase 4: Ensemble Models")
    print(f"{'='*80}")
    
    # Load data
    print("\nLoading data...")
    train_idx = np.load(config.SPLITS_DIR / "train_indices.npy")
    test_idx = np.load(config.SPLITS_DIR / "test_indices.npy")
    
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
    
    # Load base models
    base_models = load_base_models()
    
    if len(base_models) == 0:
        print("\nError: No base models found. Please run Phase 0 or Phase 3 first.")
        return
    
    print(f"\nLoaded {len(base_models)} base models: {list(base_models.keys())}")
    
    # Store ensemble results
    ensemble_results = {}
    
    # 1. Voting Ensemble
    voting_clf, voting_metrics = voting_ensemble(base_models, X_train, y_train, X_test, y_test)
    ensemble_results['Voting Ensemble'] = voting_metrics
    
    # 2. Blending Ensemble
    blended_proba, blending_metrics = blending_ensemble(base_models, X_train, y_train, X_test, y_test)
    ensemble_results['Blending Ensemble'] = blending_metrics
    
    # 3. Stacking Ensemble
    stacking_clf, stacking_metrics = stacking_ensemble(base_models, X_train, y_train, X_test, y_test)
    ensemble_results['Stacking Ensemble'] = stacking_metrics
    
    # Compare all models
    results_df = compare_with_base_models(base_models, ensemble_results, X_test, y_test)
    
    # Save results
    config.ensure_dir(config.TABLES_DIR)
    output_file = config.TABLES_DIR / "table_4_4_ensemble_performance.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_file}")
    
    print(f"\n{'='*80}")
    print("Phase 4 completed successfully!")
    print(f"{'='*80}")
    
    return results_df, ensemble_results

if __name__ == "__main__":
    results_df, ensemble_results = main()
