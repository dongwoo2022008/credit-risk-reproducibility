"""
Phase 4: Ensemble Models - Simplified version with all 5 methods
Voting-H, Voting-S, Voting-W, BLD (Blending), STK (Stacking)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import config

def main():
    print("="*80)
    print("Phase 4: Ensemble Models (All 5 Methods)")
    print("="*80)
    
    # Load TF-IDF preprocessed data
    print("\nLoading TF-IDF preprocessed data...")
    data_path = config.DATA_DIR / "preprocessed" / "preprocessed_merged_struct_tfidf_binary.pkl"
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Load Phase 2 Stage 1 models
    print("\nLoading Phase 2 Stage 1 models...")
    models_dir = config.MODELS_DIR / "phase2" / "stage1_tfidf"
    
    rf_model = pickle.load(open(models_dir / "rf_model.pkl", 'rb'))
    gb_model = pickle.load(open(models_dir / "gb_model.pkl", 'rb'))
    xgb_model = pickle.load(open(models_dir / "xgb_model.pkl", 'rb'))
    
    print("  Loaded RF, GB, XGB models")
    
    base_models = [
        ('rf', rf_model),
        ('gb', gb_model),
        ('xgb', xgb_model)
    ]
    
    results = []
    
    # 1. Voting-H (Hard Voting)
    print("\n" + "="*80)
    print("1. Voting-H (Hard Voting)")
    print("="*80)
    
    voting_hard = VotingClassifier(estimators=base_models, voting='hard')
    voting_hard.fit(X_train, y_train)
    y_pred = voting_hard.predict(X_test)
    
    # For ROC-AUC, use soft voting probabilities
    y_proba = np.mean([m.predict_proba(X_test)[:, 1] for _, m in base_models], axis=0)
    
    results.append({
        'model': 'Voting-H',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    })
    print(f"  ROC-AUC: {results[-1]['roc_auc']:.4f}, Recall: {results[-1]['recall']:.4f}, F1: {results[-1]['f1_score']:.4f}")
    
    # 2. Voting-S (Soft Voting)
    print("\n" + "="*80)
    print("2. Voting-S (Soft Voting)")
    print("="*80)
    
    voting_soft = VotingClassifier(estimators=base_models, voting='soft')
    voting_soft.fit(X_train, y_train)
    y_pred = voting_soft.predict(X_test)
    y_proba = voting_soft.predict_proba(X_test)[:, 1]
    
    results.append({
        'model': 'Voting-S',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    })
    print(f"  ROC-AUC: {results[-1]['roc_auc']:.4f}, Recall: {results[-1]['recall']:.4f}, F1: {results[-1]['f1_score']:.4f}")
    
    # 3. Voting-W (Weighted Voting)
    print("\n" + "="*80)
    print("3. Voting-W (Weighted Voting)")
    print("="*80)
    
    # Use ROC-AUC as weights (from Phase 2 results)
    weights = [0.794, 0.810, 0.797]  # RF, GB, XGB
    
    voting_weighted = VotingClassifier(estimators=base_models, voting='soft', weights=weights)
    voting_weighted.fit(X_train, y_train)
    y_pred = voting_weighted.predict(X_test)
    y_proba = voting_weighted.predict_proba(X_test)[:, 1]
    
    results.append({
        'model': 'Voting-W',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    })
    print(f"  ROC-AUC: {results[-1]['roc_auc']:.4f}, Recall: {results[-1]['recall']:.4f}, F1: {results[-1]['f1_score']:.4f}")
    
    # 4. BLD (Blending)
    print("\n" + "="*80)
    print("4. BLD (Blending)")
    print("="*80)
    
    # Split train into train and blend
    X_train_blend, X_blend, y_train_blend, y_blend = train_test_split(
        X_train, y_train, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y_train
    )
    
    # Train base models on blend train set
    rf_blend = pickle.loads(pickle.dumps(rf_model))
    gb_blend = pickle.loads(pickle.dumps(gb_model))
    xgb_blend = pickle.loads(pickle.dumps(xgb_model))
    
    rf_blend.fit(X_train_blend, y_train_blend)
    gb_blend.fit(X_train_blend, y_train_blend)
    xgb_blend.fit(X_train_blend, y_train_blend)
    
    # Get predictions on blend set
    blend_features = np.column_stack([
        rf_blend.predict_proba(X_blend)[:, 1],
        gb_blend.predict_proba(X_blend)[:, 1],
        xgb_blend.predict_proba(X_blend)[:, 1]
    ])
    
    # Train meta-learner
    meta_learner = LogisticRegression(random_state=config.RANDOM_SEED)
    meta_learner.fit(blend_features, y_blend)
    
    # Get test predictions
    test_features = np.column_stack([
        rf_blend.predict_proba(X_test)[:, 1],
        gb_blend.predict_proba(X_test)[:, 1],
        xgb_blend.predict_proba(X_test)[:, 1]
    ])
    
    y_pred = meta_learner.predict(test_features)
    y_proba = meta_learner.predict_proba(test_features)[:, 1]
    
    results.append({
        'model': 'BLD',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    })
    print(f"  ROC-AUC: {results[-1]['roc_auc']:.4f}, Recall: {results[-1]['recall']:.4f}, F1: {results[-1]['f1_score']:.4f}")
    
    # 5. STK (Stacking)
    print("\n" + "="*80)
    print("5. STK (Stacking)")
    print("="*80)
    
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(random_state=config.RANDOM_SEED),
        cv=5
    )
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    y_proba = stacking.predict_proba(X_test)[:, 1]
    
    results.append({
        'model': 'STK',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    })
    print(f"  ROC-AUC: {results[-1]['roc_auc']:.4f}, Recall: {results[-1]['recall']:.4f}, F1: {results[-1]['f1_score']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = config.RESULTS_DIR / "tables" / "phase4_ensemble_all_methods.csv"
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_path}")
    print("\nAll Results:")
    print(results_df.to_string(index=False))
    print("="*80)

if __name__ == '__main__':
    main()
