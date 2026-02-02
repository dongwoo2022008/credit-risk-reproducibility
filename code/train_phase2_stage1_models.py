"""
Train and save Phase 2 Stage 1 (TF-IDF) models for Phase 4 ensemble
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import config

def main():
    print("="*80)
    print("Training Phase 2 Stage 1 (TF-IDF) Models")
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
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Initialize models (same as Phase 2)
    models = {
        'rf': RandomForestClassifier(
            n_estimators=100,
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=100,
            random_state=config.RANDOM_SEED
        ),
        'xgb': XGBClassifier(
            n_estimators=100,
            random_state=config.RANDOM_SEED,
            eval_metric='logloss',
            n_jobs=-1
        )
    }
    
    # Train and save models
    save_dir = config.MODELS_DIR / "phase2" / "stage1_tfidf"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        print(f"\nTraining {name.upper()}...")
        model.fit(X_train, y_train)
        
        # Save model
        model_path = save_dir / f"{name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"  Saved to: {model_path}")
        
        # Quick evaluation
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"  Train accuracy: {train_score:.4f}")
        print(f"  Test accuracy: {test_score:.4f}")
    
    print("\n" + "="*80)
    print("Phase 2 Stage 1 models trained and saved successfully!")
    print("="*80)

if __name__ == '__main__':
    main()
