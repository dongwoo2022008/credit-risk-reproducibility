"""
Evaluate Original Paper Models (Phase 3 & 4)
Loads and evaluates the original trained models from the paper to reproduce exact results
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pickle
import pandas as pd
import config
from utils.evaluator import evaluate_model

def load_minilm_data():
    """
    Load preprocessed MiniLM data (397 features: 13 structured + 384 MiniLM)
    """
    print("\nLoading preprocessed merged data (MiniLM - 397 features)...")
    data_path = config.PROJECT_ROOT / "data" / "preprocessed" / "preprocessed_merged_struct_minilm_binary.pkl"
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"Test samples: {len(X_test)}, Features: {X_test.shape[1]}")
    
    return X_test, y_test

def evaluate_phase3_models(X_test, y_test):
    """
    Evaluate Phase 3 (hyperparameter tuning) original models
    """
    print(f"\n{'='*80}")
    print("Phase 3: 하이퍼파라미터 튜닝 모델 (논문 원본)")
    print(f"{'='*80}")
    
    models_dir = config.MODELS_DIR / "phase3_original"
    
    models = {
        'LR': 'lr_tuned_model.pkl',
        'GB': 'gb_tuned_model.pkl',
        'XGB': 'xgb_tuned_model.pkl',
        'RF': 'rf_tuned_model.pkl'
    }
    
    results = []
    
    for name, filename in models.items():
        model_path = models_dir / filename
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract model from dict if needed
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
            else:
                model = model_data
            
            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = evaluate_model(y_test, y_pred, y_proba)
            
            print(f"\n{name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            results.append({
                'model': name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc']
            })
            
        except Exception as e:
            print(f"\n{name}: Error - {e}")
    
    return results

def evaluate_phase4_models(X_test, y_test):
    """
    Evaluate Phase 4 (ensemble) original models
    """
    print(f"\n{'='*80}")
    print("Phase 4: 앙상블 모델 (논문 원본)")
    print(f"{'='*80}")
    
    models_dir = config.MODELS_DIR / "phase4_original"
    
    models = {
        'Voting (Soft)': 'voting_soft_model.pkl',
        'Voting (Weighted)': 'voting_weighted_model.pkl',
        'Stacking': 'stacking_model.pkl'
    }
    
    results = []
    
    for name, filename in models.items():
        model_path = models_dir / filename
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract model from dict if needed
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
            else:
                model = model_data
            
            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = evaluate_model(y_test, y_pred, y_proba)
            
            print(f"\n{name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            results.append({
                'model': name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc']
            })
            
        except Exception as e:
            print(f"\n{name}: Error - {e}")
    
    return results

def main():
    print(f"{'='*80}")
    print("논문 원본 모델 평가 (Phase 3 & 4)")
    print(f"{'='*80}")
    
    # Load data
    X_test, y_test = load_minilm_data()
    
    # Evaluate Phase 3
    results_phase3 = evaluate_phase3_models(X_test, y_test)
    
    # Evaluate Phase 4
    results_phase4 = evaluate_phase4_models(X_test, y_test)
    
    # Save results
    print(f"\n{'='*80}")
    print("결과 저장")
    print(f"{'='*80}")
    
    config.ensure_dir(config.TABLES_DIR)
    
    # Phase 3 results
    df_phase3 = pd.DataFrame(results_phase3)
    output_phase3 = config.TABLES_DIR / "table_phase3_paper_original.csv"
    df_phase3.to_csv(output_phase3, index=False, encoding='utf-8-sig')
    print(f"\nPhase 3 results saved to: {output_phase3}")
    
    # Phase 4 results
    df_phase4 = pd.DataFrame(results_phase4)
    output_phase4 = config.TABLES_DIR / "table_phase4_paper_original.csv"
    df_phase4.to_csv(output_phase4, index=False, encoding='utf-8-sig')
    print(f"Phase 4 results saved to: {output_phase4}")
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    
    print("\nPhase 3 (하이퍼파라미터 튜닝):")
    for r in results_phase3:
        print(f"  {r['model']:10s} - ROC-AUC: {r['roc_auc']:.4f}, F1: {r['f1_score']:.4f}")
    
    print("\nPhase 4 (앙상블):")
    for r in results_phase4:
        print(f"  {r['model']:20s} - ROC-AUC: {r['roc_auc']:.4f}, F1: {r['f1_score']:.4f}")
    
    print(f"\n{'='*80}")
    print("✅ 논문 원본 모델 평가 완료!")
    print(f"{'='*80}")
    
    return df_phase3, df_phase4

if __name__ == "__main__":
    df_phase3, df_phase4 = main()
