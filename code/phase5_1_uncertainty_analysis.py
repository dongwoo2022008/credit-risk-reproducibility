"""
Phase 5-1: Uncertainty Interval Analysis (Marginal vs Clear Cases)
Analyzes model performance in uncertain vs certain prediction regions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pickle
import joblib

import config
from utils.evaluator import evaluate_model

def load_models_and_data():
    """
    Load trained models and preprocessed data
    """
    # Load structured-only model
    struct_model_path = config.MODELS_DIR / "phase0" / "gb_model.joblib"
    struct_model = joblib.load(struct_model_path)
    
    # Load merged model (GB + TF-IDF)
    merged_model_path = config.MODELS_DIR / "phase2" / "stage1_tfidf_gb_model.joblib"
    merged_model = joblib.load(merged_model_path)
    
    # Load preprocessed data
    struct_data_path = '/home/ubuntu/upload/preprocessed_struct_only_binary.pkl'
    with open(struct_data_path, 'rb') as f:
        struct_data = pickle.load(f)
    
    merged_data_path = '/home/ubuntu/upload/preprocessed_merged_struct_tfidf_binary.pkl'
    with open(merged_data_path, 'rb') as f:
        merged_data = pickle.load(f)
    
    return struct_model, merged_model, struct_data, merged_data

def analyze_uncertainty_intervals(struct_model, merged_model, struct_data, merged_data):
    """
    Analyze performance in marginal (0.3-0.7) vs clear cases
    """
    print(f"\n{'='*80}")
    print("Phase 5-1: Uncertainty Interval Analysis")
    print(f"{'='*80}")
    
    # Get test data
    X_test_struct = struct_data['X_test']
    X_test_merged = merged_data['X_test']
    y_test = struct_data['y_test']
    
    # Get predictions
    struct_proba = struct_model.predict_proba(X_test_struct)[:, 1]
    merged_proba = merged_model.predict_proba(X_test_merged)[:, 1]
    
    # Define marginal and clear cases based on structured model predictions
    marginal_mask = (struct_proba >= 0.3) & (struct_proba <= 0.7)
    clear_mask = ~marginal_mask
    
    print(f"\nTotal test samples: {len(y_test)}")
    print(f"Marginal cases (0.3-0.7): {marginal_mask.sum()} ({marginal_mask.sum()/len(y_test)*100:.2f}%)")
    print(f"Clear cases: {clear_mask.sum()} ({clear_mask.sum()/len(y_test)*100:.2f}%)")
    
    # Analyze marginal cases
    print(f"\n{'='*80}")
    print("MARGINAL CASES (Predicted Probability 0.3-0.7)")
    print(f"{'='*80}")
    
    if marginal_mask.sum() > 0:
        y_test_marginal = y_test[marginal_mask]
        struct_proba_marginal = struct_proba[marginal_mask]
        merged_proba_marginal = merged_proba[marginal_mask]
        
        struct_pred_marginal = (struct_proba_marginal >= 0.5).astype(int)
        merged_pred_marginal = (merged_proba_marginal >= 0.5).astype(int)
        
        struct_metrics_marginal = evaluate_model(y_test_marginal, struct_pred_marginal, struct_proba_marginal)
        merged_metrics_marginal = evaluate_model(y_test_marginal, merged_pred_marginal, merged_proba_marginal)
        
        print("\nStructured-only model:")
        print(f"  Accuracy:  {struct_metrics_marginal['accuracy']:.4f}")
        print(f"  Precision: {struct_metrics_marginal['precision']:.4f}")
        print(f"  Recall:    {struct_metrics_marginal['recall']:.4f}")
        print(f"  F1-Score:  {struct_metrics_marginal['f1_score']:.4f}")
        print(f"  ROC-AUC:   {struct_metrics_marginal['roc_auc']:.4f}")
        
        print("\nGB+Text model:")
        print(f"  Accuracy:  {merged_metrics_marginal['accuracy']:.4f}")
        print(f"  Precision: {merged_metrics_marginal['precision']:.4f}")
        print(f"  Recall:    {merged_metrics_marginal['recall']:.4f}")
        print(f"  F1-Score:  {merged_metrics_marginal['f1_score']:.4f}")
        print(f"  ROC-AUC:   {merged_metrics_marginal['roc_auc']:.4f}")
        
        print("\nImprovement:")
        print(f"  ROC-AUC: {(merged_metrics_marginal['roc_auc'] - struct_metrics_marginal['roc_auc']) / struct_metrics_marginal['roc_auc'] * 100:+.2f}%")
        print(f"  F1-Score: {(merged_metrics_marginal['f1_score'] - struct_metrics_marginal['f1_score']) / struct_metrics_marginal['f1_score'] * 100:+.2f}%")
    
    # Analyze clear cases
    print(f"\n{'='*80}")
    print("CLEAR CASES (Predicted Probability < 0.3 or > 0.7)")
    print(f"{'='*80}")
    
    if clear_mask.sum() > 0:
        y_test_clear = y_test[clear_mask]
        struct_proba_clear = struct_proba[clear_mask]
        merged_proba_clear = merged_proba[clear_mask]
        
        struct_pred_clear = (struct_proba_clear >= 0.5).astype(int)
        merged_pred_clear = (merged_proba_clear >= 0.5).astype(int)
        
        struct_metrics_clear = evaluate_model(y_test_clear, struct_pred_clear, struct_proba_clear)
        merged_metrics_clear = evaluate_model(y_test_clear, merged_pred_clear, merged_proba_clear)
        
        print("\nStructured-only model:")
        print(f"  Accuracy:  {struct_metrics_clear['accuracy']:.4f}")
        print(f"  Precision: {struct_metrics_clear['precision']:.4f}")
        print(f"  Recall:    {struct_metrics_clear['recall']:.4f}")
        print(f"  F1-Score:  {struct_metrics_clear['f1_score']:.4f}")
        print(f"  ROC-AUC:   {struct_metrics_clear['roc_auc']:.4f}")
        
        print("\nGB+Text model:")
        print(f"  Accuracy:  {merged_metrics_clear['accuracy']:.4f}")
        print(f"  Precision: {merged_metrics_clear['precision']:.4f}")
        print(f"  Recall:    {merged_metrics_clear['recall']:.4f}")
        print(f"  F1-Score:  {merged_metrics_clear['f1_score']:.4f}")
        print(f"  ROC-AUC:   {merged_metrics_clear['roc_auc']:.4f}")
        
        print("\nImprovement:")
        print(f"  ROC-AUC: {(merged_metrics_clear['roc_auc'] - struct_metrics_clear['roc_auc']) / struct_metrics_clear['roc_auc'] * 100:+.2f}%")
        print(f"  F1-Score: {(merged_metrics_clear['f1_score'] - struct_metrics_clear['f1_score']) / struct_metrics_clear['f1_score'] * 100:+.2f}%")
    
    # Create results table
    results = []
    
    if marginal_mask.sum() > 0:
        results.append({
            'case_type': 'Marginal',
            'count': marginal_mask.sum(),
            'percentage': marginal_mask.sum() / len(y_test) * 100,
            'struct_roc_auc': struct_metrics_marginal['roc_auc'],
            'merged_roc_auc': merged_metrics_marginal['roc_auc'],
            'roc_auc_improvement': (merged_metrics_marginal['roc_auc'] - struct_metrics_marginal['roc_auc']) / struct_metrics_marginal['roc_auc'] * 100,
            'struct_f1': struct_metrics_marginal['f1_score'],
            'merged_f1': merged_metrics_marginal['f1_score'],
            'f1_improvement': (merged_metrics_marginal['f1_score'] - struct_metrics_marginal['f1_score']) / struct_metrics_marginal['f1_score'] * 100
        })
    
    if clear_mask.sum() > 0:
        results.append({
            'case_type': 'Clear',
            'count': clear_mask.sum(),
            'percentage': clear_mask.sum() / len(y_test) * 100,
            'struct_roc_auc': struct_metrics_clear['roc_auc'],
            'merged_roc_auc': merged_metrics_clear['roc_auc'],
            'roc_auc_improvement': (merged_metrics_clear['roc_auc'] - struct_metrics_clear['roc_auc']) / struct_metrics_clear['roc_auc'] * 100,
            'struct_f1': struct_metrics_clear['f1_score'],
            'merged_f1': merged_metrics_clear['f1_score'],
            'f1_improvement': (merged_metrics_clear['f1_score'] - struct_metrics_clear['f1_score']) / struct_metrics_clear['f1_score'] * 100
        })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    config.ensure_dir(config.TABLES_DIR)
    output_file = config.TABLES_DIR / "table_4_7_uncertainty_analysis.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_file}")
    
    return results_df

def main():
    print(f"{'='*80}")
    print("Phase 5-1: Uncertainty Interval Analysis")
    print(f"{'='*80}")
    
    # Load models and data
    struct_model, merged_model, struct_data, merged_data = load_models_and_data()
    
    # Analyze uncertainty intervals
    results_df = analyze_uncertainty_intervals(struct_model, merged_model, struct_data, merged_data)
    
    print(f"\n{'='*80}")
    print("Phase 5-1 completed successfully!")
    print(f"{'='*80}")
    
    return results_df

if __name__ == "__main__":
    results = main()
