"""
Phase 5-2: False Negative Recovery Rate Analysis
Analyzes how text integration recovers defaults missed by structured-only model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.metrics import confusion_matrix

import config

def load_models_and_data():
    """
    Load trained models and preprocessed data
    """
    # Load structured-only model (GB)
    struct_model_path = config.MODELS_DIR / "phase0" / "gb_model.joblib"
    struct_model = joblib.load(struct_model_path)
    
    # Load merged models (RF + TF-IDF, GB + TF-IDF)
    rf_merged_path = config.MODELS_DIR / "phase2" / "stage1_tfidf_rf_model.joblib"
    gb_merged_path = config.MODELS_DIR / "phase2" / "stage1_tfidf_gb_model.joblib"
    rf_merged_model = joblib.load(rf_merged_path)
    gb_merged_model = joblib.load(gb_merged_path)
    
    # Load preprocessed data
    struct_data_path = '/home/ubuntu/upload/preprocessed_struct_only_binary.pkl'
    with open(struct_data_path, 'rb') as f:
        struct_data = pickle.load(f)
    
    merged_data_path = '/home/ubuntu/upload/preprocessed_merged_struct_tfidf_binary.pkl'
    with open(merged_data_path, 'rb') as f:
        merged_data = pickle.load(f)
    
    # Load raw data for credit score groups
    raw_df = pd.read_excel(config.RAW_DATA_FILE)
    
    return struct_model, rf_merged_model, gb_merged_model, struct_data, merged_data, raw_df

def get_credit_score_groups(raw_df, test_idx):
    """
    Get high-risk and low-risk groups based on credit score
    """
    test_df = raw_df.iloc[test_idx]
    
    # Calculate percentiles
    low_threshold = raw_df['신용평점'].quantile(0.30)
    high_threshold = raw_df['신용평점'].quantile(0.70)
    
    # High risk: bottom 30%
    high_risk_mask = test_df['신용평점'] <= low_threshold
    
    # Low risk: top 30%
    low_risk_mask = test_df['신용평점'] >= high_threshold
    
    return high_risk_mask.values, low_risk_mask.values

def analyze_confusion_matrix(struct_model, rf_model, gb_model, struct_data, merged_data):
    """
    Analyze confusion matrices and FN changes
    """
    print(f"\n{'='*80}")
    print("Confusion Matrix Comparison")
    print(f"{'='*80}")
    
    X_test_struct = struct_data['X_test']
    X_test_merged = merged_data['X_test']
    y_test = struct_data['y_test']
    
    # Get predictions
    struct_pred = struct_model.predict(X_test_struct)
    rf_pred = rf_model.predict(X_test_merged)
    gb_pred = gb_model.predict(X_test_merged)
    
    # Calculate confusion matrices
    cm_struct = confusion_matrix(y_test, struct_pred)
    cm_rf = confusion_matrix(y_test, rf_pred)
    cm_gb = confusion_matrix(y_test, gb_pred)
    
    tn_s, fp_s, fn_s, tp_s = cm_struct.ravel()
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
    tn_gb, fp_gb, fn_gb, tp_gb = cm_gb.ravel()
    
    print("\nStructured-only model:")
    print(f"  TN: {tn_s:>4}  FP: {fp_s:>4}")
    print(f"  FN: {fn_s:>4}  TP: {tp_s:>4}")
    
    print("\nRF+Text model:")
    print(f"  TN: {tn_rf:>4}  FP: {fp_rf:>4}")
    print(f"  FN: {fn_rf:>4}  TP: {tp_rf:>4}")
    print(f"  FN Change: {fn_rf - fn_s:+d} ({(fn_rf - fn_s) / fn_s * 100:+.2f}%)")
    
    print("\nGB+Text model:")
    print(f"  TN: {tn_gb:>4}  FP: {fp_gb:>4}")
    print(f"  FN: {fn_gb:>4}  TP: {tp_gb:>4}")
    print(f"  FN Change: {fn_gb - fn_s:+d} ({(fn_gb - fn_s) / fn_s * 100:+.2f}%)")
    
    # Create confusion matrix table
    cm_results = pd.DataFrame([
        {
            'model': 'Structured Standalone',
            'TN': tn_s,
            'FP': fp_s,
            'FN': fn_s,
            'TP': tp_s,
            'FN_change': 0,
            'FN_change_rate': 0.0
        },
        {
            'model': 'RF + Text',
            'TN': tn_rf,
            'FP': fp_rf,
            'FN': fn_rf,
            'TP': tp_rf,
            'FN_change': fn_rf - fn_s,
            'FN_change_rate': (fn_rf - fn_s) / fn_s * 100
        },
        {
            'model': 'GB + Text',
            'TN': tn_gb,
            'FP': fp_gb,
            'FN': fn_gb,
            'TP': tp_gb,
            'FN_change': fn_gb - fn_s,
            'FN_change_rate': (fn_gb - fn_s) / fn_s * 100
        }
    ])
    
    return cm_results, struct_pred, rf_pred, gb_pred

def calculate_fn_recovery_rate(struct_pred, merged_pred, y_test, mask=None):
    """
    Calculate FN recovery rate
    FN Recovery = cases where struct predicted 0 but actual is 1, and merged predicted 1
    """
    if mask is not None:
        struct_pred = struct_pred[mask]
        merged_pred = merged_pred[mask]
        y_test = y_test[mask]
    
    # Find FNs in structured model
    struct_fn_mask = (struct_pred == 0) & (y_test == 1)
    struct_fn_count = struct_fn_mask.sum()
    
    if struct_fn_count == 0:
        return 0, 0.0
    
    # Find recovered FNs (struct FN -> merged TP)
    recovered_mask = struct_fn_mask & (merged_pred == 1)
    recovered_count = recovered_mask.sum()
    
    recovery_rate = recovered_count / struct_fn_count * 100
    
    return recovered_count, recovery_rate

def analyze_fn_recovery(struct_pred, rf_pred, gb_pred, y_test, high_risk_mask, low_risk_mask):
    """
    Analyze FN recovery rates overall and by credit tier
    """
    print(f"\n{'='*80}")
    print("False Negative Recovery Rate Analysis")
    print(f"{'='*80}")
    
    # Overall FN count in structured model
    struct_fn_count = ((struct_pred == 0) & (y_test == 1)).sum()
    print(f"\nStructured-only model FN count: {struct_fn_count}")
    
    # Overall recovery
    rf_recovery, rf_rate = calculate_fn_recovery_rate(struct_pred, rf_pred, y_test)
    gb_recovery, gb_rate = calculate_fn_recovery_rate(struct_pred, gb_pred, y_test)
    
    print(f"\nOverall FN Recovery:")
    print(f"  RF+Text: {rf_recovery}/{struct_fn_count} ({rf_rate:.2f}%)")
    print(f"  GB+Text: {gb_recovery}/{struct_fn_count} ({gb_rate:.2f}%)")
    
    # High-risk group recovery
    struct_fn_high = ((struct_pred[high_risk_mask] == 0) & (y_test[high_risk_mask] == 1)).sum()
    rf_recovery_high, rf_rate_high = calculate_fn_recovery_rate(struct_pred, rf_pred, y_test, high_risk_mask)
    gb_recovery_high, gb_rate_high = calculate_fn_recovery_rate(struct_pred, gb_pred, y_test, high_risk_mask)
    
    print(f"\nHigh-risk group (bottom 30%) FN Recovery:")
    print(f"  Structured FN count: {struct_fn_high}")
    print(f"  RF+Text: {rf_recovery_high}/{struct_fn_high} ({rf_rate_high:.2f}%)")
    print(f"  GB+Text: {gb_recovery_high}/{struct_fn_high} ({gb_rate_high:.2f}%)")
    
    # Low-risk group recovery
    struct_fn_low = ((struct_pred[low_risk_mask] == 0) & (y_test[low_risk_mask] == 1)).sum()
    rf_recovery_low, rf_rate_low = calculate_fn_recovery_rate(struct_pred, rf_pred, y_test, low_risk_mask)
    gb_recovery_low, gb_rate_low = calculate_fn_recovery_rate(struct_pred, gb_pred, y_test, low_risk_mask)
    
    print(f"\nLow-risk group (top 30%) FN Recovery:")
    print(f"  Structured FN count: {struct_fn_low}")
    print(f"  RF+Text: {rf_recovery_low}/{struct_fn_low} ({rf_rate_low:.2f}%)")
    print(f"  GB+Text: {gb_recovery_low}/{struct_fn_low} ({gb_rate_low:.2f}%)")
    
    # Create recovery rate table
    recovery_results = pd.DataFrame([
        {
            'category': 'Overall',
            'structured_FN': struct_fn_count,
            'RF_recovery': rf_recovery,
            'RF_recovery_rate': rf_rate,
            'GB_recovery': gb_recovery,
            'GB_recovery_rate': gb_rate
        },
        {
            'category': 'High-risk group (bottom 30%)',
            'structured_FN': struct_fn_high,
            'RF_recovery': rf_recovery_high,
            'RF_recovery_rate': rf_rate_high,
            'GB_recovery': gb_recovery_high,
            'GB_recovery_rate': gb_rate_high
        },
        {
            'category': 'Low-risk group (top 30%)',
            'structured_FN': struct_fn_low,
            'RF_recovery': rf_recovery_low,
            'RF_recovery_rate': rf_rate_low,
            'GB_recovery': gb_recovery_low,
            'GB_recovery_rate': gb_rate_low
        }
    ])
    
    return recovery_results

def main():
    print(f"{'='*80}")
    print("Phase 5-2: False Negative Recovery Rate Analysis")
    print(f"{'='*80}")
    
    # Load models and data
    struct_model, rf_model, gb_model, struct_data, merged_data, raw_df = load_models_and_data()
    
    # Get test data
    y_test = struct_data['y_test']
    
    # Load test indices from saved file
    test_idx_file = config.SPLITS_DIR / "test_indices.npy"
    test_idx = np.load(test_idx_file)
    
    # Get credit score groups
    high_risk_mask, low_risk_mask = get_credit_score_groups(raw_df, test_idx)
    
    # Analyze confusion matrices
    cm_results, struct_pred, rf_pred, gb_pred = analyze_confusion_matrix(
        struct_model, rf_model, gb_model, struct_data, merged_data
    )
    
    # Analyze FN recovery rates
    recovery_results = analyze_fn_recovery(
        struct_pred, rf_pred, gb_pred, y_test, high_risk_mask, low_risk_mask
    )
    
    # Save results
    config.ensure_dir(config.TABLES_DIR)
    
    cm_output = config.TABLES_DIR / "table_4_8_confusion_matrix.csv"
    cm_results.to_csv(cm_output, index=False, encoding='utf-8-sig')
    print(f"\nConfusion matrix results saved to: {cm_output}")
    
    recovery_output = config.TABLES_DIR / "table_4_9_fn_recovery_rate.csv"
    recovery_results.to_csv(recovery_output, index=False, encoding='utf-8-sig')
    print(f"FN recovery results saved to: {recovery_output}")
    
    print(f"\n{'='*80}")
    print("Phase 5-2 completed successfully!")
    print(f"{'='*80}")
    
    return cm_results, recovery_results

if __name__ == "__main__":
    cm_results, recovery_results = main()
