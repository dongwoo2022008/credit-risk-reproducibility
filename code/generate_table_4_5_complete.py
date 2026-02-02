"""
Generate Table 4-5: Performance Comparison: Single Model vs. Ensemble Model (Complete)
Uses Phase 2 Stage 1 (TF-IDF) results for single models and Phase 4 results for all 5 ensemble models
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import config

def main():
    print("="*80)
    print("Generating Table 4-5: Single vs Ensemble Model Comparison (Complete)")
    print("="*80)
    
    # Load Phase 2 Stage 1 (TF-IDF) results
    print("\nLoading Phase 2 Stage 1 (TF-IDF) results...")
    phase2_path = config.RESULTS_DIR / "tables" / "phase2_stage1_tfidf_performance.csv"
    phase2_df = pd.read_csv(phase2_path, encoding='utf-8-sig')
    
    # Load Phase 4 ensemble results (all 5 methods)
    print("Loading Phase 4 ensemble results...")
    phase4_path = config.RESULTS_DIR / "tables" / "phase4_ensemble_all_methods.csv"
    phase4_df = pd.read_csv(phase4_path, encoding='utf-8-sig')
    
    # Select single models: GB, XGB, RF
    single_models = ['GB', 'XGB', 'RF']
    single_df = phase2_df[phase2_df['model'].isin(single_models)].copy()
    
    # Prepare table data
    table_data = []
    
    # Add single models
    for _, row in single_df.iterrows():
        table_data.append({
            'Category': 'Single',
            'Model': row['model'],
            'ROC-AUC': f"{row['roc_auc']:.3f}",
            'Recall': f"{row['recall']:.3f}",
            'F1-score': f"{row['f1_score']:.3f}"
        })
    
    # Add ensemble models
    ensemble_mapping = {
        'Voting-H': 'Voting-H',
        'Voting-S': 'Voting-S',
        'Voting-W': 'Voting-W',
        'BLD': 'BLD',
        'STK': 'STK'
    }
    
    for _, row in phase4_df.iterrows():
        model_name = row['model']
        if model_name in ensemble_mapping:
            table_data.append({
                'Category': 'Ensemble',
                'Model': ensemble_mapping[model_name],
                'ROC-AUC': f"{row['roc_auc']:.3f}",
                'Recall': f"{row['recall']:.3f}",
                'F1-score': f"{row['f1_score']:.3f}"
            })
    
    # Create DataFrame
    table_df = pd.DataFrame(table_data)
    
    # Save table
    output_path = config.RESULTS_DIR / "tables" / "table_4_5_single_vs_ensemble.csv"
    table_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nTable 4-5 saved to: {output_path}")
    print("\nTable Preview:")
    print(table_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("Table 4-5 generated successfully!")
    print("="*80)

if __name__ == '__main__':
    main()
