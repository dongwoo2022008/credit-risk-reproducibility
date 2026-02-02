"""
Generate Table 4-5: Performance Comparison: Single Model vs. Ensemble Model
"""

import pandas as pd

# Single models (Phase 3 tuning results)
single_models = {
    'GB': {'roc_auc': 0.7883, 'recall': 0.8077, 'f1_score': 0.7612},
    'XGB': {'roc_auc': 0.7932, 'recall': 0.7943, 'f1_score': 0.7636},
    'RF': {'roc_auc': 0.7397, 'recall': 0.8256, 'f1_score': 0.7461}
}

# Ensemble models (Phase 4 results)
# Note: Phase 4 file only has Voting (Soft), Voting (Weighted), Stacking
# Need to check if Voting (Hard) and Blending exist
ensemble_models = {
    'Voting-S': {'roc_auc': 0.7935, 'recall': 0.8197, 'f1_score': 0.7655},
    'Voting-W': {'roc_auc': 0.7946, 'recall': 0.8137, 'f1_score': 0.7647},
    'STK': {'roc_auc': 0.7959, 'recall': 0.8077, 'f1_score': 0.7645}
}

# Build table data
table_data = []

# Single models
for model_name, metrics in single_models.items():
    row = [
        'Single',
        model_name,
        f"{metrics['roc_auc']:.3f}",
        f"{metrics['recall']:.3f}",
        f"{metrics['f1_score']:.3f}"
    ]
    table_data.append(row)

# Ensemble models
for model_name, metrics in ensemble_models.items():
    row = [
        'Ensemble',
        model_name,
        f"{metrics['roc_auc']:.3f}",
        f"{metrics['recall']:.3f}",
        f"{metrics['f1_score']:.3f}"
    ]
    table_data.append(row)

# Create DataFrame
df = pd.DataFrame(table_data, columns=[
    'Category', 'Model', 'ROC-AUC', 'Recall', 'F1-score'
])

# Save to CSV
output_path = '/home/ubuntu/credit-risk-reproducibility/results/tables/table_4_5_single_vs_ensemble.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("Table 4-5 generated successfully!")
print("\nTable Preview:")
print(df.to_string(index=False))

print("\n\nNote: Phase 4 file only contains 3 ensemble models.")
print("If Voting-H and BLD exist, please provide their results.")
