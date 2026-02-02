"""
Generate Table 4-4: Comparison of Key Performance Metrics Before and After Hyperparameter Tuning
Compare specific models: XGB (ROC-AUC, F1-score), RF (Recall)
"""

import pandas as pd

# Before (reference): Best performance from Phase 2 merged models (Stage 1-4)
# XGB and RF only

# Stage 1: TF-IDF + Structured
stage1_results = {
    'RF': {'roc_auc': 0.7941, 'recall': 0.7601, 'f1_score': 0.7533},
    'XGB': {'roc_auc': 0.7971, 'recall': 0.7645, 'f1_score': 0.7516}
}

# Stage 2: Subword + Structured
stage2_results = {
    'RF': {'roc_auc': 0.7754, 'recall': 0.7660, 'f1_score': 0.7482},
    'XGB': {'roc_auc': 0.7966, 'recall': 0.7556, 'f1_score': 0.7556}
}

# Stage 3: MiniLM + Structured
stage3_results = {
    'RF': {'roc_auc': 0.7396, 'recall': 0.8331, 'f1_score': 0.7443},
    'XGB': {'roc_auc': 0.7843, 'recall': 0.7884, 'f1_score': 0.7584}
}

# Stage 4: KoSimCSE + Structured
stage4_results = {
    'RF': {'roc_auc': 0.7040, 'recall': 0.8152, 'f1_score': 0.7226},
    'XGB': {'roc_auc': 0.7662, 'recall': 0.7645, 'f1_score': 0.7355}
}

# Combine all stages
all_stages = [stage1_results, stage2_results, stage3_results, stage4_results]
stage_names = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']

# Find XGB's best ROC-AUC across all stages
xgb_best_roc = 0
xgb_best_roc_stage = ''
for stage_idx, stage in enumerate(all_stages):
    if stage['XGB']['roc_auc'] > xgb_best_roc:
        xgb_best_roc = stage['XGB']['roc_auc']
        xgb_best_roc_stage = stage_names[stage_idx]

# Find RF's best Recall across all stages
rf_best_recall = 0
rf_best_recall_stage = ''
for stage_idx, stage in enumerate(all_stages):
    if stage['RF']['recall'] > rf_best_recall:
        rf_best_recall = stage['RF']['recall']
        rf_best_recall_stage = stage_names[stage_idx]

# Find XGB's best F1-score across all stages
xgb_best_f1 = 0
xgb_best_f1_stage = ''
for stage_idx, stage in enumerate(all_stages):
    if stage['XGB']['f1_score'] > xgb_best_f1:
        xgb_best_f1 = stage['XGB']['f1_score']
        xgb_best_f1_stage = stage_names[stage_idx]

print(f"XGB Best ROC-AUC: {xgb_best_roc:.3f} ({xgb_best_roc_stage})")
print(f"RF Best Recall: {rf_best_recall:.3f} ({rf_best_recall_stage})")
print(f"XGB Best F1-score: {xgb_best_f1:.3f} ({xgb_best_f1_stage})")

# After (tuned): Phase 3 hyperparameter tuning results
tuned_results = {
    'XGB': {'roc_auc': 0.7932, 'recall': 0.7943, 'f1_score': 0.7636},
    'RF': {'roc_auc': 0.7397, 'recall': 0.8256, 'f1_score': 0.7461}
}

# Build Table 4-4
table_data = []

# ROC-AUC row (XGB)
roc_row = [
    'ROC-AUC',
    'XGB',
    f"{xgb_best_roc:.3f} ({xgb_best_roc_stage} best)",
    f"{tuned_results['XGB']['roc_auc']:.3f}",
    f"{tuned_results['XGB']['roc_auc'] - xgb_best_roc:.3f}"
]
table_data.append(roc_row)

# Recall row (RF)
recall_row = [
    'Recall',
    'RF',
    f"{rf_best_recall:.3f} ({rf_best_recall_stage} best)",
    f"{tuned_results['RF']['recall']:.3f}",
    f"{tuned_results['RF']['recall'] - rf_best_recall:.3f}"
]
table_data.append(recall_row)

# F1-score row (XGB)
f1_row = [
    'F1-score',
    'XGB',
    f"{xgb_best_f1:.3f} ({xgb_best_f1_stage} best)",
    f"{tuned_results['XGB']['f1_score']:.3f}",
    f"{tuned_results['XGB']['f1_score'] - xgb_best_f1:.3f}"
]
table_data.append(f1_row)

# Create DataFrame
df = pd.DataFrame(table_data, columns=[
    'Metric', 'Tuned model', 'Before (reference)', 'After (tuned)', 'Δ'
])

# Save to CSV
output_path = '/home/ubuntu/credit-risk-reproducibility/results/tables/table_4_4_hyperparameter_tuning.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("\nTable 4-4 generated successfully!")
print("\nTable Preview:")
print(df.to_string(index=False))
