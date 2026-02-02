"""
Generate Table 4-4: Comparison of Key Performance Metrics Before and After Hyperparameter Tuning
"""

import pandas as pd

# Before (reference): Best performance from Phase 2 merged models (Stage 1-4)
# Only consider models that were tuned: XGB, RF, GB, LR

# Stage 1: TF-IDF + Structured
stage1_results = {
    'LR': {'roc_auc': 0.7545, 'recall': 0.7645, 'f1_score': 0.7282},
    'RF': {'roc_auc': 0.7941, 'recall': 0.7601, 'f1_score': 0.7533},
    'GB': {'roc_auc': 0.8103, 'recall': 0.8018, 'f1_score': 0.7724},
    'XGB': {'roc_auc': 0.7971, 'recall': 0.7645, 'f1_score': 0.7516}
}

# Stage 2: Subword + Structured
stage2_results = {
    'LR': {'roc_auc': 0.7622, 'recall': 0.7779, 'f1_score': 0.7321},
    'RF': {'roc_auc': 0.7754, 'recall': 0.7660, 'f1_score': 0.7482},
    'GB': {'roc_auc': 0.8081, 'recall': 0.7958, 'f1_score': 0.7711},
    'XGB': {'roc_auc': 0.7966, 'recall': 0.7556, 'f1_score': 0.7556}
}

# Stage 3: MiniLM + Structured
stage3_results = {
    'LR': {'roc_auc': 0.7589, 'recall': 0.7779, 'f1_score': 0.7321},
    'RF': {'roc_auc': 0.7396, 'recall': 0.8331, 'f1_score': 0.7443},
    'GB': {'roc_auc': 0.8011, 'recall': 0.8033, 'f1_score': 0.7711},
    'XGB': {'roc_auc': 0.7843, 'recall': 0.7884, 'f1_score': 0.7584}
}

# Stage 4: KoSimCSE + Structured
stage4_results = {
    'LR': {'roc_auc': 0.7249, 'recall': 0.7630, 'f1_score': 0.7211},
    'RF': {'roc_auc': 0.7040, 'recall': 0.8152, 'f1_score': 0.7226},
    'GB': {'roc_auc': 0.7964, 'recall': 0.8092, 'f1_score': 0.7702},
    'XGB': {'roc_auc': 0.7662, 'recall': 0.7645, 'f1_score': 0.7355}
}

# Combine all stages to find best performance per metric (only tuned models)
all_stages = [stage1_results, stage2_results, stage3_results, stage4_results]
stage_names = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']

# Find best ROC-AUC among tuned models
best_roc_auc = 0
best_roc_model = ''
best_roc_stage = ''
for stage_idx, stage in enumerate(all_stages):
    for model, metrics in stage.items():
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_roc_model = model
            best_roc_stage = stage_names[stage_idx]

# Find best Recall among tuned models
best_recall = 0
best_recall_model = ''
best_recall_stage = ''
for stage_idx, stage in enumerate(all_stages):
    for model, metrics in stage.items():
        if metrics['recall'] > best_recall:
            best_recall = metrics['recall']
            best_recall_model = model
            best_recall_stage = stage_names[stage_idx]

# Find best F1-score among tuned models
best_f1 = 0
best_f1_model = ''
best_f1_stage = ''
for stage_idx, stage in enumerate(all_stages):
    for model, metrics in stage.items():
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_f1_model = model
            best_f1_stage = stage_names[stage_idx]

print(f"Best ROC-AUC (tuned models only): {best_roc_auc:.3f} ({best_roc_model}, {best_roc_stage})")
print(f"Best Recall (tuned models only): {best_recall:.3f} ({best_recall_model}, {best_recall_stage})")
print(f"Best F1-score (tuned models only): {best_f1:.3f} ({best_f1_model}, {best_f1_stage})")

# After (tuned): Phase 3 hyperparameter tuning results
tuned_results = {
    'XGB': {'roc_auc': 0.7932, 'recall': 0.7943, 'f1_score': 0.7636},
    'RF': {'roc_auc': 0.7397, 'recall': 0.8256, 'f1_score': 0.7461},
    'GB': {'roc_auc': 0.7883, 'recall': 0.8077, 'f1_score': 0.7612},
    'LR': {'roc_auc': 0.7612, 'recall': 0.7750, 'f1_score': 0.7334}
}

# Build Table 4-4
table_data = []

# ROC-AUC row
roc_row = [
    'ROC-AUC',
    best_roc_model,
    f"{best_roc_auc:.3f} ({best_roc_stage} best)",
    f"{tuned_results[best_roc_model]['roc_auc']:.3f}",
    f"{tuned_results[best_roc_model]['roc_auc'] - best_roc_auc:.3f}"
]
table_data.append(roc_row)

# Recall row
recall_row = [
    'Recall',
    best_recall_model,
    f"{best_recall:.3f} ({best_recall_stage} best)",
    f"{tuned_results[best_recall_model]['recall']:.3f}",
    f"{tuned_results[best_recall_model]['recall'] - best_recall:.3f}"
]
table_data.append(recall_row)

# F1-score row
f1_row = [
    'F1-score',
    best_f1_model,
    f"{best_f1:.3f} ({best_f1_stage} best)",
    f"{tuned_results[best_f1_model]['f1_score']:.3f}",
    f"{tuned_results[best_f1_model]['f1_score'] - best_f1:.3f}"
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
