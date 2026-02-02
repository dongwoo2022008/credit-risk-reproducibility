"""
Generate Table 4-4: Comparison of Key Performance Metrics Before and After Hyperparameter Tuning
Logic: Find best model per metric AFTER tuning, then compare with that model's best BEFORE tuning
"""

import pandas as pd

# Before (reference): Best performance from Phase 2 merged models (Stage 1-4)
# RF, GB, XGB only (tuned models)

# Stage 1: TF-IDF + Structured
stage1_results = {
    'RF': {'roc_auc': 0.7941, 'recall': 0.7601, 'f1_score': 0.7533},
    'GB': {'roc_auc': 0.8103, 'recall': 0.8018, 'f1_score': 0.7724},
    'XGB': {'roc_auc': 0.7971, 'recall': 0.7645, 'f1_score': 0.7516}
}

# Stage 2: Subword + Structured
stage2_results = {
    'RF': {'roc_auc': 0.7754, 'recall': 0.7660, 'f1_score': 0.7482},
    'GB': {'roc_auc': 0.8081, 'recall': 0.7958, 'f1_score': 0.7711},
    'XGB': {'roc_auc': 0.7966, 'recall': 0.7556, 'f1_score': 0.7556}
}

# Stage 3: MiniLM + Structured
stage3_results = {
    'RF': {'roc_auc': 0.7396, 'recall': 0.8331, 'f1_score': 0.7443},
    'GB': {'roc_auc': 0.8011, 'recall': 0.8033, 'f1_score': 0.7711},
    'XGB': {'roc_auc': 0.7843, 'recall': 0.7884, 'f1_score': 0.7584}
}

# Stage 4: KoSimCSE + Structured
stage4_results = {
    'RF': {'roc_auc': 0.7040, 'recall': 0.8152, 'f1_score': 0.7226},
    'GB': {'roc_auc': 0.7964, 'recall': 0.8092, 'f1_score': 0.7702},
    'XGB': {'roc_auc': 0.7662, 'recall': 0.7645, 'f1_score': 0.7355}
}

# Combine all stages
all_stages = [stage1_results, stage2_results, stage3_results, stage4_results]
stage_names = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']

# After (tuned): Phase 3 hyperparameter tuning results
tuned_results = {
    'XGB': {'roc_auc': 0.7932, 'recall': 0.7943, 'f1_score': 0.7636},
    'RF': {'roc_auc': 0.7397, 'recall': 0.8256, 'f1_score': 0.7461},
    'GB': {'roc_auc': 0.7883, 'recall': 0.8077, 'f1_score': 0.7612}
}

# Step 1: Find best model per metric AFTER tuning
best_roc_after = max(tuned_results.items(), key=lambda x: x[1]['roc_auc'])
best_recall_after = max(tuned_results.items(), key=lambda x: x[1]['recall'])
best_f1_after = max(tuned_results.items(), key=lambda x: x[1]['f1_score'])

print("After tuning (best per metric):")
print(f"  ROC-AUC: {best_roc_after[0]} = {best_roc_after[1]['roc_auc']:.3f}")
print(f"  Recall: {best_recall_after[0]} = {best_recall_after[1]['recall']:.3f}")
print(f"  F1-score: {best_f1_after[0]} = {best_f1_after[1]['f1_score']:.3f}")

# Step 2: Find that model's best performance BEFORE tuning (across Stage 1-4)
def find_model_best_before(model_name, metric):
    """Find the best performance of a specific model across all stages for a metric"""
    best_val = 0
    best_stage = ''
    for stage_idx, stage in enumerate(all_stages):
        val = stage[model_name][metric]
        if val > best_val:
            best_val = val
            best_stage = stage_names[stage_idx]
    return best_val, best_stage

# ROC-AUC: best model after tuning
roc_model = best_roc_after[0]
roc_after = best_roc_after[1]['roc_auc']
roc_before, roc_stage = find_model_best_before(roc_model, 'roc_auc')

# Recall: best model after tuning
recall_model = best_recall_after[0]
recall_after = best_recall_after[1]['recall']
recall_before, recall_stage = find_model_best_before(recall_model, 'recall')

# F1-score: best model after tuning
f1_model = best_f1_after[0]
f1_after = best_f1_after[1]['f1_score']
f1_before, f1_stage = find_model_best_before(f1_model, 'f1_score')

print("\nBefore tuning (same model's best):")
print(f"  ROC-AUC: {roc_model} = {roc_before:.3f} ({roc_stage})")
print(f"  Recall: {recall_model} = {recall_before:.3f} ({recall_stage})")
print(f"  F1-score: {f1_model} = {f1_before:.3f} ({f1_stage})")

# Build Table 4-4
table_data = []

# ROC-AUC row
roc_row = [
    'ROC-AUC',
    roc_model,
    f"{roc_before:.3f} ({roc_stage} best)",
    f"{roc_after:.3f}",
    f"{roc_after - roc_before:.3f}"
]
table_data.append(roc_row)

# Recall row
recall_row = [
    'Recall',
    recall_model,
    f"{recall_before:.3f} ({recall_stage} best)",
    f"{recall_after:.3f}",
    f"{recall_after - recall_before:.3f}"
]
table_data.append(recall_row)

# F1-score row
f1_row = [
    'F1-score',
    f1_model,
    f"{f1_before:.3f} ({f1_stage} best)",
    f"{f1_after:.3f}",
    f"{f1_after - f1_before:.3f}"
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
