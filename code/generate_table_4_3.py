"""
Generate Table 4-3: Stage-by-Stage Best Performance Comparison Against Baseline Models
Matches paper format exactly with LR and XGB baselines only
"""

import pandas as pd
import numpy as np

# Phase 0 (Stage 0) - Baseline with structured variables only
phase0_results = {
    'LR': {'roc_auc': 0.7583, 'recall': 0.8048, 'f1_score': 0.7474},
    'XGB': {'roc_auc': 0.8142, 'recall': 0.8018, 'f1_score': 0.7763}
}

# Stage 1: TF-IDF (100 features) + Structured
stage1_results = {
    'LR': {'roc_auc': 0.7545, 'recall': 0.7645, 'f1_score': 0.7282},
    'SVM': {'roc_auc': 0.7497, 'recall': 0.7541, 'f1_score': 0.7260},
    'KNN': {'roc_auc': 0.6996, 'recall': 0.7168, 'f1_score': 0.6986},
    'DT': {'roc_auc': 0.6820, 'recall': 0.7079, 'f1_score': 0.7132},
    'NB': {'roc_auc': 0.6409, 'recall': 0.9016, 'f1_score': 0.6891},
    'RF': {'roc_auc': 0.7941, 'recall': 0.7601, 'f1_score': 0.7533},
    'GB': {'roc_auc': 0.8103, 'recall': 0.8018, 'f1_score': 0.7724},
    'XGB': {'roc_auc': 0.7971, 'recall': 0.7645, 'f1_score': 0.7516}
}

# Stage 2: Subword embeddings + Structured
stage2_results = {
    'LR': {'roc_auc': 0.7622, 'recall': 0.7779, 'f1_score': 0.7321},
    'SVM': {'roc_auc': 0.7501, 'recall': 0.7526, 'f1_score': 0.7251},
    'KNN': {'roc_auc': 0.7075, 'recall': 0.7109, 'f1_score': 0.6933},
    'DT': {'roc_auc': 0.6251, 'recall': 0.6587, 'f1_score': 0.6627},
    'NB': {'roc_auc': 0.6516, 'recall': 0.2146, 'f1_score': 0.3134},
    'RF': {'roc_auc': 0.7754, 'recall': 0.7660, 'f1_score': 0.7482},
    'GB': {'roc_auc': 0.8081, 'recall': 0.7958, 'f1_score': 0.7711},
    'XGB': {'roc_auc': 0.7966, 'recall': 0.7556, 'f1_score': 0.7556}
}

# Stage 3: MiniLM (384-dim) + Structured
stage3_results = {
    'LR': {'roc_auc': 0.7589, 'recall': 0.7779, 'f1_score': 0.7321},
    'SVM': {'roc_auc': 0.7492, 'recall': 0.7571, 'f1_score': 0.7273},
    'KNN': {'roc_auc': 0.7053, 'recall': 0.7288, 'f1_score': 0.7046},
    'DT': {'roc_auc': 0.6057, 'recall': 0.6587, 'f1_score': 0.6524},
    'NB': {'roc_auc': 0.5950, 'recall': 0.6393, 'f1_score': 0.6133},
    'RF': {'roc_auc': 0.7396, 'recall': 0.8331, 'f1_score': 0.7443},
    'GB': {'roc_auc': 0.8011, 'recall': 0.8033, 'f1_score': 0.7711},
    'XGB': {'roc_auc': 0.7843, 'recall': 0.7884, 'f1_score': 0.7584}
}

# Stage 4: KoSimCSE (768-dim) + Structured
stage4_results = {
    'LR': {'roc_auc': 0.7249, 'recall': 0.7630, 'f1_score': 0.7211},
    'SVM': {'roc_auc': 0.6382, 'recall': 0.8316, 'f1_score': 0.6971},
    'KNN': {'roc_auc': 0.6431, 'recall': 0.7019, 'f1_score': 0.6653},
    'DT': {'roc_auc': 0.6137, 'recall': 0.6766, 'f1_score': 0.6637},
    'NB': {'roc_auc': 0.5776, 'recall': 0.7213, 'f1_score': 0.6462},
    'RF': {'roc_auc': 0.7040, 'recall': 0.8152, 'f1_score': 0.7226},
    'GB': {'roc_auc': 0.7964, 'recall': 0.8092, 'f1_score': 0.7702},
    'XGB': {'roc_auc': 0.7662, 'recall': 0.7645, 'f1_score': 0.7355}
}

def find_best_model_per_metric(stage_results, metric):
    """Find the best model for a given metric in a stage"""
    best_model = max(stage_results.items(), key=lambda x: x[1][metric])
    return best_model[0], best_model[1][metric]

# Build table data
table_data = []

# LR Baseline
lr_baseline = phase0_results['LR']
lr_row_roc = ['LR', 'ROC-AUC', f"{lr_baseline['roc_auc']:.3f}"]
lr_row_recall = ['LR', 'Recall', f"{lr_baseline['recall']:.3f}"]
lr_row_f1 = ['LR', 'F1-score', f"{lr_baseline['f1_score']:.3f}"]

# Stage 1
model_roc, val_roc = find_best_model_per_metric(stage1_results, 'roc_auc')
lr_row_roc.append(f"{val_roc:.3f} ({model_roc})")
model_recall, val_recall = find_best_model_per_metric(stage1_results, 'recall')
lr_row_recall.append(f"{val_recall:.3f} ({model_recall})")
model_f1, val_f1 = find_best_model_per_metric(stage1_results, 'f1_score')
lr_row_f1.append(f"{val_f1:.3f} ({model_f1})")

# Stage 2
model_roc, val_roc = find_best_model_per_metric(stage2_results, 'roc_auc')
lr_row_roc.append(f"{val_roc:.3f} ({model_roc})")
model_recall, val_recall = find_best_model_per_metric(stage2_results, 'recall')
lr_row_recall.append(f"{val_recall:.3f} ({model_recall})")
model_f1, val_f1 = find_best_model_per_metric(stage2_results, 'f1_score')
lr_row_f1.append(f"{val_f1:.3f} ({model_f1})")

# Stage 3
model_roc, val_roc = find_best_model_per_metric(stage3_results, 'roc_auc')
lr_row_roc.append(f"{val_roc:.3f} ({model_roc})")
model_recall, val_recall = find_best_model_per_metric(stage3_results, 'recall')
lr_row_recall.append(f"{val_recall:.3f} ({model_recall})")
model_f1, val_f1 = find_best_model_per_metric(stage3_results, 'f1_score')
lr_row_f1.append(f"{val_f1:.3f} ({model_f1})")

# Stage 4
model_roc, val_roc = find_best_model_per_metric(stage4_results, 'roc_auc')
lr_row_roc.append(f"{val_roc:.3f} ({model_roc})")
model_recall, val_recall = find_best_model_per_metric(stage4_results, 'recall')
lr_row_recall.append(f"{val_recall:.3f} ({model_recall})")
model_f1, val_f1 = find_best_model_per_metric(stage4_results, 'f1_score')
lr_row_f1.append(f"{val_f1:.3f} ({model_f1})")

# Calculate Δ for LR (compare baseline against MAX across Stage 1-4)
max_roc_lr = max([float(lr_row_roc[i].split()[0]) for i in range(3, 7)])
max_recall_lr = max([float(lr_row_recall[i].split()[0]) for i in range(3, 7)])
max_f1_lr = max([float(lr_row_f1[i].split()[0]) for i in range(3, 7)])

delta_roc_lr = ((max_roc_lr - lr_baseline['roc_auc']) / lr_baseline['roc_auc']) * 100
delta_recall_lr = ((max_recall_lr - lr_baseline['recall']) / lr_baseline['recall']) * 100
delta_f1_lr = ((max_f1_lr - lr_baseline['f1_score']) / lr_baseline['f1_score']) * 100

lr_row_roc.append(f"{delta_roc_lr:+.1f}%")
lr_row_recall.append(f"{delta_recall_lr:+.1f}%")
lr_row_f1.append(f"{delta_f1_lr:+.1f}%")

table_data.extend([lr_row_roc, lr_row_recall, lr_row_f1])

# XGB Baseline
xgb_baseline = phase0_results['XGB']
xgb_row_roc = ['XGB', 'ROC-AUC', f"{xgb_baseline['roc_auc']:.3f}"]
xgb_row_recall = ['XGB', 'Recall', f"{xgb_baseline['recall']:.3f}"]
xgb_row_f1 = ['XGB', 'F1-score', f"{xgb_baseline['f1_score']:.3f}"]

# Stage 1
model_roc, val_roc = find_best_model_per_metric(stage1_results, 'roc_auc')
xgb_row_roc.append(f"{val_roc:.3f} ({model_roc})")
model_recall, val_recall = find_best_model_per_metric(stage1_results, 'recall')
xgb_row_recall.append(f"{val_recall:.3f} ({model_recall})")
model_f1, val_f1 = find_best_model_per_metric(stage1_results, 'f1_score')
xgb_row_f1.append(f"{val_f1:.3f} ({model_f1})")

# Stage 2
model_roc, val_roc = find_best_model_per_metric(stage2_results, 'roc_auc')
xgb_row_roc.append(f"{val_roc:.3f} ({model_roc})")
model_recall, val_recall = find_best_model_per_metric(stage2_results, 'recall')
xgb_row_recall.append(f"{val_recall:.3f} ({model_recall})")
model_f1, val_f1 = find_best_model_per_metric(stage2_results, 'f1_score')
xgb_row_f1.append(f"{val_f1:.3f} ({model_f1})")

# Stage 3
model_roc, val_roc = find_best_model_per_metric(stage3_results, 'roc_auc')
xgb_row_roc.append(f"{val_roc:.3f} ({model_roc})")
model_recall, val_recall = find_best_model_per_metric(stage3_results, 'recall')
xgb_row_recall.append(f"{val_recall:.3f} ({model_recall})")
model_f1, val_f1 = find_best_model_per_metric(stage3_results, 'f1_score')
xgb_row_f1.append(f"{val_f1:.3f} ({model_f1})")

# Stage 4
model_roc, val_roc = find_best_model_per_metric(stage4_results, 'roc_auc')
xgb_row_roc.append(f"{val_roc:.3f} ({model_roc})")
model_recall, val_recall = find_best_model_per_metric(stage4_results, 'recall')
xgb_row_recall.append(f"{val_recall:.3f} ({model_recall})")
model_f1, val_f1 = find_best_model_per_metric(stage4_results, 'f1_score')
xgb_row_f1.append(f"{val_f1:.3f} ({model_f1})")

# Calculate Δ for XGB (compare baseline against MAX across Stage 1-4)
max_roc_xgb = max([float(xgb_row_roc[i].split()[0]) for i in range(3, 7)])
max_recall_xgb = max([float(xgb_row_recall[i].split()[0]) for i in range(3, 7)])
max_f1_xgb = max([float(xgb_row_f1[i].split()[0]) for i in range(3, 7)])

delta_roc_xgb = ((max_roc_xgb - xgb_baseline['roc_auc']) / xgb_baseline['roc_auc']) * 100
delta_recall_xgb = ((max_recall_xgb - xgb_baseline['recall']) / xgb_baseline['recall']) * 100
delta_f1_xgb = ((max_f1_xgb - xgb_baseline['f1_score']) / xgb_baseline['f1_score']) * 100

xgb_row_roc.append(f"{delta_roc_xgb:+.1f}%")
xgb_row_recall.append(f"{delta_recall_xgb:+.1f}%")
xgb_row_f1.append(f"{delta_f1_xgb:+.1f}%")

table_data.extend([xgb_row_roc, xgb_row_recall, xgb_row_f1])

# Create DataFrame
df = pd.DataFrame(table_data, columns=[
    'Baseline Model', 'Metric', 'Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Δ'
])

# Save to CSV
output_path = '/home/ubuntu/credit-risk-reproducibility/results/tables/table_4_3_stage_comparison.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("Table 4-3 generated successfully!")
print("\nTable Preview:")
print(df.to_string(index=False))
