"""
Generate all 8 figures from the paper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
TABLES_DIR = Path("/home/ubuntu/credit-risk-reproducibility/results/tables")
FIGURES_DIR = Path("/home/ubuntu/credit-risk-reproducibility/results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set matplotlib style - black and white, Arial font
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['black', 'gray', 'darkgray'])

print("="*80)
print("Generating all figures...")
print("="*80)

# ============================================================================
# Figure 1-1: Research framework (conceptual diagram - skip for now)
# ============================================================================
print("\nFigure 1-1: Research framework (conceptual - requires manual design)")
print("  → Skipping (requires conceptual diagram design)")

# ============================================================================
# Figure 4-1: Phase 5 analysis framework (conceptual diagram - skip for now)
# ============================================================================
print("\nFigure 4-1: Phase 5 framework (conceptual - requires manual design)")
print("  → Skipping (requires conceptual diagram design)")

# ============================================================================
# Figure 4-2: Marginal vs Clear cases improvement
# ============================================================================
print("\nGenerating Figure 4-2...")
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Marginal Cases', 'Clear Cases']
roc_auc_improvement = [9.32, -1.87]
f1_improvement = [12.78, -0.35]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, roc_auc_improvement, width, label='ROC-AUC Improvement (%)', 
               color='black', edgecolor='black')
bars2 = ax.bar(x + width/2, f1_improvement, width, label='F1-score Improvement (%)',
               color='gray', edgecolor='black')

ax.set_xlabel('Case Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Improvement Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Text Combination Effect (GB+Text) in Marginal vs. Clear Cases', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(frameon=True, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure_4_2_marginal_vs_clear.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure_4_2_marginal_vs_clear.png")

# ============================================================================
# Figure 4-3: FN Recovery Rate by group
# ============================================================================
print("\nGenerating Figure 4-3...")
fig, ax = plt.subplots(figsize=(10, 6))

groups = ['Overall', 'High-risk\n(bottom 30%)', 'Low-risk\n(top 30%)']
rf_recovery = [41.30, 52.63, 39.71]
gb_recovery = [26.81, 36.84, 19.12]

x = np.arange(len(groups))
width = 0.35

bars1 = ax.bar(x - width/2, rf_recovery, width, label='RF+Text', 
               color='black', edgecolor='black')
bars2 = ax.bar(x + width/2, gb_recovery, width, label='GB+Text',
               color='gray', edgecolor='black')

ax.set_xlabel('Group', fontsize=12, fontweight='bold')
ax.set_ylabel('FN Recovery Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Group-wise FN Recovery Rate', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend(frameon=True, edgecolor='black')
ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure_4_3_fn_recovery_rate.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure_4_3_fn_recovery_rate.png")

# ============================================================================
# Figure 4-4: Threshold sensitivity (F1-score gap)
# ============================================================================
print("\nGenerating Figure 4-4...")
table_4_10 = pd.read_csv(TABLES_DIR / 'table_4_10_threshold_sensitivity.csv')

fig, ax = plt.subplots(figsize=(10, 6))

rf_data = table_4_10[table_4_10['Model'] == 'RF+Text']
gb_data = table_4_10[table_4_10['Model'] == 'GB+Text']

ax.plot(rf_data['Threshold'], rf_data['F1-score Gap (%)'], 
        marker='o', linewidth=2, markersize=8, color='black', label='RF+Text')
ax.plot(gb_data['Threshold'], gb_data['F1-score Gap (%)'], 
        marker='s', linewidth=2, markersize=8, color='gray', label='GB+Text')

ax.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-score Gap (%)', fontsize=12, fontweight='bold')
ax.set_title('Threshold Sensitivity of F1-score Gap\n(High-risk vs. Low-risk Groups)', 
             fontsize=14, fontweight='bold')
ax.legend(frameon=True, edgecolor='black')
ax.grid(True, alpha=0.3, linestyle='--', color='gray')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure_4_4_threshold_f1_gap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure_4_4_threshold_f1_gap.png")

# ============================================================================
# Figure 4-5: Threshold sensitivity (Recall gap)
# ============================================================================
print("\nGenerating Figure 4-5...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(rf_data['Threshold'], rf_data['Recall Gap (%)'], 
        marker='o', linewidth=2, markersize=8, color='black', label='RF+Text')
ax.plot(gb_data['Threshold'], gb_data['Recall Gap (%)'], 
        marker='s', linewidth=2, markersize=8, color='gray', label='GB+Text')

ax.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('Recall Gap (%)', fontsize=12, fontweight='bold')
ax.set_title('Threshold Sensitivity of Recall Gap\n(High-risk vs. Low-risk Groups)', 
             fontsize=14, fontweight='bold')
ax.legend(frameon=True, edgecolor='black')
ax.grid(True, alpha=0.3, linestyle='--', color='gray')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure_4_5_threshold_recall_gap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure_4_5_threshold_recall_gap.png")

# ============================================================================
# Figure 4-6: Default rate by text length decile
# ============================================================================
print("\nGenerating Figure 4-6...")
table_4_11 = pd.read_csv(TABLES_DIR / 'table_4_11_length_decile_analysis.csv')

fig, ax = plt.subplots(figsize=(12, 6))

deciles = range(1, 11)
default_rates = table_4_11['Observed default rate (%)'].values

ax.plot(deciles, default_rates, marker='o', linewidth=2, markersize=8, color='black')
ax.fill_between(deciles, default_rates, alpha=0.2, color='gray')

ax.set_xlabel('Text Length Decile', fontsize=12, fontweight='bold')
ax.set_ylabel('Observed Default Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Actual Default Rate Trends by Text Length Decile', 
             fontsize=14, fontweight='bold')
ax.set_xticks(deciles)
ax.set_xticklabels([f'{i}' for i in deciles])
ax.grid(True, alpha=0.3, linestyle='--', color='gray')

# Add trend line
z = np.polyfit(deciles, default_rates, 1)
p = np.poly1d(z)
ax.plot(deciles, p(deciles), linestyle='--', color='darkgray', linewidth=1.5, 
        label=f'Trend (r={-0.9439:.3f})')
ax.legend(frameon=True, edgecolor='black')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure_4_6_length_default_rate.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure_4_6_length_default_rate.png")

# ============================================================================
# Figure 4-7: Length-Prediction relationship
# ============================================================================
print("\nGenerating Figure 4-7...")
table_4_12 = pd.read_csv(TABLES_DIR / 'table_4_12_length_correlation.csv')

fig, ax = plt.subplots(figsize=(10, 6))

variables = table_4_12['Variable'].values
correlations = table_4_12['Correlation_with_Length'].values

colors = ['black', 'darkgray', 'gray', 'lightgray']
bars = ax.barh(variables, correlations, color=colors, edgecolor='black')

ax.set_xlabel('Correlation with Text Length', fontsize=12, fontweight='bold')
ax.set_ylabel('Variable', fontsize=12, fontweight='bold')
ax.set_title('Comparison of Text Length–Prediction Probability Relationship', 
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3, linestyle='--', color='gray')

# Add value labels
for i, (var, corr) in enumerate(zip(variables, correlations)):
    ax.text(corr - 0.01, i, f'{corr:.4f}', 
            ha='right', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure_4_7_length_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure_4_7_length_correlation.png")

print("\n" + "="*80)
print("Figure generation complete!")
print("="*80)
print("\nGenerated figures:")
print("  • Figure 4-2: Marginal vs Clear cases")
print("  • Figure 4-3: FN Recovery Rate")
print("  • Figure 4-4: Threshold sensitivity (F1-score)")
print("  • Figure 4-5: Threshold sensitivity (Recall)")
print("  • Figure 4-6: Default rate by text length")
print("  • Figure 4-7: Length-Prediction correlation")
print("\nSkipped (conceptual diagrams):")
print("  • Figure 1-1: Research framework")
print("  • Figure 4-1: Phase 5 analysis framework")
print("="*80)
