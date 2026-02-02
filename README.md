# Reproducible Research Repository: Credit Risk Assessment with Borrower Narratives

This repository contains the complete reproducible code and data for the research paper:

**"Conditional Value of Borrower Narratives in Credit Risk Prediction: Evidence from a Korean P2P Platform"**

## Overview

This study examines whether borrower narratives add predictive value to default-risk assessment in an early Korean peer-to-peer lending platform (6,057 loan applications, 2006–2016). We compare models using structured variables only, text only, and combined structured-and-text inputs, and diagnose where text contributes beyond average performance.

### Key Findings

- **Structured variables alone** deliver strong baseline performance (GB model: ROC-AUC 0.812)
- **Text-only models** perform near random ranking (ROC-AUC ≈ 0.49-0.50)
- **Adding text** does not improve average ranking beyond the strong structured benchmark
- **Conditional value**: Text provides clear benefits in:
  - Marginal cases (predicted risk 0.3-0.7): +11% ROC-AUC improvement
  - False negative recovery: 18-26% recovery rate
  - High-risk tier: Higher recovery rates (26% vs 18% overall)
  - Longer narratives: +1.2% F1 improvement (vs -3.1% for short text)

## Repository Structure

```
credit-risk-reproducibility/
├── code/                          # All source code
│   ├── config.py                  # Global configuration
│   ├── utils/                     # Utility modules
│   │   ├── data_loader.py         # Data loading and preprocessing
│   │   ├── evaluator.py           # Model evaluation functions
│   │   └── __init__.py
│   ├── phase0_structured_baseline.py    # Phase 0: Structured-only models
│   ├── phase2_merged_models.py          # Phase 2: Merged models (struct + text)
│   ├── phase5_1_uncertainty_analysis.py # Phase 5-1: Marginal vs clear cases
│   └── phase5_2_fn_recovery.py          # Phase 5-2: FN recovery analysis
├── data/                          # Data directory
│   ├── raw/                       # Raw data files
│   │   └── sentiment_scoring.25.12.30.xlsx
│   ├── processed/                 # Processed data files
│   └── splits/                    # Train/test split indices
├── models/                        # Trained models
│   ├── phase0/                    # Structured-only models
│   ├── phase2/                    # Merged models
│   └── phase5/                    # Phase 5 models
├── results/                       # Results directory
│   ├── tables/                    # Generated tables (CSV)
│   └── figures/                   # Generated figures (PNG)
├── docs/                          # Documentation
│   └── REPRODUCIBILITY.md         # Reproducibility guide
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## Quick Start

### Prerequisites

- Python 3.11+
- Required packages: pandas, numpy, scikit-learn, xgboost, openpyxl, matplotlib, seaborn

### Installation

```bash
# Clone the repository
git clone https://github.com/dongwoo2022008/credit-risk-reproducibility.git
cd credit-risk-reproducibility

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

#### Phase 0: Structured-only Baseline

```bash
python code/phase0_structured_baseline.py
```

**Expected output**: Table 4-1 with 8 model performances (GB: ROC-AUC 0.812)

#### Phase 2: Merged Models (Structured + Text)

```bash
python code/phase2_merged_models.py
```

**Expected output**: Performance tables for 4 text representation stages

#### Phase 5: Conditional Effect Analysis

```bash
# Phase 5-1: Uncertainty analysis
python code/phase5_1_uncertainty_analysis.py

# Phase 5-2: FN recovery analysis
python code/phase5_2_fn_recovery.py
```

**Expected output**: Tables 4-7, 4-8, 4-9

## Data

### Raw Data

- **File**: `data/raw/sentiment_scoring.25.12.30.xlsx`
- **Size**: 4.4 MB
- **Samples**: 6,057 loan applications (2006-2016)
- **Features**: 46 columns including:
  - 13 structured variables (credit score, loan amount, income, etc.)
  - 3 text columns (title, loan purpose, repayment plan)
  - 1 target variable (default = 1, repayment = 0)

### Data Split

- **Training set**: 80% (4,845 samples)
- **Test set**: 20% (1,212 samples)
- **Stratified sampling**: Maintains target class proportions
- **Random seed**: 42 (for reproducibility)

## Models

### Phase 0: Structured-only Baseline (8 models)

1. Logistic Regression (LR)
2. Support Vector Machine (SVM)
3. k-Nearest Neighbors (KNN)
4. Decision Tree (DT)
5. Naive Bayes (NB)
6. Random Forest (RF)
7. Gradient Boosting (GB) ← **Best model**
8. XGBoost (XGB)

### Phase 2: Merged Models (Structured + Text)

**Text Representation Stages:**
- Stage 1: TF-IDF (100 features)
- Stage 2: Subword-based features (character n-grams)
- Stage 3: MiniLM sentence embeddings (384 dimensions)
- Stage 4: KoSimCSE sentence embeddings (768 dimensions)

### Phase 5: Conditional Effect Analysis

1. **Uncertainty Interval Analysis**: Marginal (0.3-0.7) vs Clear cases
2. **FN Recovery Analysis**: False negative recovery rates by credit tier
3. **Threshold Sensitivity**: Performance gaps across thresholds
4. **Text Length Analysis**: Information richness effects

## Results

### Main Results (Table 4-1)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| LR    | 0.6988   | 0.6977    | 0.8048 | 0.7474   | 0.7583  |
| SVM   | 0.7120   | 0.7199    | 0.7854 | 0.7512   | 0.7628  |
| KNN   | 0.6469   | 0.6719    | 0.7079 | 0.6894   | 0.7013  |
| DT    | 0.6559   | 0.6873    | 0.6945 | 0.6909   | 0.6513  |
| NB    | 0.6584   | 0.6532    | 0.8167 | 0.7258   | 0.7234  |
| RF    | 0.7409   | 0.7583    | 0.7809 | 0.7695   | 0.8015  |
| **GB**| **0.7417**| **0.7528**| **0.7943**| **0.7730**| **0.8124** |
| XGB   | 0.7442   | 0.7524    | 0.8018 | 0.7763   | 0.8142  |

### Conditional Effects (Table 4-7)

| Case Type | Count | Struct ROC-AUC | Merged ROC-AUC | Improvement |
|-----------|-------|----------------|----------------|-------------|
| Marginal (0.3-0.7) | 503 (41.5%) | 0.6095 | 0.6770 | **+11.08%** |
| Clear | 709 (58.5%) | 0.8606 | 0.8655 | +0.57% |

### FN Recovery (Table 4-9)

| Category | Structured FN | GB+Text Recovery | Recovery Rate |
|----------|---------------|------------------|---------------|
| Overall | 154 | 28 | 18.18% |
| High-risk (bottom 30%) | 19 | 5 | 26.32% |
| Low-risk (top 30%) | 81 | 15 | 18.52% |

## Reproducibility

All results are fully reproducible with:
- **Fixed random seed**: 42
- **Fixed train/test split**: Saved in `data/splits/`
- **Fixed hyperparameters**: Defined in `config.py`
- **Version-controlled code**: All code in `code/` directory

### Verification

To verify results match the paper:

```bash
# Run all phases
python code/phase0_structured_baseline.py
python code/phase2_merged_models.py
python code/phase5_1_uncertainty_analysis.py
python code/phase5_2_fn_recovery.py

# Check results
ls results/tables/
```

Expected tables:
- `table_4_1_phase0_performance.csv` (Phase 0 results)
- `phase2_stage1_tfidf_performance.csv` (Phase 2 Stage 1)
- `table_4_7_uncertainty_analysis.csv` (Phase 5-1)
- `table_4_8_confusion_matrix.csv` (Phase 5-2)
- `table_4_9_fn_recovery_rate.csv` (Phase 5-2)

## Citation

If you use this code or data, please cite:

```bibtex
@article{kim2026conditional,
  title={Conditional Value of Borrower Narratives in Credit Risk Prediction: Evidence from a Korean P2P Platform},
  author={Kim, Dongwoo},
  journal={[Journal Name]},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Author**: Dongwoo Kim
- **Affiliation**: Baekseok University
- **Email**: dongwoo.kim@bu.ac.kr
- **GitHub**: https://github.com/dongwoo2022008

## Acknowledgments

- Data source: Korean P2P lending platform (2006-2016)
- Text processing: KoSimCSE, MiniLM models
- Machine learning: scikit-learn, XGBoost

## Version History

- **v1.0.0** (2026-02-02): Initial release with Phase 0, 2, and 5 implementations

---

**Note**: This repository is designed for reproducibility and transparency in research. All code, data processing steps, and model configurations are documented and version-controlled.
