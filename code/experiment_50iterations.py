"""
50-Iteration Experimental Design
Approach: "5 tuning runs → fixed parameters → 50 evaluation runs"

Structure:
1. split_data(seed) -> (X_train, y_train, X_test, y_test)
2. fit_and_eval_phase(phase_id, X_train, y_train, X_test, y_test, seed, params=None) -> dict
3. run_one_seed(seed, fixed_params=None) -> list[dict]
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

import config

# ============================================================================
# 1. Data Split Function
# ============================================================================

def split_data(seed, test_size=0.2):
    """
    Split data into train/test with stratification
    
    Args:
        seed: random seed for reproducibility
        test_size: proportion of test set (default 0.2)
    
    Returns:
        X_train, y_train, X_test, y_test, feature_names
    """
    print(f"\n{'='*80}")
    print(f"Split Data (Seed: {seed})")
    print(f"{'='*80}")
    
    # Load original data
    data_path = config.DATA_DIR / "raw" / "sentiment_scoring.25.12.30.xlsx"
    df = pd.read_excel(data_path)
    
    print(f"Total samples: {len(df)}")
    
    # Create target variable (채무불이행 = 1, others = 0)
    df['target'] = (df['상환결과'] == '채무불이행').astype(int)
    
    # Select structured variables
    structured_vars = [
        '대출시기', '취소횟수', '실패횟수', '성공횟수', '총횟수', '성공률',
        '성별(남0)', '나이', '지역(수도권0)', '신용평점', '월DTI', '연소득(만원)', '월소득(만원)'
    ]
    
    X_structured = df[structured_vars].values
    
    # Encode categorical variables
    le_purpose = LabelEncoder()
    le_insurance = LabelEncoder()
    
    df['대출용도_encoded'] = le_purpose.fit_transform(df['대출용도'])
    df['4대보험_encoded'] = le_insurance.fit_transform(df['4대보험'])
    
    X_structured = np.column_stack([
        X_structured,
        df['대출용도_encoded'].values,
        df['4대보험_encoded'].values
    ])
    
    # Create TF-IDF features from text
    text_cols = ['제목', '신청목적', '상환계획']
    df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
    
    tfidf = TfidfVectorizer(max_features=100)
    X_tfidf = tfidf.fit_transform(df['combined_text']).toarray()
    
    # Merge features
    X = np.column_stack([X_structured, X_tfidf])
    y = df['target'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"Train default rate: {y_train.mean():.2%}, Test default rate: {y_test.mean():.2%}")
    print(f"Features: {X.shape[1]}")
    
    feature_names = structured_vars + ['대출용도', '4대보험'] + [f'tfidf_{i}' for i in range(100)]
    
    return X_train, y_train, X_test, y_test, feature_names

# ============================================================================
# 2. Phase Evaluation Function
# ============================================================================

def fit_and_eval_phase(phase_id, X_train, y_train, X_test, y_test, seed, params=None):
    """
    Fit and evaluate a single phase
    
    Args:
        phase_id: 'phase0_lr', 'phase0_xgb', etc.
        X_train, y_train: training data
        X_test, y_test: test data
        seed: random seed
        params: fixed hyperparameters (optional)
    
    Returns:
        dict with results
    """
    results = {
        'seed': seed,
        'phase': phase_id,
        'roc_auc': None,
        'recall': None,
        'f1_score': None
    }
    
    # Define models
    if phase_id == 'phase0_lr':
        model = LogisticRegression(random_state=seed, max_iter=1000)
    elif phase_id == 'phase0_svm':
        model = SVC(probability=True, random_state=seed)
    elif phase_id == 'phase0_knn':
        model = KNeighborsClassifier()
    elif phase_id == 'phase0_dt':
        model = DecisionTreeClassifier(random_state=seed)
    elif phase_id == 'phase0_nb':
        model = GaussianNB()
    elif phase_id == 'phase0_rf':
        if params and 'rf' in params:
            model = RandomForestClassifier(**params['rf'], random_state=seed)
        else:
            model = RandomForestClassifier(random_state=seed)
    elif phase_id == 'phase0_gb':
        if params and 'gb' in params:
            model = GradientBoostingClassifier(**params['gb'], random_state=seed)
        else:
            model = GradientBoostingClassifier(random_state=seed)
    elif phase_id == 'phase0_xgb':
        if params and 'xgb' in params:
            model = XGBClassifier(**params['xgb'], random_state=seed, eval_metric='logloss')
        else:
            model = XGBClassifier(random_state=seed, eval_metric='logloss')
    else:
        raise ValueError(f"Unknown phase_id: {phase_id}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    results['roc_auc'] = roc_auc_score(y_test, y_proba)
    results['recall'] = recall_score(y_test, y_pred)
    results['f1_score'] = f1_score(y_test, y_pred)
    
    return results

# ============================================================================
# 3. OOF-based Ensemble Functions
# ============================================================================

def create_oof_predictions(base_models, X_train, y_train, seed, n_folds=5):
    """
    Create Out-of-Fold predictions for meta-learner training
    
    Args:
        base_models: dict of {'model_name': model_instance}
        X_train, y_train: training data
        seed: random seed
        n_folds: number of CV folds
    
    Returns:
        oof_predictions: (n_train, n_models) array
        trained_models: list of trained base models on full train set
    """
    n_samples = len(X_train)
    n_models = len(base_models)
    oof_predictions = np.zeros((n_samples, n_models))
    
    # Create CV folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Generate OOF predictions
    for model_idx, (model_name, model) in enumerate(base_models.items()):
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            # Clone and train model
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_tr, y_tr)
            
            # Predict on validation set
            oof_predictions[val_idx, model_idx] = fold_model.predict_proba(X_val)[:, 1]
    
    # Train base models on full train set
    trained_models = {}
    for model_name, model in base_models.items():
        trained_model = clone(model)
        trained_model.fit(X_train, y_train)
        trained_models[model_name] = trained_model
    
    return oof_predictions, trained_models

def eval_ensemble_stacking(X_train, y_train, X_test, y_test, seed, params=None):
    """
    Evaluate Stacking ensemble with OOF
    """
    # Define base models
    base_models = {
        'rf': RandomForestClassifier(**(params['rf'] if params and 'rf' in params else {}), random_state=seed),
        'gb': GradientBoostingClassifier(**(params['gb'] if params and 'gb' in params else {}), random_state=seed),
        'xgb': XGBClassifier(**(params['xgb'] if params and 'xgb' in params else {}), random_state=seed, eval_metric='logloss')
    }
    
    # Create OOF predictions
    oof_preds, trained_models = create_oof_predictions(base_models, X_train, y_train, seed)
    
    # Train meta-learner on OOF predictions
    meta_learner = LogisticRegression(random_state=seed)
    meta_learner.fit(oof_preds, y_train)
    
    # Generate test predictions
    test_preds = np.column_stack([
        model.predict_proba(X_test)[:, 1] for model in trained_models.values()
    ])
    
    # Final prediction
    y_pred_proba = meta_learner.predict_proba(test_preds)[:, 1]
    y_pred = meta_learner.predict(test_preds)
    
    return {
        'seed': seed,
        'phase': 'ensemble_stacking',
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

def eval_ensemble_blending(X_train, y_train, X_test, y_test, seed, params=None):
    """
    Evaluate Blending ensemble with OOF (simple linear combination)
    """
    # Same as stacking but with simpler meta-learner
    base_models = {
        'rf': RandomForestClassifier(**(params['rf'] if params and 'rf' in params else {}), random_state=seed),
        'gb': GradientBoostingClassifier(**(params['gb'] if params and 'gb' in params else {}), random_state=seed),
        'xgb': XGBClassifier(**(params['xgb'] if params and 'xgb' in params else {}), random_state=seed, eval_metric='logloss')
    }
    
    oof_preds, trained_models = create_oof_predictions(base_models, X_train, y_train, seed)
    
    # Simple linear blending (Ridge regression)
    from sklearn.linear_model import Ridge
    meta_learner = Ridge(random_state=seed)
    meta_learner.fit(oof_preds, y_train)
    
    test_preds = np.column_stack([
        model.predict_proba(X_test)[:, 1] for model in trained_models.values()
    ])
    
    y_pred_proba = meta_learner.predict(test_preds)
    y_pred_proba = np.clip(y_pred_proba, 0, 1)  # Clip to [0, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    return {
        'seed': seed,
        'phase': 'ensemble_blending',
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

def eval_ensemble_voting(X_train, y_train, X_test, y_test, seed, voting_type='soft', params=None):
    """
    Evaluate Voting ensemble
    """
    base_models = [
        ('rf', RandomForestClassifier(**(params['rf'] if params and 'rf' in params else {}), random_state=seed)),
        ('gb', GradientBoostingClassifier(**(params['gb'] if params and 'gb' in params else {}), random_state=seed)),
        ('xgb', XGBClassifier(**(params['xgb'] if params and 'xgb' in params else {}), random_state=seed, eval_metric='logloss'))
    ]
    
    voting = VotingClassifier(estimators=base_models, voting=voting_type)
    voting.fit(X_train, y_train)
    
    y_pred = voting.predict(X_test)
    if voting_type == 'soft':
        y_pred_proba = voting.predict_proba(X_test)[:, 1]
    else:
        # Hard voting: use average of base model probabilities from fitted models
        y_pred_proba = np.mean([voting.estimators_[i].predict_proba(X_test)[:, 1] for i in range(len(voting.estimators_))], axis=0)
    
    return {
        'seed': seed,
        'phase': f'ensemble_voting_{voting_type[0]}',  # voting_h or voting_s
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

# ============================================================================
# 4. Run One Seed Function
# ============================================================================

def run_one_seed(seed, fixed_params=None):
    """
    Run all phases for one seed
    
    Args:
        seed: random seed
        fixed_params: fixed hyperparameters from tuning phase
    
    Returns:
        list of result dicts
    """
    print(f"\n{'='*80}")
    print(f"Running Seed: {seed}")
    print(f"{'='*80}")
    
    # Split data
    X_train, y_train, X_test, y_test, feature_names = split_data(seed)
    
    results = []
    
    # Phase 0: Baseline models
    phase0_models = ['lr', 'svm', 'knn', 'dt', 'nb', 'rf', 'gb', 'xgb']
    for model_name in phase0_models:
        phase_id = f'phase0_{model_name}'
        result = fit_and_eval_phase(phase_id, X_train, y_train, X_test, y_test, seed, fixed_params)
        results.append(result)
        print(f"  {phase_id}: ROC-AUC={result['roc_auc']:.4f}, Recall={result['recall']:.4f}, F1={result['f1_score']:.4f}")
    
    # Ensemble methods
    results.append(eval_ensemble_voting(X_train, y_train, X_test, y_test, seed, 'hard', fixed_params))
    results.append(eval_ensemble_voting(X_train, y_train, X_test, y_test, seed, 'soft', fixed_params))
    results.append(eval_ensemble_blending(X_train, y_train, X_test, y_test, seed, fixed_params))
    results.append(eval_ensemble_stacking(X_train, y_train, X_test, y_test, seed, fixed_params))
    
    print(f"  Completed seed {seed}: {len(results)} results")
    
    return results

# ============================================================================
# Main execution will be in separate scripts
# ============================================================================

if __name__ == '__main__':
    # Test with one seed
    test_results = run_one_seed(seed=1)
    print(f"\nTest completed: {len(test_results)} results")
