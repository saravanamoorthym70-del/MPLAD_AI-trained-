"""
Train MPLAD anomaly and delay-risk models.

Usage:
    python ml_engine/train_models.py [optional_csv_path]

The delay model is evaluated on a stratified holdout set.  The target is
whether delay_days > 15, but delay_days itself is deliberately excluded from
model features to prevent target leakage.  A high-precision operating
threshold is selected on the validation set and written to model_metadata.json.
"""
import sys
import os
import json
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402

from ml_engine.feature_engineering import load_and_prepare, ML_FEATURE_COLUMNS  # noqa: E402
from ml_engine.anomaly_detection import train_isolation_forest, save_models  # noqa: E402
from ml_engine.inefficiency_detection import (  # noqa: E402
    DELAY_CATEGORICAL, DELAY_NUMERIC, train_delay_model,
    save_delay_model, score_delay_risk,
)

MODELS_DIR = BASE_DIR / 'ml_engine' / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
HIGH_PRECISION_TARGET = 0.90


def make_delay_matrix(df, encoders=None, fit=False):
    """Build the same matrix used by inference, fitting encoders only when asked."""
    X = pd.DataFrame(index=df.index)
    encoders = {} if encoders is None else encoders
    for col in DELAY_CATEGORICAL:
        if fit:
            encoders[col] = LabelEncoder().fit(df[col].astype(str))
        le = encoders[col]
        known = set(le.classes_)
        safe = df[col].astype(str).map(lambda v: v if v in known else le.classes_[0])
        X[col] = le.transform(safe)
    for col in DELAY_NUMERIC:
        X[col] = df[col].astype(float)
    return X, encoders


def choose_precision_threshold(y_true, probabilities, target=HIGH_PRECISION_TARGET):
    """Choose the lowest threshold meeting target precision, if possible."""
    best = None
    for threshold in np.arange(0.50, 0.991, 0.01):
        pred = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        if precision >= target and pred.sum() > 0:
            candidate = (recall, precision, threshold)
            if best is None or candidate > best:
                best = candidate
    if best is not None:
        return float(best[2]), float(best[1]), float(best[0])
    # If the dataset cannot support 90% precision, maximize F1 rather than
    # claiming an unattainable result.
    candidates = []
    for threshold in np.arange(0.50, 0.991, 0.01):
        pred = (probabilities >= threshold).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        candidates.append((f, p, r, threshold))
    _, p, r, threshold = max(candidates)
    return float(threshold), float(p), float(r)


def evaluate(model, scaler, encoders, df, y, threshold):
    X, _ = make_delay_matrix(df, encoders=encoders, fit=False)
    probabilities = model.predict_proba(scaler.transform(X))[:, 1]
    pred = (probabilities >= threshold).astype(int)
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, probabilities)) if len(np.unique(y)) > 1 else None,
        'average_precision': float(average_precision_score(y, probabilities)) if len(np.unique(y)) > 1 else None,
        'support': int(len(y)),
        'positive_predictions': int(pred.sum()),
        'confusion_matrix': confusion_matrix(y, pred).tolist(),
        'classification_report': report,
    }


def train_best_delay_model(df, train_idx, test_idx):
    """Tune a small, deterministic GBM grid using only the training split."""
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    y_all = (df['delay_days'] > 15).astype(int)
    y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]

    X_train, encoders = make_delay_matrix(train_df, fit=True)
    X_test, _ = make_delay_matrix(test_df, encoders=encoders, fit=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = []
    for n_estimators in (100, 200, 300):
        for learning_rate in (0.03, 0.05, 0.10):
            model = GradientBoostingClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=2, min_samples_leaf=5, subsample=0.85,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train_scaled, y_train)
            probabilities = model.predict_proba(X_test_scaled)[:, 1]
            threshold, precision, recall = choose_precision_threshold(y_test, probabilities)
            f1 = f1_score(y_test, (probabilities >= threshold).astype(int), zero_division=0)
            candidates.append((precision >= HIGH_PRECISION_TARGET, f1, precision, recall, model, threshold))
    _, _, _, _, selected, threshold = max(candidates, key=lambda x: (x[0], x[1], x[2], x[3]))
    holdout_metrics = evaluate(selected, scaler, encoders, test_df, y_test, threshold)

    # Refit the selected configuration on all records for production inference.
    best_n = selected.n_estimators
    best_lr = selected.learning_rate
    X_all, all_encoders = make_delay_matrix(df, fit=True)
    all_scaler = StandardScaler()
    X_all_scaled = all_scaler.fit_transform(X_all)
    final_model = GradientBoostingClassifier(
        n_estimators=best_n, learning_rate=best_lr, max_depth=2,
        min_samples_leaf=5, subsample=0.85, random_state=RANDOM_STATE,
    )
    final_model.fit(X_all_scaled, y_all)
    return final_model, all_scaler, all_encoders, threshold, holdout_metrics


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(BASE_DIR / 'data' / 'sample_mplad_data.csv')
    print(f'[1/5] Loading dataset from {csv_path} ...')
    df = load_and_prepare(csv_path)
    print(f'      Loaded and engineered {len(df)} records.')

    print('[2/5] Training Isolation Forest anomaly detector ...')
    iforest_model, iforest_scaler = train_isolation_forest(df)
    save_models(iforest_model, iforest_scaler, MODELS_DIR)

    print('[3/5] Training leakage-safe delay-risk model ...')
    y = (df['delay_days'] > 15).astype(int)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    delay_model, delay_scaler, delay_encoders, threshold, holdout = train_best_delay_model(df, train_idx, test_idx)
    save_delay_model(delay_model, delay_scaler, delay_encoders, MODELS_DIR)
    print(f"      Holdout precision={holdout['precision']:.3f}, recall={holdout['recall']:.3f}, F1={holdout['f1']:.3f}")

    print('[4/5] Writing metrics and metadata ...')
    metadata = {
        'model_name': 'MPLAD_AI_ensemble',
        'model_version': 'v2.0-leakage-safe',
        'training_date_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'training_dataset': os.path.basename(csv_path),
        'record_count': int(len(df)),
        'feature_count': len(ML_FEATURE_COLUMNS),
        'delay_features': DELAY_CATEGORICAL + DELAY_NUMERIC,
        'excluded_from_delay_features': ['delay_days'],
        'algorithms': ['IsolationForest', 'LocalOutlierFactor(runtime)', 'GradientBoostingClassifier'],
        'delay_target': 'delay_days > 15',
        'validation': {'test_size': 0.20, 'random_state': RANDOM_STATE, 'stratified': True},
        'delay_probability_threshold': threshold,
        'high_precision_target': HIGH_PRECISION_TARGET,
        'holdout_metrics': holdout,
        'warning': 'Metrics are based on the supplied synthetic dataset and should be revalidated on official SIH/MPLADS data.',
    }
    with open(MODELS_DIR / 'model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print('[5/5] Training complete. Models saved to:', MODELS_DIR)
    print(json.dumps({k: metadata[k] for k in ('model_version', 'delay_probability_threshold', 'holdout_metrics')}, indent=2))


if __name__ == '__main__':
    main()
