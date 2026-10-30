"""
Unsupervised anomaly detection (Stage 7-A, 7-B):
 - Isolation Forest
 - Local Outlier Factor (LOF)

Both models score every project; we compare and blend their outputs so a
project only needs to look unusual to ONE robust method to be surfaced,
while agreement between both raises confidence.
"""
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from ml_engine.feature_engineering import ML_FEATURE_COLUMNS, build_feature_matrix


def train_isolation_forest(df, contamination=0.07, random_state=42):
    X = build_feature_matrix(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_scaled)
    return model, scaler


def score_isolation_forest(model, scaler, df):
    X = build_feature_matrix(df)
    X_scaled = scaler.transform(X)
    raw_scores = model.decision_function(X_scaled)  # higher = more normal
    labels = model.predict(X_scaled)  # -1 anomaly, 1 normal

    # Convert to an intuitive 0-100 anomaly score (higher = more anomalous)
    normalized = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-9)
    anomaly_score = np.clip(normalized * 100, 0, 100)
    return anomaly_score, labels


def run_lof(df, n_neighbors=20, contamination=0.07):
    """LOF is transductive (no separate predict on new data in classic
    scikit-learn usage), so it is fit and scored on the same dataset."""
    X = build_feature_matrix(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, max(2, len(df) - 1)),
                              contamination=contamination)
    labels = lof.fit_predict(X_scaled)  # -1 anomaly, 1 normal
    neg_outlier_factor = lof.negative_outlier_factor_  # more negative = more anomalous

    normalized = (neg_outlier_factor.max() - neg_outlier_factor) / (
        neg_outlier_factor.max() - neg_outlier_factor.min() + 1e-9)
    lof_score = np.clip(normalized * 100, 0, 100)
    return lof_score, labels


def explain_anomaly(row, feature_medians):
    """Return the top contributing feature deviations, in plain language,
    for explainability (Stage 24)."""
    reasons = []
    checks = [
        ('utilization_ratio', 'Budget utilization is far from the typical range', 1.5),
        ('cost_overrun', 'Cost overrun is unusually high compared to similar projects', 0.25),
        ('delay_days', 'Project delay is significantly longer than average', 60),
        ('cost_per_beneficiary', 'Cost per beneficiary is significantly above the peer-group median', None),
        ('completion_efficiency', 'Completion progress is unusually slow relative to elapsed time', None),
    ]
    for feature, message, hard_threshold in checks:
        value = row.get(feature)
        median = feature_medians.get(feature, 0)
        if value is None:
            continue
        if hard_threshold is not None and abs(value) >= hard_threshold:
            reasons.append(message)
        elif median and abs(value - median) > 1.5 * abs(median if median != 0 else 1):
            reasons.append(message)
    return reasons or ['Statistical pattern deviates from similar projects in the dataset']


def save_models(model, scaler, models_dir):
    joblib.dump(model, models_dir / 'isolation_forest.joblib')
    joblib.dump(scaler, models_dir / 'iforest_scaler.joblib')


def load_models(models_dir):
    model = joblib.load(models_dir / 'isolation_forest.joblib')
    scaler = joblib.load(models_dir / 'iforest_scaler.joblib')
    return model, scaler
