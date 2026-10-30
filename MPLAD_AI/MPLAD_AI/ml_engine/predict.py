"""
ml_engine/predict.py

Orchestrates all ML modules to produce, per project:
  anomaly_score, fraud_risk_score, inefficiency_score, delay_risk_score,
  overall_risk_score, risk_level, and human-readable contributing factors
  (Stage 12 — Overall Risk Score, Stage 24 — Explainability).

This is used by:
  - projects/services.py (POST /api/ml/analyze/, CSV import pipeline)
  - ml_engine/train_models.py indirectly (via the same feature pipeline)
"""
from pathlib import Path
import numpy as np
import pandas as pd

from django.conf import settings
from ml_engine.feature_engineering import ML_FEATURE_COLUMNS
from ml_engine.anomaly_detection import (
    load_models as load_iforest_models, score_isolation_forest, run_lof, explain_anomaly,
)
from ml_engine.fraud_detection import compute_fraud_risk_score, top_fraud_reasons
from ml_engine.inefficiency_detection import (
    compute_efficiency_score, inefficiency_flags,
    load_delay_model, score_delay_risk,
    detect_duplicate_projects, duplicate_score_per_project,
)


def _risk_level(score):
    if score >= 75:
        return 'Critical'
    if score >= 50:
        return 'High'
    if score >= 25:
        return 'Moderate'
    return 'Low'


def run_full_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete ML pipeline over a cleaned+feature-engineered
    DataFrame (see feature_engineering.load_and_prepare) and return a new
    DataFrame with one row per project_id containing every score plus a
    JSON-serializable 'reasons' list and 'contributing_factors' dict.
    """
    models_dir = Path(settings.ML_MODELS_DIR)
    result = pd.DataFrame(index=df.index)
    result['project_id'] = df['project_id']

    # ---- Anomaly detection: Isolation Forest + LOF ----
    try:
        iforest_model, iforest_scaler = load_iforest_models(models_dir)
        iforest_score, iforest_labels = score_isolation_forest(iforest_model, iforest_scaler, df)
    except FileNotFoundError:
        # Models not yet trained -> fall back to fitting on the fly (still valid, unsupervised)
        from ml_engine.anomaly_detection import train_isolation_forest
        iforest_model, iforest_scaler = train_isolation_forest(df)
        iforest_score, iforest_labels = score_isolation_forest(iforest_model, iforest_scaler, df)

    lof_score, lof_labels = run_lof(df)
    # Blend: average of both, weighted slightly toward agreement
    blended_anomaly = (0.55 * iforest_score) + (0.45 * lof_score)
    agreement_bonus = np.where((iforest_labels == -1) & (lof_labels == -1), 10, 0)
    anomaly_score = np.clip(blended_anomaly + agreement_bonus, 0, 100)
    result['anomaly_score'] = anomaly_score

    # ---- Fraud risk (Stage 7-C) ----
    fraud_score, fraud_label, fraud_indicators = compute_fraud_risk_score(df)
    result['fraud_risk_score'] = fraud_score
    result['fraud_risk_label'] = fraud_label

    # ---- Inefficiency (Stage 8) ----
    efficiency_score = compute_efficiency_score(df)
    result['inefficiency_score'] = (100 - efficiency_score).clip(0, 100)
    flags = inefficiency_flags(df)

    # ---- Delay risk (Stage 9) ----
    try:
        delay_model, delay_scaler, delay_encoders = load_delay_model(models_dir)
        delay_risk_score = score_delay_risk(delay_model, delay_scaler, delay_encoders, df)
    except FileNotFoundError:
        from ml_engine.inefficiency_detection import train_delay_model
        delay_model, delay_scaler, delay_encoders = train_delay_model(df)
        delay_risk_score = score_delay_risk(delay_model, delay_scaler, delay_encoders, df)
    result['delay_risk_score'] = delay_risk_score

    # ---- Duplicate detection (Stage 10) ----
    duplicate_pairs = detect_duplicate_projects(df)
    dup_score = duplicate_score_per_project(df, duplicate_pairs) * 100
    result['duplicate_score'] = dup_score.values

    # ---- Overall Risk Score (Stage 12) ----
    weights = settings.RISK_WEIGHTS
    financial_risk = (0.5 * result['fraud_risk_score'] + 0.5 * (df['cost_overrun'].clip(lower=0) * 100).clip(0, 100))
    result['overall_risk_score'] = np.clip(
        weights['anomaly'] * result['anomaly_score'] +
        weights['financial'] * financial_risk +
        weights['delay'] * result['delay_risk_score'] +
        weights['inefficiency'] * result['inefficiency_score'] +
        weights['duplicate'] * result['duplicate_score'],
        0, 100,
    )
    result['risk_level'] = result['overall_risk_score'].apply(_risk_level)

    # ---- Explainability: contributing factors + reasons (Stage 24) ----
    feature_medians = df[ML_FEATURE_COLUMNS].median().to_dict()
    reasons_list = []
    factors_list = []
    for idx in df.index:
        row = df.loc[idx]
        reasons = []

        anomaly_reasons = explain_anomaly(row, feature_medians)
        if result.loc[idx, 'anomaly_score'] >= 50:
            reasons.extend(anomaly_reasons[:2])

        reasons.extend(top_fraud_reasons(fraud_indicators.loc[idx], top_n=2))

        if flags.loc[idx, 'delayed_project']:
            reasons.append('Project delay detected beyond the expected completion date')
        if flags.loc[idx, 'under_utilized_funds']:
            reasons.append('Funds released appear significantly under-utilized')
        if flags.loc[idx, 'cost_overrun_flag']:
            reasons.append('Cost overrun exceeds the acceptable threshold')
        if not reasons:
            reasons = ['No significant risk indicators detected']

        # de-duplicate while preserving order
        seen = set()
        deduped = [r for r in reasons if not (r in seen or seen.add(r))]
        reasons_list.append(deduped[:6])

        factors_list.append({
            'Anomaly Pattern': round(weights['anomaly'] * result.loc[idx, 'anomaly_score'], 1),
            'Financial Risk': round(weights['financial'] * financial_risk.loc[idx], 1),
            'Delay Risk': round(weights['delay'] * result.loc[idx, 'delay_risk_score'], 1),
            'Inefficiency': round(weights['inefficiency'] * result.loc[idx, 'inefficiency_score'], 1),
            'Duplicate/Suspicious Pattern': round(weights['duplicate'] * result.loc[idx, 'duplicate_score'], 1),
        })

    result['reasons'] = reasons_list
    result['contributing_factors'] = factors_list

    return result, duplicate_pairs
