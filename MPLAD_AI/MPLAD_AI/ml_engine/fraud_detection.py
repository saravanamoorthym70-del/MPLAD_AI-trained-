"""
Fraud-risk indicator scoring (Stage 7-C).

IMPORTANT: this module NEVER claims to prove fraud. It only produces a
0-100 "fraud_risk_score" plus the specific indicators that fired, so a
human reviewer (Auditor/Admin) can investigate further. Output labels are
always "Low Risk" / "Medium Risk" / "High Risk".
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _score_indicator(series, upper_quantile=0.95):
    """Score 0-1 for how far above the "normal" band (95th percentile) a
    value sits. Returns a Series of 0-1 severity per row."""
    threshold = series.quantile(upper_quantile)
    spread = series.max() - threshold
    if spread <= 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    severity = ((series - threshold) / spread).clip(lower=0, upper=1)
    return severity


def compute_fraud_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute individual 0-1 indicator severities for each of the fraud
    signals listed in the spec, then combine into fraud_risk_score."""
    out = pd.DataFrame(index=df.index)

    # 1. Very high expenditure (relative to peers)
    out['ind_high_expenditure'] = _score_indicator(df['expenditure_amount'])

    # 2. Unusual cost overrun
    out['ind_cost_overrun'] = _score_indicator(df['cost_overrun'].abs())

    # 3. Multiple projects with similar descriptions -> handled via duplicate_detection.py,
    #    surfaced here as a placeholder column filled in by predict.py when available.
    out['ind_similar_description'] = 0.0

    # 4. Repeated contractor association (same contractor with many projects)
    contractor_counts = df['contractor'].value_counts()
    max_count = max(contractor_counts.max(), 1)
    out['ind_repeated_contractor'] = df['contractor'].map(
        lambda c: contractor_counts.get(c, 0) / max_count
    ).clip(0, 1) * (contractor_counts.max() > 3)

    # 5. Unusual payment timing -> approximated with abnormal utilization timing
    out['ind_payment_timing'] = _score_indicator((df['utilization_ratio'] - 1).abs())

    # 6. High-value transactions relative to sanctioned amount
    out['ind_high_value_txn'] = _score_indicator(df['actual_cost'] / df['sanctioned_amount'].replace(0, np.nan).fillna(1))

    # 7. Very low completion with high expenditure
    low_completion_high_spend = ((df['completion_percentage'] < 40) & (df['expenditure_ratio'] > 0.7)).astype(float)
    out['ind_low_completion_high_spend'] = low_completion_high_spend

    # 8. Duplicate project locations (same lat/lon rounded to 3 decimals used by many projects)
    loc_key = df['latitude'].round(3).astype(str) + '_' + df['longitude'].round(3).astype(str)
    loc_counts = loc_key.value_counts()
    out['ind_duplicate_location'] = loc_key.map(lambda k: min(loc_counts.get(k, 0) / 5.0, 1.0))

    # 9. Sudden expenditure spikes -> approximated via expenditure_ratio far above 1
    out['ind_expenditure_spike'] = _score_indicator((df['expenditure_ratio'] - 1).clip(lower=0))

    # 10. Abnormal cost per beneficiary
    out['ind_abnormal_cost_per_beneficiary'] = _score_indicator(df['cost_per_beneficiary'])

    return out


FRAUD_INDICATOR_WEIGHTS = {
    'ind_high_expenditure': 0.12,
    'ind_cost_overrun': 0.14,
    'ind_similar_description': 0.08,
    'ind_repeated_contractor': 0.10,
    'ind_payment_timing': 0.08,
    'ind_high_value_txn': 0.10,
    'ind_low_completion_high_spend': 0.16,
    'ind_duplicate_location': 0.08,
    'ind_expenditure_spike': 0.08,
    'ind_abnormal_cost_per_beneficiary': 0.06,
}

FRAUD_INDICATOR_LABELS = {
    'ind_high_expenditure': 'Very high expenditure relative to peer projects',
    'ind_cost_overrun': 'Unusual cost overrun',
    'ind_similar_description': 'Multiple projects with highly similar descriptions',
    'ind_repeated_contractor': 'Repeated contractor association across many projects',
    'ind_payment_timing': 'Unusual payment/utilization timing pattern',
    'ind_high_value_txn': 'High-value transactions relative to sanctioned amount',
    'ind_low_completion_high_spend': 'Very low completion with high expenditure',
    'ind_duplicate_location': 'Duplicate or clustered project locations',
    'ind_expenditure_spike': 'Sudden expenditure spike',
    'ind_abnormal_cost_per_beneficiary': 'Abnormal cost per beneficiary',
}


def compute_fraud_risk_score(df: pd.DataFrame):
    indicators = compute_fraud_indicators(df)
    weighted = sum(indicators[col] * w for col, w in FRAUD_INDICATOR_WEIGHTS.items())
    fraud_risk_score = (weighted * 100).clip(0, 100)

    def label(score):
        if score >= 60:
            return 'High Risk'
        if score >= 30:
            return 'Medium Risk'
        return 'Low Risk'

    risk_labels = fraud_risk_score.apply(label)
    return fraud_risk_score, risk_labels, indicators


def top_fraud_reasons(indicators_row, top_n=4):
    """Return the strongest-firing indicators for one project, in plain
    English, for the explainability UI (Stage 18/24)."""
    scored = [(FRAUD_INDICATOR_LABELS[c], indicators_row[c]) for c in FRAUD_INDICATOR_WEIGHTS if indicators_row.get(c, 0) > 0.2]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [label for label, _ in scored[:top_n]]
