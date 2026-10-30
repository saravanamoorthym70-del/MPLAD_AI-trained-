"""
Inefficiency scoring (Stage 8), delay-risk model (Stage 9), and duplicate
project detection (Stage 10).
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------- Stage 8: Inefficiency ----------------------------

def compute_efficiency_score(df: pd.DataFrame) -> pd.Series:
    """efficiency_score = 0-100, HIGHER is more efficient.
    Combines budget utilization sanity, completion %, duration, cost
    overrun, and beneficiary coverage into one interpretable score."""

    # Budget utilization: best around 0.85-1.0 (spent what was needed, not
    # wildly under or over)
    util = df['utilization_ratio'].clip(0, 3)
    util_score = 100 - (np.abs(util - 0.9) * 60).clip(0, 100)

    completion_score = df['completion_percentage'].clip(0, 100)

    # Duration: penalize excessively long durations vs category median
    duration_by_cat = df.groupby('project_category')['project_duration'].transform('median')
    duration_ratio = df['project_duration'] / duration_by_cat.replace(0, np.nan).fillna(df['project_duration'].median())
    duration_score = 100 - ((duration_ratio - 1).clip(lower=0) * 50).clip(0, 100)

    overrun_score = 100 - (df['cost_overrun'].clip(lower=0) * 100).clip(0, 100)

    beneficiaries_by_cat = df.groupby('project_category')['beneficiaries'].transform('median')
    coverage_ratio = df['beneficiaries'] / beneficiaries_by_cat.replace(0, np.nan).fillna(df['beneficiaries'].median() or 1)
    coverage_score = (coverage_ratio.clip(0, 2) * 50).clip(0, 100)

    efficiency_score = (
        0.25 * util_score +
        0.25 * completion_score +
        0.20 * duration_score +
        0.20 * overrun_score +
        0.10 * coverage_score
    ).clip(0, 100)

    return efficiency_score


def inefficiency_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)
    flags['under_utilized_funds'] = df['utilization_ratio'] < 0.5
    flags['delayed_project'] = df['delay_days'] > 30
    flags['high_cost_project'] = df['cost_overrun'] > 0.25
    flags['low_beneficiary_coverage'] = df['beneficiaries'] < df['beneficiaries'].quantile(0.15)
    flags['cost_overrun_flag'] = df['cost_overrun'] > 0.15
    return flags


# ---------------------------- Stage 9: Delay prediction ----------------------------

DELAY_CATEGORICAL = ['project_category', 'implementing_agency', 'contractor', 'state']
DELAY_NUMERIC = ['sanctioned_amount', 'expenditure_ratio', 'completion_percentage', 'project_duration']


def build_delay_training_set(df: pd.DataFrame):
    """Label: was the project actually delayed (delay_days > 15)?"""
    y = (df['delay_days'] > 15).astype(int)

    X = pd.DataFrame(index=df.index)
    encoders = {}
    for col in DELAY_CATEGORICAL:
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    for col in DELAY_NUMERIC:
        X[col] = df[col]

    return X, y, encoders


def train_delay_model(df: pd.DataFrame):
    X, y, encoders = build_delay_training_set(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, encoders


def score_delay_risk(model, scaler, encoders, df: pd.DataFrame):
    X = pd.DataFrame(index=df.index)
    for col in DELAY_CATEGORICAL:
        le = encoders[col]
        # Handle unseen categories gracefully by mapping to a known class
        known = set(le.classes_)
        safe_values = df[col].astype(str).map(lambda v: v if v in known else le.classes_[0])
        X[col] = le.transform(safe_values)
    for col in DELAY_NUMERIC:
        X[col] = df[col]

    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]  # probability of class "delayed"
    delay_risk_score = np.clip(probabilities * 100, 0, 100)
    return delay_risk_score


def save_delay_model(model, scaler, encoders, models_dir):
    joblib.dump(model, models_dir / 'delay_model.joblib')
    joblib.dump(scaler, models_dir / 'delay_scaler.joblib')
    joblib.dump(encoders, models_dir / 'delay_encoders.joblib')


def load_delay_model(models_dir):
    model = joblib.load(models_dir / 'delay_model.joblib')
    scaler = joblib.load(models_dir / 'delay_scaler.joblib')
    encoders = joblib.load(models_dir / 'delay_encoders.joblib')
    return model, scaler, encoders


# ---------------------------- Stage 10: Duplicate detection ----------------------------

def detect_duplicate_projects(df: pd.DataFrame, similarity_threshold=0.80, top_k=200):
    """TF-IDF + cosine similarity over a composite text field (name +
    location + contractor + rounded amount), flags likely duplicate pairs."""
    composite = (
        df['project_name'].astype(str) + ' ' +
        df['district'].astype(str) + ' ' + df['state'].astype(str) + ' ' +
        df['contractor'].astype(str) + ' ' +
        (df['sanctioned_amount'] // 10000).astype(str)
    )

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(composite)

    # For large datasets, cap comparisons for performance
    n = tfidf_matrix.shape[0]
    sim_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)

    pairs = []
    seen = set()
    idxs = np.argwhere(sim_matrix >= similarity_threshold)
    for i, j in idxs:
        if i >= j:
            continue
        key = (df.iloc[i]['project_id'], df.iloc[j]['project_id'])
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            'project_a': df.iloc[i]['project_id'],
            'project_a_name': df.iloc[i]['project_name'],
            'project_b': df.iloc[j]['project_id'],
            'project_b_name': df.iloc[j]['project_name'],
            'similarity': round(float(sim_matrix[i, j]) * 100, 1),
            'status': 'Requires Review',
        })
        if len(pairs) >= top_k:
            break

    pairs.sort(key=lambda p: p['similarity'], reverse=True)
    return pairs


def duplicate_score_per_project(df: pd.DataFrame, pairs):
    """Turn pairwise duplicate results into a per-project 0-1 severity used
    in the overall risk score (10% weight)."""
    max_sim = {}
    for p in pairs:
        max_sim[p['project_a']] = max(max_sim.get(p['project_a'], 0), p['similarity'])
        max_sim[p['project_b']] = max(max_sim.get(p['project_b'], 0), p['similarity'])
    return df['project_id'].map(lambda pid: max_sim.get(pid, 0) / 100.0).fillna(0)
