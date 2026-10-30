"""
Business logic for the projects app:
  - CSV import / validation pipeline (Stage 27, 28)
  - Converting MongoDB project documents -> pandas DataFrame and back
  - Running the ML pipeline and persisting anomalies + ml_predictions
"""
import uuid
import datetime
import pandas as pd
import numpy as np

from projects.models import Project, Anomaly, MLPrediction
from ml_engine.feature_engineering import clean_dataframe, engineer_features, RAW_COLUMNS
from ml_engine.predict import run_full_analysis


# ---------------------------- CSV Import (Stage 27) ----------------------------

REQUIRED_COLUMNS = set(RAW_COLUMNS)


def validate_csv_columns(df: pd.DataFrame):
    missing = REQUIRED_COLUMNS - set(df.columns)
    return list(missing)


def import_csv_to_mongo(df: pd.DataFrame):
    """Validate, clean, engineer features, then upsert into MongoDB
    `projects` collection. Returns a summary dict (Stage 27/28)."""
    missing_cols = validate_csv_columns(df)
    if missing_cols:
        return {'success': False, 'error': f'Missing required columns: {missing_cols}'}

    original_count = len(df)
    df = clean_dataframe(df)
    df = engineer_features(df)

    inserted, updated, errors = 0, 0, []

    for _, row in df.iterrows():
        try:
            defaults = dict(
                project_name=str(row.get('project_id', '')) + ' Project' if pd.isna(row.get('project_name', np.nan)) else row.get('project_name'),
                mp_name=row.get('mp_name', 'Unknown'),
                constituency=row.get('constituency', 'Unknown'),
                district=row.get('district', 'Unknown'),
                state=row.get('state', 'Unknown'),
                implementing_agency=row.get('implementing_agency', 'Unknown'),
                contractor=row.get('contractor', 'Unknown'),
                project_category=row.get('project_category', 'Public Infrastructure'),
                sanctioned_amount=float(row.get('sanctioned_amount', 0) or 0),
                released_amount=float(row.get('released_amount', 0) or 0),
                expenditure_amount=float(row.get('expenditure_amount', 0) or 0),
                estimated_cost=float(row.get('estimated_cost', 0) or 0),
                actual_cost=float(row.get('actual_cost', 0) or 0),
                start_date=row['start_date'].date() if pd.notna(row.get('start_date')) else None,
                expected_completion_date=row['expected_completion_date'].date() if pd.notna(row.get('expected_completion_date')) else None,
                actual_completion_date=row['actual_completion_date'].date() if pd.notna(row.get('actual_completion_date')) else None,
                project_status=row.get('project_status', 'In Progress'),
                completion_percentage=float(row.get('completion_percentage', 0) or 0),
                latitude=float(row['latitude']) if pd.notna(row.get('latitude')) else None,
                longitude=float(row['longitude']) if pd.notna(row.get('longitude')) else None,
                beneficiaries=int(row.get('beneficiaries', 0) or 0),
                expenditure_ratio=float(row.get('expenditure_ratio', 0) or 0),
                cost_overrun=float(row.get('cost_overrun', 0) or 0),
                delay_days=int(row.get('delay_days', 0) or 0),
                utilization_ratio=float(row.get('utilization_ratio', 0) or 0),
                completion_efficiency=float(row.get('completion_efficiency', 0) or 0),
                cost_per_beneficiary=float(row.get('cost_per_beneficiary', 0) or 0),
                project_duration=int(row.get('project_duration', 0) or 0),
                updated_at=datetime.datetime.utcnow(),
            )
            existing = Project.objects(project_id=row['project_id']).first()
            if existing:
                for k, v in defaults.items():
                    setattr(existing, k, v)
                existing.save()
                updated += 1
            else:
                Project(project_id=row['project_id'], mp_id=row.get('mp_id', row.get('mp_name', 'MP')), **defaults).save()
                inserted += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{row.get('project_id', '?')}: {e}")

    return {
        'success': True,
        'total_rows': original_count,
        'inserted': inserted,
        'updated': updated,
        'errors': errors[:50],
        'error_count': len(errors),
    }


# ---------------------------- Mongo <-> DataFrame ----------------------------

def projects_queryset_to_dataframe(queryset=None):
    qs = queryset if queryset is not None else Project.objects()
    records = [p.to_dict() for p in qs]
    if not records:
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.DataFrame(records)
    for col in ['start_date', 'expected_completion_date', 'actual_completion_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


# ---------------------------- ML analysis + persistence (Stage 21) ----------------------------

def run_ml_analysis_and_save(project_ids=None):
    """Run the full ML pipeline over all (or selected) projects and persist
    results into `anomalies` and `ml_predictions` collections."""
    queryset = Project.objects(project_id__in=project_ids) if project_ids else Project.objects()
    df = projects_queryset_to_dataframe(queryset)
    if df.empty:
        return {'success': False, 'error': 'No projects available to analyze'}

    results_df, duplicate_pairs = run_full_analysis(df)

    saved = 0
    for _, row in results_df.iterrows():
        pid = row['project_id']

        MLPrediction.objects(project_id=pid).update_one(
            upsert=True,
            set__anomaly_probability=round(float(row['anomaly_score']), 2),
            set__fraud_risk_score=round(float(row['fraud_risk_score']), 2),
            set__inefficiency_score=round(float(row['inefficiency_score']), 2),
            set__delay_risk_score=round(float(row['delay_risk_score']), 2),
            set__overall_risk_score=round(float(row['overall_risk_score']), 2),
            set__risk_level=row['risk_level'],
            set__reasons=row['reasons'],
            set__contributing_factors=row['contributing_factors'],
            set__prediction_date=datetime.datetime.utcnow(),
            set__model_version='v1.0',
        )

        # Only create an Anomaly record when the project is genuinely flagged
        if row['anomaly_score'] >= 50 or row['overall_risk_score'] >= 50:
            severity = 'Critical' if row['overall_risk_score'] >= 75 else (
                'High' if row['overall_risk_score'] >= 50 else (
                    'Medium' if row['overall_risk_score'] >= 25 else 'Low'))
            existing = Anomaly.objects(project_id=pid, review_status='Pending').first()
            anomaly_id = existing.anomaly_id if existing else f'ANOM-{uuid.uuid4().hex[:10].upper()}'
            Anomaly.objects(anomaly_id=anomaly_id).update_one(
                upsert=True,
                set__project_id=pid,
                set__anomaly_type='ML Composite Risk',
                set__anomaly_score=round(float(row['anomaly_score']), 2),
                set__severity=severity,
                set__detected_features=row['contributing_factors'],
                set__explanation='; '.join(row['reasons']),
                set__detected_at=datetime.datetime.utcnow(),
            )
        saved += 1

    return {
        'success': True,
        'analyzed': saved,
        'duplicate_pairs_found': len(duplicate_pairs),
        'duplicate_pairs': duplicate_pairs[:50],
    }


def analyze_single_project(project_id):
    result = run_ml_analysis_and_save(project_ids=[project_id])
    if not result.get('success'):
        return None
    return MLPrediction.objects(project_id=project_id).first()
