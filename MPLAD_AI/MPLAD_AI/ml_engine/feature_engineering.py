"""
Feature engineering pipeline (Stage 6).

Takes a raw pandas DataFrame of MPLAD project records and:
 - cleans / validates values
 - converts dates
 - engineers the derived features used throughout the ML modules
"""
import numpy as np
import pandas as pd

RAW_COLUMNS = [
    'project_id', 'mp_name', 'constituency', 'district', 'state',
    'project_category', 'sanctioned_amount', 'released_amount',
    'expenditure_amount', 'estimated_cost', 'actual_cost', 'start_date',
    'expected_completion_date', 'actual_completion_date',
    'completion_percentage', 'contractor', 'implementing_agency',
    'beneficiaries', 'latitude', 'longitude', 'project_status',
]

DATE_COLUMNS = ['start_date', 'expected_completion_date', 'actual_completion_date']
MONEY_COLUMNS = ['sanctioned_amount', 'released_amount', 'expenditure_amount',
                  'estimated_cost', 'actual_cost']


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, invalid values (Stage 6 & 28)."""
    df = df.copy()

    # Remove duplicate project_id rows, keep the first occurrence
    df = df.drop_duplicates(subset=['project_id'], keep='first')

    # Convert dates
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Numeric coercion + fix negative monetary values (invalid -> NaN -> filled)
    for col in MONEY_COLUMNS + ['completion_percentage', 'beneficiaries', 'latitude', 'longitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in MONEY_COLUMNS:
        df.loc[df[col] < 0, col] = np.nan

    df.loc[(df['completion_percentage'] < 0) | (df['completion_percentage'] > 100), 'completion_percentage'] = np.nan
    df.loc[df['beneficiaries'] < 0, 'beneficiaries'] = np.nan
    df.loc[(df['latitude'] < -90) | (df['latitude'] > 90), 'latitude'] = np.nan
    df.loc[(df['longitude'] < -180) | (df['longitude'] > 180), 'longitude'] = np.nan

    # Fill missing numeric values with the column median (robust to outliers)
    for col in MONEY_COLUMNS + ['completion_percentage', 'beneficiaries']:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)

    for col in ['latitude', 'longitude']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # Fill missing categorical values
    for col in ['project_category', 'project_status', 'implementing_agency', 'contractor', 'state', 'district']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the derived features described in Stage 6 of the spec."""
    df = df.copy()
    today = pd.Timestamp.today()

    # project_duration in days (planned duration if not completed yet)
    end_for_duration = df['actual_completion_date'].fillna(df['expected_completion_date'])
    df['project_duration'] = (end_for_duration - df['start_date']).dt.days
    df['project_duration'] = df['project_duration'].replace(0, 1).fillna(1)
    df.loc[df['project_duration'] <= 0, 'project_duration'] = 1

    # expenditure_ratio = expenditure_amount / sanctioned_amount
    df['expenditure_ratio'] = df['expenditure_amount'] / df['sanctioned_amount'].replace(0, np.nan)
    df['expenditure_ratio'] = df['expenditure_ratio'].fillna(0)

    # cost_overrun = (actual_cost - estimated_cost) / estimated_cost
    df['cost_overrun'] = (df['actual_cost'] - df['estimated_cost']) / df['estimated_cost'].replace(0, np.nan)
    df['cost_overrun'] = df['cost_overrun'].fillna(0)

    # delay_days = actual_completion_date - expected_completion_date
    delay = (df['actual_completion_date'] - df['expected_completion_date']).dt.days
    # For ongoing projects, compare "today" against the expected date if already overdue
    ongoing_overdue = (df['actual_completion_date'].isna()) & (today > df['expected_completion_date'])
    delay = delay.fillna(0)
    delay[ongoing_overdue] = (today - df.loc[ongoing_overdue, 'expected_completion_date']).dt.days
    df['delay_days'] = delay.fillna(0).astype(int)

    # utilization_ratio = expenditure_amount / released_amount
    df['utilization_ratio'] = df['expenditure_amount'] / df['released_amount'].replace(0, np.nan)
    df['utilization_ratio'] = df['utilization_ratio'].fillna(0)

    # completion_efficiency = completion_percentage / project_duration
    df['completion_efficiency'] = df['completion_percentage'] / df['project_duration']

    # cost_per_beneficiary = actual_cost / beneficiaries
    df['cost_per_beneficiary'] = df['actual_cost'] / df['beneficiaries'].replace(0, np.nan)
    df['cost_per_beneficiary'] = df['cost_per_beneficiary'].fillna(df['actual_cost'].median())

    return df


ML_FEATURE_COLUMNS = [
    'sanctioned_amount', 'expenditure_amount', 'utilization_ratio',
    'cost_overrun', 'delay_days', 'completion_percentage', 'beneficiaries',
    'cost_per_beneficiary', 'project_duration',
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return just the numeric feature matrix used to feed ML models,
    with standardization-friendly clean numeric values (no NaN/inf)."""
    X = df[ML_FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Full pipeline: read CSV -> clean -> engineer features."""
    df = pd.read_csv(csv_path)
    df = clean_dataframe(df)
    df = engineer_features(df)
    return df
