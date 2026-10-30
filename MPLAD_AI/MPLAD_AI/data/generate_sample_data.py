"""
data/generate_sample_data.py  (Stage 5)

Generates a realistic synthetic MPLADS dataset of 1000+ records with a mix
of normal projects and intentionally unusual/anomalous ones, so the ML
modules in ml_engine/ have genuine patterns to learn from and detect.

Run with:
    python data/generate_sample_data.py
Writes: data/sample_mplad_data.csv
"""
import random
import string
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker('en_IN')
random.seed(42)
np.random.seed(42)

STATES_DISTRICTS = {
    'Tamil Nadu': ['Coimbatore', 'Chennai', 'Madurai', 'Salem', 'Tiruchirappalli'],
    'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Varanasi', 'Agra', 'Allahabad'],
    'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Nashik', 'Aurangabad'],
    'Karnataka': ['Bengaluru', 'Mysuru', 'Hubballi', 'Mangaluru', 'Belagavi'],
    'West Bengal': ['Kolkata', 'Howrah', 'Durgapur', 'Siliguri', 'Asansol'],
    'Rajasthan': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota', 'Ajmer'],
    'Bihar': ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur', 'Darbhanga'],
    'Kerala': ['Thiruvananthapuram', 'Kochi', 'Kozhikode', 'Thrissur', 'Kollam'],
    'Punjab': ['Amritsar', 'Ludhiana', 'Jalandhar', 'Patiala', 'Bathinda'],
    'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Bhavnagar'],
}

STATE_COORDS = {
    'Tamil Nadu': (11.1, 78.6), 'Uttar Pradesh': (26.8, 80.9), 'Maharashtra': (19.7, 75.7),
    'Karnataka': (15.3, 75.7), 'West Bengal': (22.9, 87.8), 'Rajasthan': (27.0, 74.2),
    'Bihar': (25.0, 85.3), 'Kerala': (10.8, 76.2), 'Punjab': (31.1, 75.3), 'Gujarat': (22.2, 71.1),
}

CATEGORIES = ['Roads', 'Bridges', 'Schools', 'Hospitals', 'Drinking Water',
              'Sanitation', 'Community Halls', 'Electricity',
              'Public Infrastructure', 'Sports Facilities']

CATEGORY_COST_RANGE = {
    'Roads': (800000, 6000000), 'Bridges': (2000000, 15000000), 'Schools': (1000000, 8000000),
    'Hospitals': (2000000, 20000000), 'Drinking Water': (500000, 4000000),
    'Sanitation': (300000, 2500000), 'Community Halls': (500000, 3500000),
    'Electricity': (400000, 3000000), 'Public Infrastructure': (700000, 5000000),
    'Sports Facilities': (600000, 4500000),
}

AGENCIES = ['District Rural Development Agency', 'PWD Division', 'Municipal Corporation',
            'Zilla Parishad', 'State Infrastructure Board', 'Panchayati Raj Dept']

# A deliberately small contractor pool so some contractors legitimately
# repeat often (feeds the "repeated contractor" fraud indicator)
CONTRACTORS = [fake.company() for _ in range(60)]
MPS = [fake.name() for _ in range(45)]

N_RECORDS = 1200
ANOMALY_FRACTION = 0.09  # ~9% intentionally unusual records


TODAY = datetime.date.today()


def random_date(start_year=2019, end_year=2024):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    delta = (end - start).days
    return start + datetime.timedelta(days=random.randint(0, delta))


def make_normal_record(i):
    state = random.choice(list(STATES_DISTRICTS.keys()))
    district = random.choice(STATES_DISTRICTS[state])
    category = random.choice(CATEGORIES)
    low, high = CATEGORY_COST_RANGE[category]

    estimated_cost = round(random.uniform(low, high), 2)
    sanctioned_amount = round(estimated_cost * random.uniform(0.95, 1.05), 2)
    released_amount = round(sanctioned_amount * random.uniform(0.6, 1.0), 2)

    completion_percentage = round(random.uniform(20, 100), 1)
    is_completed = completion_percentage >= 99.5

    if is_completed:
        # Fully historical: everything already happened in the past.
        start_date = TODAY - datetime.timedelta(days=random.randint(400, 2000))
        planned_duration = random.randint(90, 540)
        expected_completion_date = start_date + datetime.timedelta(days=planned_duration)
        actual_completion_date = expected_completion_date + datetime.timedelta(days=random.randint(-20, 40))
        project_status = 'Completed'
        actual_cost = round(estimated_cost * random.uniform(0.95, 1.12), 2)
        expenditure_amount = round(sanctioned_amount * random.uniform(0.9, 1.0), 2)
    else:
        # Ongoing project: anchor start_date recently so expected_completion_date
        # sits realistically close to "today" (not artificially years overdue).
        start_date = TODAY - datetime.timedelta(days=random.randint(30, 540))
        planned_duration = random.randint(90, 540)
        expected_completion_date = start_date + datetime.timedelta(days=planned_duration)
        actual_completion_date = None
        # Bias expected_completion_date to be near "today" +/- a reasonable window
        offset_from_today = random.randint(-120, 240)
        expected_completion_date = TODAY + datetime.timedelta(days=offset_from_today)
        project_status = 'Delayed' if offset_from_today < -15 else random.choice(['In Progress', 'Not Started'])
        actual_cost = round(estimated_cost * random.uniform(0.9, 1.1), 2)
        expenditure_amount = round(released_amount * (completion_percentage / 100) * random.uniform(0.85, 1.05), 2)

    lat_base, lon_base = STATE_COORDS[state]
    latitude = round(lat_base + random.uniform(-0.8, 0.8), 6)
    longitude = round(lon_base + random.uniform(-0.8, 0.8), 6)

    beneficiaries = int(random.uniform(200, 15000))

    return {
        'project_id': f'MPLAD{i:05d}',
        'project_name': f'{category} Development Project - {district}',
        'mp_name': random.choice(MPS),
        'mp_id': f'MP{random.randint(1, 45):03d}',
        'constituency': f'{district} Constituency',
        'district': district,
        'state': state,
        'project_category': category,
        'sanctioned_amount': sanctioned_amount,
        'released_amount': released_amount,
        'expenditure_amount': expenditure_amount,
        'estimated_cost': estimated_cost,
        'actual_cost': actual_cost,
        'start_date': start_date.isoformat(),
        'expected_completion_date': expected_completion_date.isoformat(),
        'actual_completion_date': actual_completion_date.isoformat() if actual_completion_date else '',
        'completion_percentage': completion_percentage,
        'contractor': random.choice(CONTRACTORS),
        'implementing_agency': random.choice(AGENCIES),
        'beneficiaries': beneficiaries,
        'latitude': latitude,
        'longitude': longitude,
        'project_status': project_status,
    }


def inject_anomaly(record):
    """Mutate a normal record into one of several deliberately unusual
    patterns, so the trained models have real anomalies to learn from."""
    kind = random.choice([
        'cost_overrun', 'low_completion_high_spend', 'delay', 'duplicate_location',
        'abnormal_cost_per_beneficiary', 'expenditure_spike', 'under_utilized',
    ])

    if kind == 'cost_overrun':
        record['actual_cost'] = round(record['estimated_cost'] * random.uniform(1.6, 2.8), 2)
        record['expenditure_amount'] = round(record['actual_cost'] * 0.9, 2)

    elif kind == 'low_completion_high_spend':
        record['completion_percentage'] = round(random.uniform(5, 30), 1)
        record['expenditure_amount'] = round(record['sanctioned_amount'] * random.uniform(0.8, 1.1), 2)
        record['project_status'] = 'Stalled'

    elif kind == 'delay':
        start = datetime.date.fromisoformat(record['start_date'])
        expected = datetime.date.fromisoformat(record['expected_completion_date'])
        record['actual_completion_date'] = (expected + datetime.timedelta(days=random.randint(180, 720))).isoformat()
        record['project_status'] = 'Delayed'

    elif kind == 'duplicate_location':
        # Snap to a small fixed set of coordinates to create clusters
        cluster_lat = round(20.5 + random.choice([0, 0, 0.001]), 3)
        cluster_lon = round(78.9 + random.choice([0, 0, 0.001]), 3)
        record['latitude'] = cluster_lat
        record['longitude'] = cluster_lon

    elif kind == 'abnormal_cost_per_beneficiary':
        record['beneficiaries'] = max(10, int(record['beneficiaries'] * 0.02))

    elif kind == 'expenditure_spike':
        record['expenditure_amount'] = round(record['sanctioned_amount'] * random.uniform(1.5, 2.2), 2)

    elif kind == 'under_utilized':
        record['expenditure_amount'] = round(record['released_amount'] * random.uniform(0.05, 0.25), 2)
        record['completion_percentage'] = round(random.uniform(5, 20), 1)

    return record


def generate_dataset(n=N_RECORDS):
    records = []
    for i in range(1, n + 1):
        rec = make_normal_record(i)
        if random.random() < ANOMALY_FRACTION:
            rec = inject_anomaly(rec)
        records.append(rec)

    # Inject a handful of near-duplicate project pairs (Stage 10 ground truth)
    for _ in range(15):
        base = random.choice(records).copy()
        dup = base.copy()
        dup['project_id'] = f'MPLAD{n + random.randint(1, 9999):05d}'
        dup['sanctioned_amount'] = round(base['sanctioned_amount'] * random.uniform(0.98, 1.02), 2)
        records.append(dup)

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=['project_id']).reset_index(drop=True)
    return df


if __name__ == '__main__':
    out_path = Path(__file__).resolve().parent / 'sample_mplad_data.csv'
    df = generate_dataset()
    df.to_csv(out_path, index=False)
    print(f'Generated {len(df)} records -> {out_path}')
