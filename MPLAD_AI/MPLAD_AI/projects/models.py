"""
MongoDB models (Stage 4) for MPLADS project data.

Collections: projects, transactions, anomalies, ml_predictions
"""
import datetime
import mongoengine as me

PROJECT_CATEGORIES = (
    'Roads', 'Bridges', 'Schools', 'Hospitals', 'Drinking Water',
    'Sanitation', 'Community Halls', 'Electricity',
    'Public Infrastructure', 'Sports Facilities',
)

PROJECT_STATUSES = ('Not Started', 'In Progress', 'Delayed', 'Completed', 'Stalled')

SEVERITY_CHOICES = ('Low', 'Medium', 'High', 'Critical')
REVIEW_STATUS_CHOICES = ('Pending', 'Under Review', 'Verified', 'Dismissed')


class Project(me.Document):
    project_id = me.StringField(required=True, unique=True, max_length=50)
    project_name = me.StringField(required=True, max_length=300)
    mp_name = me.StringField(required=True, max_length=200)
    mp_id = me.StringField(max_length=50)
    constituency = me.StringField(max_length=200)
    district = me.StringField(max_length=200)
    state = me.StringField(max_length=200)
    implementing_agency = me.StringField(max_length=300)
    contractor = me.StringField(max_length=300)
    project_category = me.StringField(choices=PROJECT_CATEGORIES)

    sanctioned_amount = me.FloatField(default=0)
    released_amount = me.FloatField(default=0)
    expenditure_amount = me.FloatField(default=0)
    estimated_cost = me.FloatField(default=0)
    actual_cost = me.FloatField(default=0)

    start_date = me.DateField()
    expected_completion_date = me.DateField()
    actual_completion_date = me.DateField(null=True)

    project_status = me.StringField(choices=PROJECT_STATUSES, default='Not Started')
    completion_percentage = me.FloatField(default=0)

    latitude = me.FloatField(null=True)
    longitude = me.FloatField(null=True)

    beneficiaries = me.IntField(default=0)

    # ---- Derived / engineered features (Stage 6), cached on the document ----
    expenditure_ratio = me.FloatField(null=True)
    cost_overrun = me.FloatField(null=True)
    delay_days = me.IntField(null=True)
    utilization_ratio = me.FloatField(null=True)
    completion_efficiency = me.FloatField(null=True)
    cost_per_beneficiary = me.FloatField(null=True)
    project_duration = me.IntField(null=True)

    created_at = me.DateTimeField(default=datetime.datetime.utcnow)
    updated_at = me.DateTimeField(default=datetime.datetime.utcnow)

    meta = {
        'collection': 'projects',
        'indexes': [
            'project_id', 'state', 'district', 'mp_id', 'project_category',
            'project_status', ('latitude', 'longitude'),
        ],
    }

    def to_dict(self):
        return {
            'project_id': self.project_id,
            'project_name': self.project_name,
            'mp_name': self.mp_name,
            'mp_id': self.mp_id,
            'constituency': self.constituency,
            'district': self.district,
            'state': self.state,
            'implementing_agency': self.implementing_agency,
            'contractor': self.contractor,
            'project_category': self.project_category,
            'sanctioned_amount': self.sanctioned_amount,
            'released_amount': self.released_amount,
            'expenditure_amount': self.expenditure_amount,
            'estimated_cost': self.estimated_cost,
            'actual_cost': self.actual_cost,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'expected_completion_date': self.expected_completion_date.isoformat() if self.expected_completion_date else None,
            'actual_completion_date': self.actual_completion_date.isoformat() if self.actual_completion_date else None,
            'project_status': self.project_status,
            'completion_percentage': self.completion_percentage,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'beneficiaries': self.beneficiaries,
            'expenditure_ratio': self.expenditure_ratio,
            'cost_overrun': self.cost_overrun,
            'delay_days': self.delay_days,
            'utilization_ratio': self.utilization_ratio,
            'completion_efficiency': self.completion_efficiency,
            'cost_per_beneficiary': self.cost_per_beneficiary,
            'project_duration': self.project_duration,
        }


class Transaction(me.Document):
    transaction_id = me.StringField(required=True, unique=True, max_length=50)
    project_id = me.StringField(required=True, max_length=50)
    transaction_date = me.DateField()
    amount = me.FloatField(default=0)
    transaction_type = me.StringField(max_length=100)
    vendor = me.StringField(max_length=300)
    contractor = me.StringField(max_length=300)
    payment_status = me.StringField(max_length=50, default='Pending')
    description = me.StringField(max_length=1000)

    meta = {
        'collection': 'transactions',
        'indexes': ['project_id', 'transaction_date'],
    }

    def to_dict(self):
        return {
            'transaction_id': self.transaction_id,
            'project_id': self.project_id,
            'transaction_date': self.transaction_date.isoformat() if self.transaction_date else None,
            'amount': self.amount,
            'transaction_type': self.transaction_type,
            'vendor': self.vendor,
            'contractor': self.contractor,
            'payment_status': self.payment_status,
            'description': self.description,
        }


class Anomaly(me.Document):
    anomaly_id = me.StringField(required=True, unique=True, max_length=50)
    project_id = me.StringField(required=True, max_length=50)
    anomaly_type = me.StringField(max_length=150)
    anomaly_score = me.FloatField(default=0)
    severity = me.StringField(choices=SEVERITY_CHOICES, default='Low')
    detected_features = me.DictField()
    explanation = me.StringField(max_length=2000)
    detected_at = me.DateTimeField(default=datetime.datetime.utcnow)
    review_status = me.StringField(choices=REVIEW_STATUS_CHOICES, default='Pending')

    meta = {
        'collection': 'anomalies',
        'indexes': ['project_id', 'severity', 'review_status', 'anomaly_type'],
    }

    def to_dict(self):
        return {
            'anomaly_id': self.anomaly_id,
            'project_id': self.project_id,
            'anomaly_type': self.anomaly_type,
            'anomaly_score': self.anomaly_score,
            'severity': self.severity,
            'detected_features': self.detected_features,
            'explanation': self.explanation,
            'detected_at': self.detected_at.isoformat() if self.detected_at else None,
            'review_status': self.review_status,
        }


class MLPrediction(me.Document):
    project_id = me.StringField(required=True, unique=True, max_length=50)
    anomaly_probability = me.FloatField(default=0)
    fraud_risk_score = me.FloatField(default=0)
    inefficiency_score = me.FloatField(default=0)
    delay_risk_score = me.FloatField(default=0)
    overall_risk_score = me.FloatField(default=0)
    risk_level = me.StringField(default='Low')
    reasons = me.ListField(me.StringField())
    contributing_factors = me.DictField()
    prediction_date = me.DateTimeField(default=datetime.datetime.utcnow)
    model_version = me.StringField(default='v1.0')

    meta = {
        'collection': 'ml_predictions',
        'indexes': ['project_id', 'overall_risk_score', 'risk_level'],
    }

    def to_dict(self):
        return {
            'project_id': self.project_id,
            'anomaly_probability': self.anomaly_probability,
            'fraud_risk_score': self.fraud_risk_score,
            'inefficiency_score': self.inefficiency_score,
            'delay_risk_score': self.delay_risk_score,
            'overall_risk_score': self.overall_risk_score,
            'risk_level': self.risk_level,
            'reasons': self.reasons,
            'contributing_factors': self.contributing_factors,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'model_version': self.model_version,
        }
