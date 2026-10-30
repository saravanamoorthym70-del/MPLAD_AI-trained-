"""
User model stored in MongoDB (collection: users).

Roles: Admin, Analyst, Auditor, Viewer (Stage 22 — role-based permissions).
"""
import datetime
import mongoengine as me
from django.contrib.auth.hashers import make_password, check_password

ROLE_CHOICES = ('Admin', 'Analyst', 'Auditor', 'Viewer')


class User(me.Document):
    username = me.StringField(required=True, unique=True, max_length=150)
    email = me.EmailField(required=True, unique=True)
    password = me.StringField(required=True)  # stored as a Django hash
    role = me.StringField(required=True, choices=ROLE_CHOICES, default='Viewer')
    full_name = me.StringField(max_length=200, default='')
    is_active = me.BooleanField(default=True)
    created_at = me.DateTimeField(default=datetime.datetime.utcnow)

    meta = {
        'collection': 'users',
        'indexes': ['username', 'email', 'role'],
    }

    # DRF's IsAuthenticated permission (and other auth machinery) checks
    # these two attributes, which Django's built-in ORM User provides
    # automatically. Our MongoEngine User must expose them explicitly.
    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)

    # ---- Role permission helpers (Stage 22) ----
    def can_edit_projects(self):
        return self.role == 'Admin'

    def can_run_ml_analysis(self):
        return self.role in ('Admin', 'Analyst')

    def can_review_anomalies(self):
        return self.role in ('Admin', 'Auditor')

    def can_view_reports(self):
        return self.role in ('Admin', 'Analyst', 'Auditor')

    def to_dict(self):
        return {
            'id': str(self.id),
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
