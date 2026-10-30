"""
Seed demo users for all four roles (Stage 22 / Stage 32 sample credentials).

Run with:
    python manage.py seed_users
"""
from django.core.management.base import BaseCommand
from accounts.models import User

DEMO_USERS = [
    ('admin', 'admin@mplad.gov.in', 'Admin@123', 'Admin', 'System Administrator'),
    ('analyst', 'analyst@mplad.gov.in', 'Analyst@123', 'Analyst', 'Data Analyst'),
    ('auditor', 'auditor@mplad.gov.in', 'Auditor@123', 'Auditor', 'Compliance Auditor'),
    ('viewer', 'viewer@mplad.gov.in', 'Viewer@123', 'Viewer', 'Public Viewer'),
]


class Command(BaseCommand):
    help = 'Create demo users for Admin, Analyst, Auditor, and Viewer roles'

    def handle(self, *args, **options):
        for username, email, password, role, full_name in DEMO_USERS:
            if User.objects(username=username).first():
                self.stdout.write(self.style.WARNING(f'User "{username}" already exists, skipping.'))
                continue
            user = User(username=username, email=email, role=role, full_name=full_name)
            user.set_password(password)
            user.save()
            self.stdout.write(self.style.SUCCESS(f'Created {role} user: {username} / {password}'))

        self.stdout.write(self.style.SUCCESS('\nDemo users ready. Login at /login/'))
