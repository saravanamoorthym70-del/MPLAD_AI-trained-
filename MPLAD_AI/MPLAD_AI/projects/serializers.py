"""Serializers for the projects REST API. MongoEngine documents are
serialized manually via to_dict() in views, but these plain
rest_framework.Serializer classes validate incoming write payloads."""
from rest_framework import serializers
from projects.models import PROJECT_CATEGORIES, PROJECT_STATUSES


class ProjectWriteSerializer(serializers.Serializer):
    project_id = serializers.CharField(max_length=50)
    project_name = serializers.CharField(max_length=300)
    mp_name = serializers.CharField(max_length=200)
    mp_id = serializers.CharField(max_length=50, required=False, allow_blank=True)
    constituency = serializers.CharField(max_length=200, required=False, allow_blank=True)
    district = serializers.CharField(max_length=200, required=False, allow_blank=True)
    state = serializers.CharField(max_length=200, required=False, allow_blank=True)
    implementing_agency = serializers.CharField(max_length=300, required=False, allow_blank=True)
    contractor = serializers.CharField(max_length=300, required=False, allow_blank=True)
    project_category = serializers.ChoiceField(choices=PROJECT_CATEGORIES)

    sanctioned_amount = serializers.FloatField(min_value=0)
    released_amount = serializers.FloatField(min_value=0, required=False, default=0)
    expenditure_amount = serializers.FloatField(min_value=0, required=False, default=0)
    estimated_cost = serializers.FloatField(min_value=0, required=False, default=0)
    actual_cost = serializers.FloatField(min_value=0, required=False, default=0)

    start_date = serializers.DateField()
    expected_completion_date = serializers.DateField()
    actual_completion_date = serializers.DateField(required=False, allow_null=True)

    project_status = serializers.ChoiceField(choices=PROJECT_STATUSES, required=False, default='Not Started')
    completion_percentage = serializers.FloatField(min_value=0, max_value=100, required=False, default=0)

    latitude = serializers.FloatField(min_value=-90, max_value=90, required=False, allow_null=True)
    longitude = serializers.FloatField(min_value=-180, max_value=180, required=False, allow_null=True)

    beneficiaries = serializers.IntegerField(min_value=0, required=False, default=0)


class AnomalyReviewSerializer(serializers.Serializer):
    review_status = serializers.ChoiceField(choices=('Pending', 'Under Review', 'Verified', 'Dismissed'))
