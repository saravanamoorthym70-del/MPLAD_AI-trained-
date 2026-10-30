"""Serializers for auth endpoints. Since the User model is MongoEngine
(not Django ORM), plain rest_framework.Serializer is used for validation."""
from rest_framework import serializers
from accounts.models import ROLE_CHOICES


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    email = serializers.EmailField()
    password = serializers.CharField(min_length=6, write_only=True)
    full_name = serializers.CharField(max_length=200, required=False, allow_blank=True)
    role = serializers.ChoiceField(choices=ROLE_CHOICES, required=False, default='Viewer')


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)
