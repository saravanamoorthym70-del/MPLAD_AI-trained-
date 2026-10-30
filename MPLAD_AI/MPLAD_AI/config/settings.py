"""
Django settings for MPLAD_AI project.
AI-Powered Detection of Anomalies, Fraud and Inefficiencies in MPLADS.
"""
import os
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
import mongoengine

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-fallback-key')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', '127.0.0.1,localhost').split(',')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'rest_framework',
    'rest_framework_simplejwt',
    'corsheaders',

    'accounts',
    'projects',
    'analytics',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'
ASGI_APPLICATION = 'config.asgi.application'

# NOTE: Django's built-in ORM still needs a database for its internal apps
# (sessions, admin, auth's default user model is NOT used for app data).
# The actual application data (projects, transactions, anomalies, users)
# lives entirely in MongoDB via MongoEngine. A local SQLite file is kept
# only to satisfy Django's internal session/admin framework.
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'django_internal.sqlite3',
    }
}

# ---- MongoDB (application database) ----
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/mplad_ai')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'mplad_ai')

if MONGO_URI.startswith('mongomock://'):
    # Allows running the whole app against an in-memory Mongo for local
    # testing/CI without a real MongoDB server installed.
    import mongomock
    mongoengine.connect(db=MONGO_DB_NAME, alias='default', mongo_client_class=mongomock.MongoClient)
else:
    mongoengine.connect(db=MONGO_DB_NAME, host=MONGO_URI, alias='default')

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'accounts.authentication.MongoJWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
}

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=8),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
}

CORS_ALLOW_ALL_ORIGINS = DEBUG

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/dashboard/'

# Risk score weighting (Stage 12)
RISK_WEIGHTS = {
    'anomaly': 0.30,
    'financial': 0.25,
    'delay': 0.20,
    'inefficiency': 0.15,
    'duplicate': 0.10,
}

ML_MODELS_DIR = BASE_DIR / 'ml_engine' / 'models'
DATA_DIR = BASE_DIR / 'data'
