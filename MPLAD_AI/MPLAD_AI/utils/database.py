"""
Reusable MongoDB connection helpers.

MongoEngine's `connect()` is already called once in config/settings.py at
startup (alias='default'), so every mongoengine.Document subclass in this
project is automatically bound to MongoDB. This module exposes a couple of
small helpers for code (e.g. ml_engine scripts run outside Django, or raw
PyMongo access for bulk operations) that needs a direct connection/handle.
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / '.env')

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/mplad_ai')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'mplad_ai')

_client = None


def get_client():
    """Return a singleton PyMongo client."""
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client


def get_db():
    """Return the application's MongoDB database handle (raw PyMongo)."""
    return get_client()[MONGO_DB_NAME]


def ensure_indexes():
    """Create indexes used by the application for query performance."""
    db = get_db()
    db.projects.create_index('project_id', unique=True)
    db.projects.create_index('state')
    db.projects.create_index('district')
    db.projects.create_index('mp_id')
    db.projects.create_index([('latitude', 1), ('longitude', 1)])
    db.transactions.create_index('project_id')
    db.transactions.create_index('transaction_date')
    db.anomalies.create_index('project_id')
    db.anomalies.create_index('severity')
    db.ml_predictions.create_index('project_id', unique=True)
    db.users.create_index('username', unique=True)
    db.users.create_index('email', unique=True)
