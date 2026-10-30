"""
Import data/sample_mplad_data.csv into MongoDB and run the full ML
pipeline (Stage 27 workflow), from the command line.

Run with:
    python manage.py import_sample_data
"""
import pandas as pd
from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand
from projects.services import import_csv_to_mongo, run_ml_analysis_and_save


class Command(BaseCommand):
    help = 'Import the sample MPLAD dataset into MongoDB and run ML analysis'

    def add_arguments(self, parser):
        parser.add_argument('--file', type=str, default=None, help='Path to a CSV file (defaults to data/sample_mplad_data.csv)')

    def handle(self, *args, **options):
        csv_path = Path(options['file']) if options['file'] else Path(settings.DATA_DIR) / 'sample_mplad_data.csv'
        if not csv_path.exists():
            self.stdout.write(self.style.ERROR(
                f'{csv_path} not found. Generate it first with: python data/generate_sample_data.py'))
            return

        self.stdout.write(f'Reading {csv_path} ...')
        df = pd.read_csv(csv_path)

        self.stdout.write('Importing into MongoDB ...')
        result = import_csv_to_mongo(df)
        if not result.get('success'):
            self.stdout.write(self.style.ERROR(result.get('error')))
            return
        self.stdout.write(self.style.SUCCESS(
            f"Inserted: {result['inserted']}, Updated: {result['updated']}, Errors: {result['error_count']}"))

        self.stdout.write('Running full ML analysis pipeline (this may take a minute) ...')
        ml_result = run_ml_analysis_and_save()
        self.stdout.write(self.style.SUCCESS(
            f"Analyzed {ml_result['analyzed']} projects. Duplicate pairs found: {ml_result['duplicate_pairs_found']}"))
        self.stdout.write(self.style.SUCCESS('Done. Visit /dashboard/ to view results.'))
