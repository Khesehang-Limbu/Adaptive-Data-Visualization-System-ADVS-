import pandas as pd
from django.conf import settings
from django.test import TestCase

from apps.visualization.services.profiler import DatasetProfiler

DATA_DIR = settings.BASE_DIR / "apps/visualization/ml/data/raw"


class TestDatasetProfiler(TestCase):
    def setUp(self):
        self.profiler = DatasetProfiler()

    def test_profiler_chart_hints(self):
        datasets = {
            f"{DATA_DIR}/uber.csv": ["line"],
        }

        for fname, expected in datasets.items():
            df = pd.read_csv(fname)

            viz_cols = self.profiler.get_visualizable_columns(df)
            profile = self.profiler.profile(df[viz_cols.get("visualizable_columns")])
            hints = self.profiler._generate_chart_hints(profile).get(
                "primary_suggestions"
            )

            recommended_charts = []
            for hint in hints:
                recommended_charts.append(hint.get("chart"))

            print(
                f"File: {fname}, Expected: {expected}, Recommended: {set(recommended_charts)}"
            )
            self.assertTrue(set(expected).issubset(set(recommended_charts)))
