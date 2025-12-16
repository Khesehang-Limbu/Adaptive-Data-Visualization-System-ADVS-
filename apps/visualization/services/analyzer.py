from typing import Dict, List

import pandas as pd

from apps.visualization.services.profiler import DatasetProfiler


class DatasetAnalyzer:
    def __init__(self):
        self.profiler = DatasetProfiler()

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform Comprehensive Dataset Analysis and Return a complete profile with metadata.
        :param df:
        :return: Dictionary of Metadata
        """
        filter_result = self.profiler.get_visualizable_columns(df)
        useful_cols = filter_result["visualizable_columns"]

        if not useful_cols:
            useful_cols = list(df.columns)

        df_clean = df[useful_cols]

        profile = self.profiler.profile(df_clean)
        profile["excluded_columns"] = filter_result["excluded_columns"]

        return profile

    def get_column_types(self, profile: Dict) -> Dict[str, List[str]]:
        """
        Extract Columns Grouped By Types
        :param profile:
        :return: Dictionary of column types mapped to a list of column names
        """
        return {
            "numeric": [
                c["name"]
                for c in profile["columns"]
                if c["dtype"] == "numeric"
                and not c.get("flags", {}).get("is_id_like", False)
            ],
            "categorical": [
                c["name"]
                for c in profile["columns"]
                if c["dtype"] == "categorical"
                and not c.get("flags", {}).get("is_id_like", False)
            ],
            "datetime": [
                c["name"] for c in profile["columns"] if c["dtype"] == "datetime"
            ],
            "text": [
                c["name"]
                for c in profile["columns"]
                if c["dtype"] == "text"
                and not c.get("flags", {}).get("is_long_text", False)
                and not c.get("flags", {}).get("is_id_like", False)
            ],
        }

    def get_patterns(self, profile: Dict) -> Dict:
        """
        Extract the patterns within the dataset
        :param profile:
        :return: A Dictionary of patterns
        """
        return {
            "strong_correlation": profile.get("correlation_analysis", {}).get(
                "correlation_pairs", []
            ),
            "has_strong_correlations": profile.get("correlation_analysis", {}).get(
                "has_strong_correlations", False
            ),
            "hierarchical_pairs": profile.get("hierarchical_analysis", {}).get(
                "hierarchical_pairs", []
            ),
            "has_hierarchy": profile.get("hierarchical_analysis", {}).get(
                "has_hierarchy", False
            ),
        }
