import random
from typing import Dict, List, Tuple

import numpy as np


class SyntheticDataGenerator:
    """
    Generates synthetic feature vectors for specific chart types (especially rare ones)
    based on their ideal statistical properties.
    """

    def __init__(self, chart_types: List[str], feature_keys: List[str]):
        self.chart_types = chart_types
        self.feature_keys = feature_keys
        self.default_noise = 0.05  # Base level of noise for features

    def generate_samples(
        self, chart_type: str, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates n_samples of feature vectors and labels for a specific chart type.
        """
        if chart_type not in self.chart_types:
            raise ValueError(f"Unknown chart type: {chart_type}")

        samples = []

        # Determine which rule function to use
        rule_func = getattr(
            self, f"_apply_{chart_type}_logic", self._apply_default_logic
        )

        for _ in range(n_samples):
            # 1. Start with a neutral base (small noise)
            feats = {
                k: random.uniform(0, self.default_noise) for k in self.feature_keys
            }

            # 2. Apply the specific chart logic
            rule_func(feats)

            # 3. Convert dictionary to the ordered feature vector array
            vector = np.array([feats.get(k, 0.0) for k in self.feature_keys])
            samples.append(vector)

        # Create label vector (one-hot)
        labels = np.zeros((n_samples, len(self.chart_types)))
        labels[:, self.chart_types.index(chart_type)] = 1

        return np.array(samples, dtype=float), labels.astype(int)

    def _apply_line_logic(self, f: Dict):
        """Line: Time/Ordinal vs. Numerical (strong time trend)."""
        f["is_time_num"] = random.uniform(0.9, 1.0)
        f["col1_is_time_series"] = 1.0
        f["time_granularity_level"] = random.uniform(0.5, 0.9)  # monthly/daily
        f["trend_strength"] = random.uniform(
            0.7, 1.0
        )  # Strong upward or downward trend

    def _apply_pie_logic(self, f: Dict):
        """Pie/Donut: 1 Cat (low cardinality, 3-10 unique) vs. 1 Num (positive count)."""
        f["is_cat_num"] = random.uniform(0.9, 1.0)
        f["col1_cardinality_ratio"] = random.uniform(0.01, 0.08)  # Low cardinality
        f["col1_unique_small"] = 1.0
        f["col2_has_negatives"] = 0.0
        f["col2_zero_ratio"] = random.uniform(0.0, 0.1)

    def _apply_donut_logic(self, f: Dict):
        # Very similar to pie, perhaps with slightly higher confidence or a specific flag
        self._apply_pie_logic(f)
        f["high_confidence_for_pie_donut"] = 1.0

    def _apply_scatter_logic(self, f: Dict):
        """Scatter: 2 Num (continuous) + some correlation."""
        f["is_num_num"] = random.uniform(0.9, 1.0)
        f["correlation_abs"] = random.uniform(0.3, 0.9)
        f["col1_unique_ratio"] = random.uniform(0.8, 1.0)
        f["col2_unique_ratio"] = random.uniform(0.8, 1.0)
        f["col1_is_binned"] = 0.0  # Not a binned chart
        f["col2_is_binned"] = 0.0

    def _apply_bar_logic(self, f: Dict):
        """Bar: 1 Cat (medium card) vs. 1 Num (aggregate count)."""
        f["is_cat_num"] = random.uniform(0.9, 1.0)
        f["col1_cardinality_ratio"] = random.uniform(
            0.05, 0.2
        )  # Medium cardinality (10-50 categories)
        f["col1_unique_small"] = 0.0  # Not super small
        f["col2_is_summed"] = 1.0  # Typically counts/sums

    def _apply_stacked_bar_logic(self, f: Dict):
        """Stacked Bar: 2 Cat (medium) vs. 1 Num, strong part-to-whole relationship."""
        # This is a multivariate chart: X (Cat), Y (Num), Color (Cat 2)
        f["is_multi_cat_num"] = 1.0
        f["is_cat_num"] = 0.5  # Weaker signal for the base pair
        f["col1_cardinality_medium"] = 1.0
        f["col2_cardinality_medium"] = 1.0
        f["part_to_whole_relationship"] = random.uniform(0.8, 1.0)
        f["col2_is_summed"] = 1.0

    def _apply_histogram_logic(self, f: Dict):
        """Histogram/Boxplot/Violin: 1 Num (non-trivial distribution)."""
        f["is_univariate"] = 1.0
        f["col1_is_numeric"] = 1.0
        f["col1_unique_ratio"] = random.uniform(0.05, 0.9)  # Not constant, not ID
        f["col1_coefficient_of_variation"] = random.uniform(0.01, 1.0)  # Has spread
        f[
            "col1_is_binned"
        ] = 1.0  # Feature that shows this numeric column is suitable for binning

    def _apply_boxplot_logic(self, f: Dict):
        self._apply_histogram_logic(f)
        f["col1_has_outliers"] = random.uniform(
            0.5, 1.0
        )  # Boxplots are good for outliers
        f["col1_skewness_abs"] = random.uniform(
            0.1, 0.9
        )  # Skewed data benefits from boxplot/violin

    def _apply_heatmap_logic(self, f: Dict):
        """Heatmap: 2 Categorical (or binned Num) with low/medium cardinality."""
        # Typically used for cross-tabulation of two categories/binned measures.
        f["is_multi_cat_cat"] = 1.0
        f["is_cat_cat"] = 1.0
        f["col1_cardinality_medium"] = 1.0
        f["col2_cardinality_medium"] = 1.0
        f["density_of_values"] = random.uniform(
            0.7, 1.0
        )  # Must have values in most cells

    def _apply_bubble_logic(self, f: Dict):
        """Bubble: 3 Numerical variables (X, Y, Size)."""
        f["is_multi_num"] = 1.0
        f["num_features_count"] = random.uniform(0.9, 1.0)
        f["col3_is_magnitude"] = 1.0  # The size variable is non-binary/continuous
        self._apply_scatter_logic(f)  # Inherits scatter properties for X/Y

    def _apply_treemap_logic(self, f: Dict):
        """Treemap/Sunburst: Hierarchical Categorical data and 1 Numerical measure."""
        f["is_hierarchical"] = 1.0
        f["cat_features_count"] = random.uniform(0.7, 1.0)
        f["hierarchy_depth"] = random.uniform(0.5, 1.0)  # 2+ levels of categories
        f["col_measure_is_sum"] = 1.0

    def _apply_map_logic(self, f: Dict):
        """Map: Geospatial data and 1 Numerical measure."""
        f["is_geospatial"] = 1.0
        f["col1_is_geo_code"] = 1.0  # Country/State code
        f["col2_is_measure"] = 1.0
        f["geo_resolution_level"] = random.uniform(0.5, 1.0)  # State/Country level

    def _apply_sankey_logic(self, f: Dict):
        """Sankey: Flow or network data (Nodes/Links or Source/Target/Value)."""
        f["is_network_flow"] = 1.0
        f["col1_is_source"] = 1.0
        f["col2_is_target"] = 1.0
        f["col3_is_flow_value"] = 1.0
        f["col1_cardinality_medium"] = 1.0
        f["col2_cardinality_medium"] = 1.0

    def _apply_parallel_coordinates_logic(self, f: Dict):
        """Parallel Coordinates: Many Numerical columns with complex relationships."""
        f["is_high_dim_num"] = 1.0
        f["num_features_count"] = random.uniform(0.9, 1.0)
        f["features_count_high"] = 1.0  # Requires many features (4+)
        f["multiple_weak_correlations"] = random.uniform(
            0.7, 1.0
        )  # Highlights patterns

    def _apply_default_logic(self, f: Dict):
        """Fallback for less specific or uncommon charts like 'area', 'waterfall', 'timeline'."""
        # For Area/Timeline (often Time-Num, similar to line)
        f["is_time_num"] = random.uniform(0.5, 0.8)
        f["col1_is_time_series"] = random.uniform(0.5, 0.8)
        f["trend_strength"] = random.uniform(0.3, 0.7)
