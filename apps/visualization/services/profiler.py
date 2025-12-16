import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

from apps.visualization.ml.constants import (
    CHART_FAMILIES,
    CURRENCY_PATTERN,
    EMAIL_PATTERN,
    URL_PATTERN,
    UUID_PATTERN,
)


class DatasetProfiler:
    def __init__(
        self,
        sample_size: int = 200,
        long_text_threshold: int = 150,
        represent_k: int = 5,
        max_corr_columns: int = 40,
        max_kmeans_rows: int = 10000,
        url_threshold: float = 0.3,
        email_threshold: float = 0.5,
        hash_threshold: float = 0.7,
        id_uniqueness_threshold: float = 0.95,
        outlier_iqr_multiplier: float = 1.5,
        enable_logging: bool = False,
    ):
        self.sample_size = sample_size
        self.long_text_threshold = long_text_threshold
        self.represent_k = represent_k
        self.max_corr_columns = max_corr_columns
        self.max_kmeans_rows = max_kmeans_rows
        self.url_threshold = url_threshold
        self.email_threshold = email_threshold
        self.hash_threshold = hash_threshold
        self.id_uniqueness_threshold = id_uniqueness_threshold
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.enable_logging = enable_logging
        self._profile_cache = {}

    def _df_fingerprint(self, df: pd.DataFrame) -> tuple:
        return (
            id(df),
            df.shape,
            tuple(df.columns),
        )

    def _log(self, message: str) -> None:
        """Log message if logging is enabled."""
        if self.enable_logging:
            print(f"[DatasetProfiler] {message}")

    def _sample(self, series: pd.Series) -> pd.Series:
        """
        Sample values from a series for pattern detection.

        Parameters
        ----------
        series : pd.Series
            Input series to sample from

        Returns
        -------
        pd.Series
            Sampled values as strings
        """
        n = min(self.sample_size, len(series))
        if n == 0:
            return pd.Series([], dtype=str)

        clean = series.dropna()
        if len(clean) == 0:
            return pd.Series([], dtype=str)

        return clean.astype(str).sample(n, random_state=0)

    def _is_probably_url(self, series: pd.Series) -> bool:
        """Check if series contains URL values."""
        if series.dtype != object:
            return False
        sample = self._sample(series)
        if len(sample) == 0:
            return False
        return sample.str.contains(URL_PATTERN, na=False).mean() > self.url_threshold

    def _is_probably_email(self, series: pd.Series) -> bool:
        """Check if series contains email addresses."""
        if series.dtype != object:
            return False
        sample = self._sample(series)
        if len(sample) == 0:
            return False
        return sample.str.match(EMAIL_PATTERN, na=False).mean() > self.email_threshold

    def _is_probably_image(self, series: pd.Series) -> bool:
        """Check if series contains image file paths."""
        if series.dtype != object:
            return False
        sample = self._sample(series)
        if len(sample) == 0:
            return False
        img_exts = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico")
        return sample.str.lower().str.endswith(img_exts).mean() > self.url_threshold

    def _is_probably_hash(self, series: pd.Series) -> bool:
        """Check if series contains hash values (MD5, SHA, etc.)."""
        if series.dtype != object:
            return False
        sample = self._sample(series)
        if len(sample) == 0:
            return False
        return (
            sample.str.fullmatch(r"[a-fA-F0-9]{32,}", na=False).mean()
            > self.hash_threshold
        )

    def _is_probably_currency(self, series: pd.Series) -> bool:
        """Check if series contains currency values."""
        if series.dtype != object:
            return False
        sample = self._sample(series)
        if len(sample) == 0:
            return False
        return (
            sample.str.match(CURRENCY_PATTERN, na=False).mean() > self.email_threshold
        )

    def _is_probably_percentage(self, series: pd.Series) -> bool:
        """Check if series contains percentage values."""
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) == 0:
                return False
            in_percent_range = ((non_null >= 0) & (non_null <= 100)).mean() > 0.9
            in_decimal_range = ((non_null >= 0) & (non_null <= 1)).mean() > 0.9
            return in_percent_range or in_decimal_range
        elif series.dtype == object:
            sample = self._sample(series)
            if len(sample) == 0:
                return False
            return sample.str.contains(r"%", na=False).mean() > self.email_threshold
        return False

    def _is_probably_geospatial(self, series: pd.Series) -> bool:
        """Check if series contains latitude or longitude coordinates."""
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) == 0:
                return False
            lat_like = ((non_null >= -90) & (non_null <= 90)).mean() > 0.9
            lon_like = ((non_null >= -180) & (non_null <= 180)).mean() > 0.9
            return lat_like or lon_like
        return False

    def _is_long_text(self, series: pd.Series) -> bool:
        """Check if series contains long text values."""
        if series.dtype != object:
            return False
        sample = self._sample(series)
        if len(sample) == 0:
            return False
        lengths = sample.str.len()
        return lengths.mean() > self.long_text_threshold

    def _looks_like_uuid(self, s: str) -> bool:
        """Check if string matches UUID format."""
        return bool(UUID_PATTERN.fullmatch(s.strip()))

    def _is_id_like(self, series: pd.Series, total_rows: int) -> bool:
        """
        Detect if column is likely an ID/primary key.

        Uses uniqueness ratio and value patterns to identify ID columns.
        """
        if pd.api.types.is_datetime64_any_dtype(series):
            return False

        if total_rows < 30:
            return False

        nunique = series.nunique(dropna=False)
        ratio = nunique / total_rows

        if ratio >= self.id_uniqueness_threshold:
            if series.dtype == object:
                sample = series.dropna().astype(str).head(30)
                if len(sample) == 0:
                    return False
                # UUID pattern
                if sample.apply(self._looks_like_uuid).mean() > 0.3:
                    return True
                # Short consistent-length strings
                if sample.str.len().mean() < 15:
                    return True

            if pd.api.types.is_numeric_dtype(series):
                s = series.dropna().astype(str)
                if len(s) > 0 and s.str.len().nunique() == 1:
                    return True
            return True
        return False

    def _detect_ordinal_statistical(
        self, cat_series: pd.Series, num_series: pd.Series
    ) -> float:
        """
        Returns ordinal strength score in [0, 1].
        0.0 means not ordinal.
        """
        if cat_series.nunique() > 20:
            return 0.0

        df = pd.DataFrame({"cat": cat_series, "num": num_series}).dropna()
        if len(df) < 20:
            return 0.0

        medians = df.groupby("cat")["num"].median()
        if medians.nunique() < 3:
            return 0.0

        ranks = medians.rank().values
        values = medians.values

        corr = pd.Series(ranks).corr(pd.Series(values), method="spearman")

        if pd.isna(corr):
            return 0.0

        return float(abs(corr))

    def _detect_ordinal(self, series: pd.Series) -> bool:
        """
        Detect if categorical column has ordinal (ordered) nature.
        """
        if series.nunique() > 20:
            return False

        sample_vals = series.dropna().astype(str).str.lower().unique()

        # Common ordinal patterns
        ordinal_patterns = [
            ["low", "medium", "high"],
            ["small", "medium", "large"],
            ["poor", "fair", "good", "excellent"],
            ["never", "rarely", "sometimes", "often", "always"],
            ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
            ["very low", "low", "medium", "high", "very high"],
        ]

        for pattern in ordinal_patterns:
            if set(sample_vals).issubset(set(pattern)):
                return True

        # Numeric-like categories (e.g., "1st", "2nd", "grade 1", "level 5")
        if all(any(c.isdigit() for c in str(v)) for v in sample_vals if v):
            return True

        return False

    def _sample_values(self, series: pd.Series, k: int = 3) -> List[str]:
        """Get sample values from series for display."""
        clean = series.dropna()
        if len(clean) == 0:
            return []
        n = min(k, len(clean))
        return list(clean.astype(str).sample(n, random_state=0))

    def _detect_time_series(self, series: pd.Series) -> Dict:
        """
        Analyze datetime series for time series properties.

        Returns information about sequentiality, gaps, and frequency.
        """
        if pd.api.types.is_datetime64_any_dtype(series):
            return {
                "is_datetime": True,
                "is_sequential": self._is_sequential_dates(series),
                "has_gaps": self._has_date_gaps(series),
                "time_span_days": (series.max() - series.min()).days
                if len(series) > 0
                else 0,
                "sampling_frequency": self._detect_frequency(series),
            }

        # Try parsing as datetime
        if series.dtype == object:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(series.dropna().head(100), errors="coerce")
                    parse_rate = parsed.notna().mean()
                    if parse_rate > 0.8:
                        return {
                            "is_datetime": False,
                            "is_parseable_datetime": True,
                            "parse_success_rate": float(parse_rate),
                        }
            except Exception as e:
                self._log(f"Datetime parsing failed: {e}")

        return {"is_datetime": False}

    def _detect_frequency(self, series: pd.Series) -> str:
        """Detect sampling frequency of datetime series."""
        if len(series) < 3:
            return "unknown"

        sorted_dates = series.dropna().sort_values()
        if len(sorted_dates) < 3:
            return "unknown"

        diffs = sorted_dates.diff().dropna()
        if len(diffs) == 0:
            return "unknown"

        median_diff = diffs.median()

        if median_diff <= pd.Timedelta(hours=1):
            return "sub_hourly"
        elif median_diff <= pd.Timedelta(days=1):
            return "daily"
        elif median_diff <= pd.Timedelta(days=7):
            return "weekly"
        elif median_diff <= pd.Timedelta(days=31):
            return "monthly"
        elif median_diff <= pd.Timedelta(days=92):
            return "quarterly"
        elif median_diff <= pd.Timedelta(days=366):
            return "yearly"
        else:
            return "irregular"

    def _is_sequential_dates(self, series: pd.Series) -> bool:
        if len(series) < 2:
            return False

        sorted_dates = series.dropna().sort_values()
        if len(sorted_dates) < 2:
            return False

        diffs = sorted_dates.diff().dropna()
        if len(diffs) == 0:
            return False

        # Convert Timedelta to seconds (float)
        diffs_seconds = diffs.dt.total_seconds()
        mean_diff = diffs_seconds.mean()
        if mean_diff == 0:
            return False
        std_ratio = diffs_seconds.std() / mean_diff
        return std_ratio < 1.0

    def _has_date_gaps(self, series: pd.Series) -> bool:
        """Detect significant gaps in datetime series."""
        if len(series) < 2:
            return False

        sorted_dates = series.dropna().sort_values()
        if len(sorted_dates) < 2:
            return False

        diffs = sorted_dates.diff().dropna()
        if len(diffs) == 0:
            return False

        median_diff = diffs.median()
        has_gap = (diffs > median_diff * 3).any()
        return bool(has_gap)

    def _analyze_cardinality(self, series: pd.Series, total_rows: int) -> Dict:
        """
        Analyze cardinality and recommend suitable chart types.
        """
        unique_count = series.nunique()
        unique_ratio = unique_count / total_rows if total_rows > 0 else 0

        return {
            "unique_count": int(unique_count),
            "unique_ratio": float(unique_ratio),
            "cardinality_category": self._categorize_cardinality(
                unique_count, total_rows
            ),
            "is_suitable_for_pie": 2 <= unique_count <= 8,
            "is_suitable_for_bar": 2 <= unique_count <= 30,
            "is_suitable_for_treemap": unique_count > 20,
            "is_likely_id": unique_ratio > self.id_uniqueness_threshold,
            "is_high_cardinality": unique_count > 100,
        }

    def _categorize_cardinality(self, unique_count: int, total_rows: int) -> str:
        """Categorize cardinality level."""
        if unique_count == 1:
            return "constant"
        elif unique_count == 2:
            return "binary"
        elif unique_count <= 8:
            return "low"
        elif unique_count <= 30:
            return "medium"
        elif unique_count <= 100:
            return "high"
        elif unique_count / total_rows > 0.9:
            return "unique"
        else:
            return "very_high"

    def _analyze_distribution(self, series: pd.Series) -> Dict:
        """
        Comprehensive distribution analysis for numeric data.
        """
        clean = series.dropna()
        if len(clean) < 10:
            return {}

        try:
            skewness = float(clean.skew())
            kurtosis = float(clean.kurtosis())
        except Exception as e:
            self._log(f"Distribution metrics failed: {e}")
            skewness = 0.0
            kurtosis = 0.0

        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "is_normal": self._test_normality(clean),
            "has_outliers": self._detect_outliers(clean),
            "outlier_count": int(self._count_outliers(clean)),
            "outlier_ratio": float(self._count_outliers(clean) / len(clean)),
            "distribution_type": self._classify_distribution(clean),
            "coefficient_of_variation": float(
                clean.std() / (abs(clean.mean()) + 1e-10)
            ),
            "range_normalized": float(
                (clean.max() - clean.min()) / (abs(clean.median()) + 1e-10)
            ),
            "zero_ratio": float((clean == 0).sum() / len(clean)),
            "negative_ratio": float((clean < 0).sum() / len(clean)),
        }

    def _test_normality(self, series: pd.Series) -> bool:
        """Simple normality test based on skewness and kurtosis."""
        if len(series) < 20:
            return False
        try:
            skew = abs(series.skew())
            kurt = abs(series.kurtosis())
            return skew < 1.0 and kurt < 3.0
        except:
            return False

    def _detect_outliers(self, series: pd.Series) -> bool:
        """Check if series has outliers."""
        return self._count_outliers(series) > 0

    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_iqr_multiplier * IQR
            upper_bound = Q3 + self.outlier_iqr_multiplier * IQR
            return int(((series < lower_bound) | (series > upper_bound)).sum())
        except:
            return 0

    def _classify_distribution(self, series: pd.Series) -> str:
        """Classify distribution shape."""
        try:
            skew = series.skew()
            kurt = series.kurtosis()

            if abs(skew) < 0.5 and abs(kurt) < 1.0:
                return "normal"
            elif skew > 1.0:
                return "right_skewed"
            elif skew < -1.0:
                return "left_skewed"
            elif kurt > 3.0:
                return "heavy_tailed"
            elif kurt < -1.0:
                return "light_tailed"
            else:
                return "irregular"
        except:
            return "unknown"

    def _analyze_correlation_strength(self, df: pd.DataFrame) -> Dict:
        """
        Analyze correlation strength between numeric columns.
        """
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[1] < 2:
            return {
                "has_strong_correlations": False,
                "max_correlation": 0,
                "correlation_pairs": [],
            }

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr_matrix = numeric_df.corr().abs()
        except Exception as e:
            self._log(f"Correlation computation failed: {e}")
            return {
                "has_strong_correlations": False,
                "max_correlation": 0,
                "correlation_pairs": [],
            }

        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        upper_triangle = corr_matrix.where(mask)

        strong_pairs = []
        for col1 in upper_triangle.columns:
            for col2 in upper_triangle.index:
                corr_val = upper_triangle.loc[col2, col1]
                if pd.notna(corr_val) and corr_val > 0.7:
                    strong_pairs.append(
                        {"col1": col1, "col2": col2, "correlation": float(corr_val)}
                    )

        max_corr = 0.0
        if not upper_triangle.empty:
            max_val = upper_triangle.max().max()
            if pd.notna(max_val):
                max_corr = float(max_val)

        return {
            "has_strong_correlations": len(strong_pairs) > 0,
            "max_correlation": max_corr,
            "strong_correlation_count": len(strong_pairs),
            "correlation_pairs": strong_pairs[:5],
        }

    def _analyze_pairwise_relationships(self, df: pd.DataFrame) -> List[Dict]:
        """
        Analyze relationships between column pairs.
        """
        relationships = []
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

        # Numeric-Numeric relationships
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                try:
                    corr = df[[col1, col2]].corr().iloc[0, 1]
                    if pd.isna(corr):
                        corr = 0.0
                    relationships.append(
                        {
                            "col1": col1,
                            "col2": col2,
                            "type": "numeric_numeric",
                            "correlation": float(corr),
                            "suggested_charts": ["scatter", "bubble"]
                            if abs(corr) < 0.7
                            else ["scatter", "line"],
                        }
                    )
                except Exception as e:
                    self._log(f"Failed to compute correlation for {col1}-{col2}: {e}")
                    continue

        # Categorical-Numeric relationships
        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                if df[cat_col].nunique() < 50:
                    relationships.append(
                        {
                            "col1": cat_col,
                            "col2": num_col,
                            "type": "categorical_numeric",
                            "category_count": int(df[cat_col].nunique()),
                            "suggested_charts": self._suggest_cat_num_charts(
                                df[cat_col], df[num_col]
                            ),
                        }
                    )

        for i, cat1 in enumerate(categorical_cols):
            for cat2 in categorical_cols[i + 1 :]:
                unique1 = df[cat1].nunique()
                unique2 = df[cat2].nunique()
                if unique1 <= 50 and unique2 <= 50:
                    if unique1 <= 12 and unique2 <= 12:
                        suggested_charts = ["grouped_bar", "stacked_bar"]
                    else:
                        suggested_charts = ["heatmap"]
                    relationships.append(
                        {
                            "col1": cat1,
                            "col2": cat2,
                            "type": "categorical_categorical",
                            "category_count_1": unique1,
                            "category_count_2": unique2,
                            "suggested_charts": suggested_charts,
                        }
                    )

        # Datetime-Numeric relationships (time series)
        for dt_col in datetime_cols:
            for num_col in numeric_cols:
                relationships.append(
                    {
                        "col1": dt_col,
                        "col2": num_col,
                        "type": "datetime_numeric",
                        "suggested_charts": ["line", "area"],
                    }
                )

        return relationships

    def _suggest_cat_num_charts(
        self, cat_series: pd.Series, num_series: pd.Series
    ) -> List[str]:
        """Suggest chart types for categorical-numeric pairs."""
        cardinality = cat_series.nunique()
        if cardinality <= 8:
            return ["bar", "pie", "donut"]
        elif cardinality <= 30:
            return ["bar", "boxplot", "violin"]
        else:
            return ["treemap", "boxplot"]

    def _analyze_categorical_balance(self, series: pd.Series) -> Dict:
        """
        Analyze class balance in categorical data using entropy and Gini.
        """
        value_counts = series.value_counts()
        if len(value_counts) == 0:
            return {}

        proportions = value_counts / len(series)

        # Entropy calculation
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        max_entropy = np.log2(len(value_counts)) if len(value_counts) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Gini coefficient
        sorted_counts = np.sort(value_counts.values)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (
            n + 1
        ) / n

        return {
            "entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "gini_coefficient": float(gini),
            "is_balanced": normalized_entropy > 0.7,
            "is_imbalanced": gini > 0.6,
            "dominant_category_percent": float(
                value_counts.iloc[0] / len(series) * 100
            ),
            "minority_category_percent": float(
                value_counts.iloc[-1] / len(series) * 100
            ),
        }

    def _detect_hierarchical_structure(self, df: pd.DataFrame) -> Dict:
        """
        Detect hierarchical relationships between categorical columns.
        """
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if len(cat_cols) < 2:
            return {"has_hierarchy": False}

        hierarchies = []
        for i, parent_col in enumerate(cat_cols):
            for child_col in cat_cols[i + 1 :]:
                try:
                    grouped = df.groupby(parent_col)[child_col].nunique()
                    strength = (grouped == 1).mean()
                    if strength > 0.7:
                        hierarchies.append(
                            {
                                "parent": parent_col,
                                "child": child_col,
                                "strength": float(strength),
                            }
                        )
                except Exception as e:
                    self._log(
                        f"Hierarchy detection failed for {parent_col}-{child_col}: {e}"
                    )
                    continue

        return {
            "has_hierarchy": len(hierarchies) > 0,
            "hierarchical_pairs": hierarchies,
        }

    def _profile_numeric(self, s: pd.Series, total_rows: int) -> Dict:
        """Profile numeric column."""
        try:
            desc = s.describe()
            profile = {
                "dtype": "numeric",
                "mean": float(desc["mean"]),
                "std": float(desc["std"]),
                "min": float(desc["min"]),
                "max": float(desc["max"]),
                "median": float(s.median()),
                "percentile_5": float(s.quantile(0.05)),
                "percentile_95": float(s.quantile(0.95)),
                "sample_values": self._sample_values(s),
                "has_negatives": bool((s < 0).any()),
                "has_zeros": bool((s == 0).any()),
            }
        except Exception as e:
            self._log(f"Numeric profiling failed: {e}")
            profile = {"dtype": "numeric", "sample_values": self._sample_values(s)}

        profile["distribution"] = self._analyze_distribution(s)
        profile["cardinality"] = self._analyze_cardinality(s, total_rows)
        profile["intent"] = self._detect_numeric_intent(s)
        return profile

    def _profile_categorical(self, s: pd.Series, total_rows: int) -> Dict:
        """Profile categorical column with special handling for booleans."""
        if pd.api.types.is_bool_dtype(s):
            unique_vals = s.dropna().unique()
            n_unique = len(unique_vals)

            value_counts = s.value_counts(normalize=True, sort=False)

            # Handle cases where only one boolean value exists
            top_values = {}
            if False in unique_vals:
                top_values["False"] = float(value_counts.get(False, 0.0))
            if True in unique_vals:
                top_values["True"] = float(value_counts.get(True, 0.0))

            is_ordinal = False
            unique_count = n_unique
            profile = {"dtype": "categorical", "is_boolean": True}
        else:
            counts = s.value_counts(normalize=True).head(10)
            top_values = {str(k): float(v) for k, v in counts.items()}
            is_ordinal = self._detect_ordinal(s)
            unique_count = int(s.nunique())
            profile = {"dtype": "categorical"}

        profile.update(
            {
                "unique_values": unique_count,
                "top_values": top_values,
                "sample_values": self._sample_values(s),
                "is_ordinal": is_ordinal,
            }
        )

        profile["cardinality"] = self._analyze_cardinality(s, total_rows)
        profile["balance"] = self._analyze_categorical_balance(s)

        profile["ordinal_statistical"] = False
        return profile

    def _detect_ordinal_relationships(
        self, df: pd.DataFrame, columns: List[Dict]
    ) -> None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        for col_meta in columns:
            if col_meta["dtype"] != "categorical":
                continue

            best_strength = 0.0

            for num_col in numeric_cols:
                num_meta = next((m for m in columns if m["name"] == num_col), None)

                # Skip ID-like numerics
                if (
                    not num_meta
                    or num_meta.get("cardinality", {}).get("unique_ratio", 1) > 0.95
                ):
                    continue

                try:
                    strength = self._detect_ordinal_statistical(
                        df[col_meta["name"]], df[num_col]
                    )
                    best_strength = max(best_strength, strength)
                except Exception:
                    continue

            col_meta["ordinal_strength"] = round(best_strength, 3)
            col_meta["ordinal_statistical"] = best_strength >= 0.7

    def _profile_datetime(self, s: pd.Series, total_rows: int) -> Dict:
        """Profile datetime column."""
        try:
            profile = {
                "dtype": "datetime",
                "min": str(s.min()),
                "max": str(s.max()),
                "sample_values": self._sample_values(s),
            }
        except Exception as e:
            self._log(f"Datetime profiling failed: {e}")
            profile = {"dtype": "datetime", "sample_values": self._sample_values(s)}

        profile["time_series"] = self._detect_time_series(s)
        return profile

    def _profile_text(self, s: pd.Series, total_rows: int) -> Dict:
        """Profile text column."""
        try:
            lengths = s.astype(str).str.len()
            profile = {
                "dtype": "text",
                "avg_length": float(lengths.mean()),
                "median_length": float(lengths.median()),
                "min_length": float(lengths.min()),
                "max_length": float(lengths.max()),
                "percent_long_text": float((lengths > 300).mean()),
                "sample_values": self._sample_values(s),
            }
        except Exception as e:
            self._log(f"Text profiling failed: {e}")
            profile = {"dtype": "text", "sample_values": self._sample_values(s)}

        profile["cardinality"] = self._analyze_cardinality(s, total_rows)
        return profile

    def _detect_flags(self, series: pd.Series, total_rows: int) -> Dict:
        """Detect special semantic types in series."""
        return {
            "is_url": self._is_probably_url(series),
            "is_email": self._is_probably_email(series),
            "is_image_url": self._is_probably_image(series),
            "is_hash_id": self._is_probably_hash(series),
            "is_long_text": self._is_long_text(series),
            "is_id_like": self._is_id_like(series, total_rows),
            "is_currency": self._is_probably_currency(series),
            "is_percentage": self._is_probably_percentage(series),
            "is_geospatial": self._is_probably_geospatial(series),
        }

    def _detect_numeric_intent(self, series: pd.Series) -> Dict:
        clean = series.dropna()
        if len(clean) < 10:
            return {}

        is_integer = (clean % 1 == 0).mean() > 0.95
        is_non_negative = (clean >= 0).mean() > 0.95

        return {
            "is_count_like": is_integer and is_non_negative and clean.max() < 1e6,
            "is_rate_like": clean.between(0, 1).mean() > 0.9,
            "is_percentage_like": clean.between(0, 100).mean() > 0.9,
            "is_cumulative_like": is_non_negative and clean.is_monotonic_increasing,
        }

    def _representative_rows(self, df: pd.DataFrame) -> List[Dict]:
        """
        Extract representative rows using KMeans clustering on numeric features.
        """
        numeric_df = df.select_dtypes(include=["number"]).dropna()

        if numeric_df.empty:
            n = min(self.represent_k, len(df))
            return df.sample(n, random_state=0).to_dict(orient="records")

        # Limit rows for KMeans performance
        if len(numeric_df) > self.max_kmeans_rows:
            numeric_df = numeric_df.sample(self.max_kmeans_rows, random_state=0)

        k = min(self.represent_k, len(numeric_df))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10, max_iter=300)
                kmeans.fit(numeric_df)

            rows = []
            for c in kmeans.cluster_centers_:
                idx = ((numeric_df - c) ** 2).sum(axis=1).idxmin()
                rows.append(df.loc[idx].to_dict())
            return rows
        except Exception as e:
            self._log(f"KMeans clustering failed: {e}")
            return df.head(self.represent_k).to_dict(orient="records")

    def _has_non_trivial_numeric(self, columns: List[Dict]) -> bool:
        """
        Check for numeric columns with meaningful variation.

        Filters out constant, ID-like, or trivial numeric columns.
        """
        for col in columns:
            if col.get("dtype") != "numeric":
                continue

            card = col.get("cardinality", {})
            unique_ratio = card.get("unique_ratio", 1.0)
            dist = col.get("distribution", {})

            # Skip near-constant
            if unique_ratio < 0.005:
                continue

            # Skip ID-like (but allow low cardinality like ratings)
            if unique_ratio > self.id_uniqueness_threshold:
                continue

            # Skip if dominated by zeros
            zero_ratio = dist.get("zero_ratio", 0)
            if zero_ratio > 0.9:
                continue

            # Require some spread
            cv = dist.get("coefficient_of_variation", 0)
            if cv < 0.001:
                continue

            return True
        return False

    def _generate_chart_hints(self, profile_data: Dict) -> Dict:
        return self._generate_decisive_chart_recommendation(profile_data)

    def _generate_decisive_chart_recommendation(self, profile_data: Dict) -> Dict:
        """
        Generate decisive chart recommendations with clear best choices.

        Returns one primary recommendation per valid column combination,
        with optional fallback alternatives ranked by suitability.
        """
        columns = profile_data["columns"]
        total_rows = profile_data["dataset_overview"]["rows"]
        avg_missing = (
            profile_data["dataset_overview"]["missing_overall_percent"] / 100.0
        )

        # Extract column metadata
        column_metadata = self._extract_column_metadata(columns)

        recommendations = {
            "primary_suggestions": [],  # Best chart for each column combo
            "fallback_suggestions": [],  # Alternatives if primary fails
            "unsuitable_combinations": [],  # Combos we can't visualize
            "reasoning": {},  # Explanation for each recommendation
        }

        # === 1. UNIVARIATE RECOMMENDATIONS ===
        for col_meta in column_metadata:
            result = self._recommend_univariate_chart(col_meta, total_rows, avg_missing)
            if result:
                recommendations["primary_suggestions"].append(result)
                recommendations["reasoning"][result["chart"]] = result["reason"]

        # === 2. BIVARIATE RECOMMENDATIONS ===
        bivariate_results = self._recommend_bivariate_charts(
            column_metadata, profile_data, total_rows, avg_missing
        )
        recommendations["primary_suggestions"].extend(bivariate_results["primary"])
        recommendations["fallback_suggestions"].extend(bivariate_results["fallback"])
        recommendations["unsuitable_combinations"].extend(
            bivariate_results["unsuitable"]
        )
        recommendations["reasoning"].update(bivariate_results["reasoning"])

        # === 3. MULTIVARIATE RECOMMENDATIONS (3+ columns) ===
        if len(column_metadata) >= 3:
            multivariate_results = self._recommend_multivariate_charts(
                column_metadata, profile_data, total_rows, avg_missing
            )
            recommendations["primary_suggestions"].extend(multivariate_results)

        return recommendations

    def _extract_column_metadata(self, columns: List[Dict]) -> List[Dict]:
        """Extract simplified metadata for decision-making."""
        metadata = []
        for col in columns:
            meta = {
                "name": col["name"],
                "dtype": col["dtype"],
                "missing_pct": col.get("missing_percent", 0),
                "cardinality": col.get("cardinality", {}),
                "flags": col.get("flags", {}),
                "distribution": col.get("distribution", {}),
                "balance": col.get("balance", {}),
                "time_series": col.get("time_series", {}),
            }
            metadata.append(meta)
        return metadata

    def _recommend_univariate_chart(
        self, col_meta: Dict, total_rows: int, avg_missing: float
    ) -> Optional[Dict]:
        """
        Decisive univariate chart recommendation.

        Decision tree:
        - Numeric → histogram (if non-trivial) or boxplot
        - Categorical → pie (low card) or bar (medium card)
        - Datetime → timeline
        - Text → word cloud (if suitable)
        """
        name = col_meta["name"]
        dtype = col_meta["dtype"]
        missing_pct = col_meta["missing_pct"]

        # Quality check
        if missing_pct > 80:
            return None  # Too sparse

        if avg_missing > 0.5:
            return None  # Overall data too sparse

        # === NUMERIC COLUMNS ===
        if dtype == "numeric":
            # Check if trivial
            unique_ratio = col_meta["cardinality"].get("unique_ratio", 1.0)
            zero_ratio = col_meta["distribution"].get("zero_ratio", 0)
            cv = col_meta["distribution"].get("coefficient_of_variation", 0)

            # Skip ID-like or constant columns
            if unique_ratio > 0.95 or unique_ratio < 0.005:
                return None

            # Skip zero-dominated
            if zero_ratio > 0.9:
                return None

            # Skip near-constant
            if cv < 0.001:
                return None

            # DECISION: histogram is best for distribution
            return {
                "chart": "histogram",
                "columns": [name],
                "reason": f"Distribution analysis of numeric variable (CV={cv:.2f})",
                "confidence": "high",
                "priority": 8,
                "chart_family": "univariate_numeric",
            }

        # === CATEGORICAL COLUMNS ===
        elif dtype == "categorical":
            unique_count = col_meta["cardinality"].get("unique_count", 0)
            is_balanced = col_meta["balance"].get("is_balanced", False)

            # Too many categories
            if unique_count > 50:
                return None

            # DECISION TREE by cardinality
            if unique_count == 1:
                return None  # Constant column

            elif 2 <= unique_count <= 8:
                # Low cardinality → PIE is best for proportions
                return {
                    "chart": "pie",
                    "columns": [name],
                    "reason": f"Low-cardinality categorical ({unique_count} categories) - ideal for proportions",
                    "confidence": "high",
                    "priority": 7,
                    "chart_family": "univariate_categorical",
                }

            elif 9 <= unique_count <= 20:
                # Medium cardinality → BAR is better
                return {
                    "chart": "bar",
                    "columns": [name],
                    "reason": f"Medium-cardinality categorical ({unique_count} categories) - bar chart for comparison",
                    "confidence": "high",
                    "priority": 7,
                    "chart_family": "univariate_categorical",
                }

            else:  # 21-50
                # High cardinality → BAR with warning
                return {
                    "chart": "bar",
                    "columns": [name],
                    "reason": f"High-cardinality categorical ({unique_count} categories) - consider filtering top N",
                    "confidence": "medium",
                    "priority": 5,
                    "chart_family": "univariate_categorical",
                }

        # === DATETIME COLUMNS ===
        elif dtype == "datetime":
            if total_rows < 5:
                return None

            return {
                "chart": "timeline",
                "columns": [name],
                "reason": "Temporal sequence visualization",
                "confidence": "high",
                "priority": 6,
                "chart_family": "univariate_datetime",
            }

        return None

    def _recommend_bivariate_charts(
        self,
        column_metadata: List[Dict],
        profile_data: Dict,
        total_rows: int,
        avg_missing: float,
    ) -> Dict:
        """
        Decisive bivariate chart recommendations.

        Decision matrix:
        - Numeric × Numeric → scatter (correlation check) or line (if sequential)
        - Categorical × Numeric → bar (low cat) or boxplot (high cat)
        - Categorical × Categorical → grouped_bar or heatmap
        - Datetime × Numeric → line (time series)
        """
        primary = []
        fallback = []
        unsuitable = []
        reasoning = {}

        # Get correlation info
        corr_analysis = profile_data.get("correlation_analysis", {})
        strong_corrs = {
            (p["col1"], p["col2"]): p["correlation"]
            for p in corr_analysis.get("correlation_pairs", [])
        }

        # Pairwise evaluation
        for i, col1_meta in enumerate(column_metadata):
            for col2_meta in column_metadata[i + 1 :]:
                col1_name = col1_meta["name"]
                col2_name = col2_meta["name"]
                dtype1 = col1_meta["dtype"]
                dtype2 = col2_meta["dtype"]

                # Quality check
                if col1_meta["missing_pct"] > 70 or col2_meta["missing_pct"] > 70:
                    unsuitable.append(
                        {
                            "columns": [col1_name, col2_name],
                            "reason": "Too much missing data",
                        }
                    )
                    continue

                # === NUMERIC × NUMERIC ===
                if dtype1 == "numeric" and dtype2 == "numeric":
                    result = self._decide_numeric_numeric(
                        col1_name,
                        col2_name,
                        col1_meta,
                        col2_meta,
                        strong_corrs,
                        total_rows,
                    )
                    if result:
                        primary.append(result["primary"])
                        reasoning[result["primary"]["chart"]] = result["primary"][
                            "reason"
                        ]
                        if "fallback" in result:
                            fallback.append(result["fallback"])

                # === CATEGORICAL × NUMERIC ===
                elif (dtype1 == "categorical" and dtype2 == "numeric") or (
                    dtype1 == "numeric" and dtype2 == "categorical"
                ):
                    cat_meta = col1_meta if dtype1 == "categorical" else col2_meta
                    num_meta = col2_meta if dtype1 == "categorical" else col1_meta

                    result = self._decide_categorical_numeric(
                        cat_meta, num_meta, total_rows
                    )
                    if result:
                        primary.append(result)
                        reasoning[result["chart"]] = result["reason"]

                # === DATETIME × NUMERIC ===
                elif (dtype1 == "datetime" and dtype2 == "numeric") or (
                    dtype1 == "numeric" and dtype2 == "datetime"
                ):
                    dt_meta = col1_meta if dtype1 == "datetime" else col2_meta
                    num_meta = col2_meta if dtype1 == "datetime" else col1_meta

                    result = self._decide_datetime_numeric(
                        dt_meta, num_meta, total_rows
                    )
                    if result:
                        primary.append(result)
                        reasoning[result["chart"]] = result["reason"]

                # === CATEGORICAL × CATEGORICAL ===
                elif dtype1 == "categorical" and dtype2 == "categorical":
                    result = self._decide_categorical_categorical(
                        col1_meta, col2_meta, total_rows
                    )
                    if result:
                        if isinstance(result, dict) and "primary" in result:
                            primary.append(result["primary"])
                            reasoning[result["primary"]["chart"]] = result["primary"][
                                "reason"
                            ]
                            if "fallback" in result:
                                fallback.append(result["fallback"])
                        else:
                            primary.append(result)
                            reasoning[result["chart"]] = result["reason"]

                # === UNSUITABLE COMBINATIONS ===
                else:
                    unsuitable.append(
                        {
                            "columns": [col1_name, col2_name],
                            "reason": f"Unsupported type combination: {dtype1} × {dtype2}",
                        }
                    )

        return {
            "primary": primary,
            "fallback": fallback,
            "unsuitable": unsuitable,
            "reasoning": reasoning,
        }

    def _decide_numeric_numeric(
        self,
        col1_name: str,
        col2_name: str,
        col1_meta: Dict,
        col2_meta: Dict,
        strong_corrs: Dict,
        total_rows: int,
    ) -> Optional[Dict]:
        """
        Decisive chart for two numeric columns.

        Decision logic:
        1. Check if strongly correlated → scatter (show relationship)
        2. Check if one is ID-like → skip
        3. Check if sequential → line
        4. Default → scatter
        """
        if total_rows < 10:
            return None

        # Check for ID columns
        if (
            col1_meta["cardinality"].get("unique_ratio", 0) > 0.95
            or col2_meta["cardinality"].get("unique_ratio", 0) > 0.95
        ):
            return None  # One is an ID

        # Check correlation
        corr_key = (col1_name, col2_name)
        corr_val = strong_corrs.get(
            corr_key, strong_corrs.get((col2_name, col1_name), 0)
        )

        if abs(corr_val) > 0.7:
            # Strong correlation → SCATTER is best
            return {
                "primary": {
                    "chart": "scatter",
                    "columns": [col1_name, col2_name],
                    "reason": f"Strong correlation detected (r={corr_val:.2f}) - relationship analysis",
                    "confidence": "high",
                    "priority": 10,
                    "metadata": {"correlation": corr_val},
                    "chart_family": "numeric_numeric",
                },
                "fallback": {
                    "chart": "line",
                    "columns": [col1_name, col2_name],
                    "reason": "Alternative: line chart for trend",
                    "confidence": "medium",
                    "priority": 7,
                    "chart_family": "numeric_numeric",
                },
            }

        # Check if data looks sequential (one column might be index-like)
        col1_is_sequential = self._looks_sequential(col1_meta)
        col2_is_sequential = self._looks_sequential(col2_meta)

        if col1_is_sequential or col2_is_sequential:
            # Sequential data → LINE is better
            x_col = col1_name if col1_is_sequential else col2_name
            y_col = col2_name if col1_is_sequential else col1_name
            return {
                "primary": {
                    "chart": "line",
                    "columns": [x_col, y_col],
                    "reason": "Sequential data detected - line chart for trends",
                    "confidence": "high",
                    "priority": 8,
                    "chart_family": "numeric_numeric",
                }
            }

        # Default: scatter for general numeric pairs
        return {
            "primary": {
                "chart": "scatter",
                "columns": [col1_name, col2_name],
                "reason": "General numeric relationship - scatter for pattern discovery",
                "confidence": "medium",
                "priority": 6,
                "chart_family": "numeric_numeric",
            }
        }

    def _decide_categorical_numeric(
        self, cat_meta: Dict, num_meta: Dict, total_rows: int
    ) -> Optional[Dict]:
        """
        Decisive chart for categorical × numeric.

        Decision logic:
        1. Low cardinality (≤8) → bar chart
        2. Medium cardinality (9-20) → grouped_bar or boxplot
        3. High cardinality (21-50) → boxplot (better for many groups)
        """
        if total_rows < 10:
            return None

        cat_name = cat_meta["name"]
        num_name = num_meta["name"]
        unique_count = cat_meta["cardinality"].get("unique_count", 0)

        # Check if numeric is trivial
        if num_meta["cardinality"].get("unique_ratio", 0) > 0.95:
            return None  # Numeric is ID-like

        ordinal_strength = cat_meta.get("ordinal_strength", 0.0)

        # DECISION TREE by category count
        if unique_count <= 8:
            if ordinal_strength >= 0.7:
                confidence = self._ordinal_confidence(ordinal_strength)

                return {
                    "chart": "line",
                    "columns": [cat_name, num_name],
                    "reason": f"Ordinal categorical detected (strength={ordinal_strength})",
                    "confidence": confidence,
                    "priority": 9,
                    "chart_family": "categorical_numeric",
                }

            return {
                "chart": "bar",
                "columns": [cat_name, num_name],
                "reason": f"Category-value comparison ({unique_count} categories) - bar chart optimal",
                "confidence": "high",
                "priority": 9,
                "chart_family": "categorical_numeric",
            }

        elif 9 <= unique_count <= 20:
            if ordinal_strength >= 0.7:
                confidence = self._ordinal_confidence(ordinal_strength)

                return {
                    "chart": "line",
                    "columns": [cat_name, num_name],
                    "reason": f"Ordinal categorical detected (strength={ordinal_strength})",
                    "confidence": confidence,
                    "priority": 9,
                    "chart_family": "categorical_numeric",
                }

            return {
                "chart": "boxplot",
                "columns": [cat_name, num_name],
                "reason": f"Distribution across {unique_count} categories - boxplot shows spread",
                "confidence": "high",
                "priority": 8,
                "chart_family": "categorical_numeric",
            }

        elif 21 <= unique_count <= 50:
            # High cardinality → BOXPLOT or treemap
            has_outliers = num_meta["distribution"].get("has_outliers", False)
            if ordinal_strength >= 0.7:
                confidence = self._ordinal_confidence(ordinal_strength)
                return {
                    "chart": "line",
                    "columns": [cat_name, num_name],
                    "reason": f"Ordinal categorical detected (strength={ordinal_strength})",
                    "confidence": confidence,
                    "priority": 9,
                    "chart_family": "categorical_numeric",
                }
            elif has_outliers:
                return {
                    "chart": "boxplot",
                    "columns": [cat_name, num_name],
                    "reason": f"Many categories ({unique_count}) with outliers - boxplot essential",
                    "confidence": "medium",
                    "priority": 7,
                    "chart_family": "categorical_numeric",
                }
            else:
                return {
                    "chart": "treemap",
                    "columns": [cat_name, num_name],
                    "reason": f"High-cardinality hierarchy ({unique_count}) - treemap for proportions",
                    "confidence": "medium",
                    "priority": 6,
                    "chart_family": "categorical_numeric",
                }

        return None

    def _ordinal_confidence(self, strength: float) -> str:
        if strength >= 0.85:
            return "very_high"
        if strength >= 0.75:
            return "high"
        return "medium"

    def _decide_datetime_numeric(
        self, dt_meta: Dict, num_meta: Dict, total_rows: int
    ) -> Optional[Dict]:
        """
        Decisive chart for datetime × numeric.

        Decision: LINE is almost always best for time series.
        Use AREA if cumulative pattern detected.
        """
        if total_rows < 5:
            return None

        dt_name = dt_meta["name"]
        num_name = num_meta["name"]

        # Check if numeric is trivial
        if num_meta["cardinality"].get("unique_ratio", 0) > 0.95:
            return None

        # Check for gaps in time series
        ts_info = dt_meta.get("time_series", {})
        has_gaps = ts_info.get("has_gaps", False)
        is_sequential = ts_info.get("is_sequential", False)

        # Check if values are always positive (cumulative indicator)
        intent = num_meta.get("intent", {})
        is_cumulative = intent.get("is_cumulative_like", False)

        if is_cumulative:
            return {
                "chart": "area",
                "columns": [dt_name, num_name],
                "reason": "Time series with cumulative pattern - area chart emphasizes total",
                "confidence": "high",
                "priority": 9,
                "chart_family": "datetime_numeric",
            }

        # Default: LINE for time series
        confidence = "high" if is_sequential else "medium"
        return {
            "chart": "line",
            "columns": [dt_name, num_name],
            "reason": f"Time series analysis - line chart for trends {'(with gaps)' if has_gaps else ''}",
            "confidence": confidence,
            "priority": 9 if is_sequential else 7,
            "chart_family": "datetime_numeric",
        }

    def _decide_categorical_categorical(
        self, cat1_meta: Dict, cat2_meta: Dict, total_rows: int
    ) -> Optional[Dict]:
        """
        Decisive chart for categorical × categorical.

        Decision logic:
        1. Both low cardinality → grouped_bar
        2. One high cardinality → heatmap
        """
        if total_rows < 10:
            return None

        cat1_name = cat1_meta["name"]
        cat2_name = cat2_meta["name"]
        unique1 = cat1_meta["cardinality"].get("unique_count", 0)
        unique2 = cat2_meta["cardinality"].get("unique_count", 0)

        # Both low cardinality
        if unique1 <= 12 and unique2 <= 12:
            return {
                "primary": {
                    "chart": "grouped_bar",
                    "columns": [cat1_name, cat2_name],
                    "reason": f"Two categorical variables ({unique1}×{unique2}) - grouped comparison",
                    "confidence": "high",
                    "priority": 7,
                    "chart_family": "categorical_categorical",
                },
                "fallback": {
                    "chart": "stacked_bar",
                    "columns": [cat1_name, cat2_name],
                    "reason": f"Two categorical variables ({unique1}×{unique2}) - grouped comparison",
                    "confidence": "high",
                    "priority": 7,
                    "chart_family": "categorical_categorical",
                },
            }
        # At least one medium/high cardinality
        elif unique1 <= 20 and unique2 <= 20:
            return {
                "chart": "heatmap",
                "columns": [cat1_name, cat2_name],
                "reason": f"Categorical contingency ({unique1}×{unique2}) - heatmap for patterns",
                "confidence": "medium",
                "priority": 6,
                "chart_family": "categorical_categorical",
            }

        return None  # Too many combinations

    def _recommend_multivariate_charts(
        self,
        column_metadata: List[Dict],
        profile_data: Dict,
        total_rows: int,
        avg_missing: float,
    ) -> List[Dict]:
        """
        Recommendations for 3+ column combinations.

        Only suggest if very clear use case (e.g., 3 numeric for bubble).
        """
        recommendations = []

        numeric_cols = [c for c in column_metadata if c["dtype"] == "numeric"]

        # BUBBLE chart for 3 numeric columns
        if len(numeric_cols) >= 3 and total_rows >= 20:
            # Pick first 3 non-ID numeric columns
            suitable_numeric = [
                c
                for c in numeric_cols
                if c["cardinality"].get("unique_ratio", 0) < 0.95
            ][:3]

            if len(suitable_numeric) == 3:
                recommendations.append(
                    {
                        "chart": "bubble",
                        "columns": [c["name"] for c in suitable_numeric],
                        "reason": "Three numeric dimensions - bubble chart for 3D relationships",
                        "confidence": "medium",
                        "priority": 6,
                        "chart_family": "numeric_numeric",
                    }
                )

        # HEATMAP for correlation matrix (4+ numeric)
        if len(numeric_cols) >= 4 and total_rows >= 10:
            recommendations.append(
                {
                    "chart": "heatmap",
                    "columns": [c["name"] for c in numeric_cols],
                    "reason": f"Correlation matrix for {len(numeric_cols)} numeric variables",
                    "confidence": "high",
                    "priority": 7,
                    "chart_family": "categorical_categorical",
                }
            )

        return recommendations

    def _looks_sequential(self, col_meta: Dict) -> bool:
        """Check if numeric column looks like sequential data (index, time steps)."""
        unique_ratio = col_meta["cardinality"].get("unique_ratio", 0)
        cv = col_meta["distribution"].get("coefficient_of_variation", 0)

        return unique_ratio > 0.9 and cv < 0.5

    def _detect_boolean_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect boolean-like columns."""
        bool_cols = []
        for col in df.columns:
            s = df[col]
            if s.nunique() == 2:
                unique_vals = set(s.dropna().astype(str).str.lower().unique())
                bool_values = {
                    "true",
                    "false",
                    "yes",
                    "no",
                    "1",
                    "0",
                    "t",
                    "f",
                    "y",
                    "n",
                    "1.0",
                    "0.0",
                }
                if unique_vals.issubset(bool_values):
                    bool_cols.append(col)
        return bool_cols

    def get_column_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get a summary of column types in the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to analyze

        Returns
        -------
        Dict
            Dictionary mapping type names to lists of column names
        """
        profile = self.profile(df)
        return {
            "numeric": [
                c["name"] for c in profile["columns"] if c["dtype"] == "numeric"
            ],
            "categorical": [
                c["name"] for c in profile["columns"] if c["dtype"] == "categorical"
            ],
            "datetime": [
                c["name"] for c in profile["columns"] if c["dtype"] == "datetime"
            ],
            "text": [c["name"] for c in profile["columns"] if c["dtype"] == "text"],
            "boolean": self._detect_boolean_columns(df),
        }

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            s = df[col]
            if s.dtype == "object":
                # Try datetime
                try:
                    parsed_dt = pd.to_datetime(s, errors="coerce")
                    if parsed_dt.notna().mean() > 0.8:
                        df[col] = parsed_dt
                        continue
                except:
                    pass
                # Try numeric
                try:
                    parsed_num = pd.to_numeric(s, errors="coerce")
                    if parsed_num.notna().mean() > 0.8:
                        df[col] = parsed_num
                        continue
                except:
                    pass
        return df

    def profile(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive profile of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to profile

        Returns
        -------
        Dict
            Complete profile including dataset overview, column statistics,
            correlations, relationships, and chart recommendations
        """
        processed_df = self._preprocess_dataframe(df)

        fingerprint = self._df_fingerprint(processed_df)
        if fingerprint in self._profile_cache:
            return self._profile_cache[fingerprint]

        total_rows = len(processed_df)

        result = {
            "dataset_overview": {
                "rows": total_rows,
                "columns": len(processed_df.columns),
                "memory_mb": round(
                    processed_df.memory_usage(deep=True).sum() / 1_000_000, 3
                ),
                "duplicate_row_percent": float(processed_df.duplicated().mean() * 100),
                "missing_overall_percent": float(
                    processed_df.isna().mean().mean() * 100
                ),
                "is_wide": len(processed_df.columns) > total_rows,
                "density": float(total_rows * len(processed_df.columns)),
            },
            "column_type_summary": {
                "numeric_count": 0,
                "categorical_count": 0,
                "datetime_count": 0,
                "text_count": 0,
            },
            "columns": [],
            "correlations": [],
            "correlation_analysis": {},
            "hierarchical_analysis": {},
            "pairwise_relationships": [],
            "representative_rows": [],
            "chart_hints": {},
        }

        for col in processed_df.columns:
            s = processed_df[col]
            if s.dtype == "object":
                try:
                    parsed_dt = pd.to_datetime(s, errors="coerce")
                    if parsed_dt.notna().mean() > 0.8:
                        processed_df[col] = parsed_dt
                        continue
                except:
                    pass

                try:
                    parsed_num = pd.to_numeric(s, errors="coerce")
                    if parsed_num.notna().mean() > 0.8:
                        processed_df[col] = parsed_num
                        continue
                except:
                    pass

        # Profile each column
        for col in processed_df.columns:
            s = processed_df[col]
            info = {
                "name": col,
                "missing_percent": float(s.isna().mean() * 100),
            }

            if pd.api.types.is_bool_dtype(s):
                info.update(self._profile_categorical(s, total_rows))
                result["column_type_summary"]["categorical_count"] += 1
            elif pd.api.types.is_numeric_dtype(s):
                info.update(self._profile_numeric(s, total_rows))
                result["column_type_summary"]["numeric_count"] += 1
            elif pd.api.types.is_datetime64_any_dtype(s):
                info.update(self._profile_datetime(s, total_rows))
                result["column_type_summary"]["datetime_count"] += 1
            else:
                if s.nunique() < 50:
                    info.update(self._profile_categorical(s.astype(str), total_rows))
                    result["column_type_summary"]["categorical_count"] += 1
                else:
                    info.update(self._profile_text(s.astype(str), total_rows))
                    result["column_type_summary"]["text_count"] += 1

            info["flags"] = self._detect_flags(s, total_rows)
            result["columns"].append(info)

        # Correlation matrix (limited to prevent performance issues)
        numeric_processed_df = processed_df.select_dtypes(include=["number"])
        if 2 <= numeric_processed_df.shape[1] <= self.max_corr_columns:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr = numeric_processed_df.corr().stack().reset_index()
                    corr.columns = ["col1", "col2", "correlation"]
                    corr = corr[corr["col1"] < corr["col2"]]
                    result["correlations"] = corr.to_dict(orient="records")
            except Exception as e:
                self._log(f"Correlation matrix computation failed: {e}")
                result["correlations"] = []

        self._detect_ordinal_relationships(processed_df, result["columns"])

        # Advanced analyses
        result["correlation_analysis"] = self._analyze_correlation_strength(
            processed_df
        )
        result["hierarchical_analysis"] = self._detect_hierarchical_structure(
            processed_df
        )
        result["pairwise_relationships"] = self._analyze_pairwise_relationships(
            processed_df
        )
        result["representative_rows"] = self._representative_rows(processed_df)
        result["chart_hints"] = self._generate_chart_hints(result)
        result["profile_rejected_charts"] = self.build_rejected_charts(
            result["chart_hints"]
        )

        # Add univariate chart suggestions to each column
        for col_profile in result["columns"]:
            col_name = col_profile["name"]
            dtype = col_profile["dtype"]
            charts = []

            if dtype == "numeric":
                if self._has_non_trivial_numeric([col_profile]):
                    charts.extend(["histogram", "boxplot"])
            elif dtype == "categorical":
                card = col_profile.get("cardinality", {})
                if card.get("is_suitable_for_pie"):
                    charts.extend(["pie", "bar"])
                elif card.get("is_suitable_for_bar"):
                    charts.append("bar")

            col_profile["suggested_univariate_charts"] = charts

        self._profile_cache[fingerprint] = result
        return result

    def build_rejected_charts(self, chart_hints):
        """
        Build meaningful rejected chart candidates based on
        chart families that were actually considered.
        """
        rejected = []

        hints = chart_hints.get("primary_suggestion", {})

        for hint in hints:
            print(hint)
            family = hint.get("chart_family")
            chosen_chart = hint.get("chart")
            columns = hint.get("columns")

            if not family or family not in CHART_FAMILIES:
                continue

            for candidate in CHART_FAMILIES[family]:
                if candidate == chosen_chart:
                    continue

                rejected.append(
                    {
                        "chart": candidate,
                        "columns": columns,
                        "rejected_because": f"'{chosen_chart}' selected as better fit",
                        "chart_family": family,
                        "rejected_by": "heuristic",
                    }
                )

        return rejected

    def _calibrate_confidence(
        self, base_confidence: str, model_score: float | None = None
    ) -> str:
        if model_score is None:
            return base_confidence

        if model_score > 0.85:
            return "very_high"
        elif model_score > 0.7:
            return "high"
        elif model_score > 0.5:
            return "medium"
        return "low"

    def get_visualizable_columns(self, df: pd.DataFrame) -> Dict:
        """
        Identify which columns are suitable for visualization.

        Filters out ID columns, URLs, hashes, and other non-visualizable types.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to analyze

        Returns
        -------
        Dict
            Dictionary with 'visualizable_columns' list and 'excluded_columns' dict
        """
        processed_df = self._preprocess_dataframe(df)
        total_rows = len(processed_df)
        visualizable = []
        excluded_reasons = {}

        for col in processed_df.columns:
            series = processed_df[col]

            # Check exclusion criteria
            if self._is_id_like(series, total_rows):
                excluded_reasons[col] = "ID column"
                continue
            if self._is_probably_url(series):
                excluded_reasons[col] = "URL column"
                continue
            if self._is_probably_image(series):
                excluded_reasons[col] = "Image path column"
                continue
            if self._is_probably_hash(series):
                excluded_reasons[col] = "Hash column"
                continue
            if self._is_long_text(series):
                excluded_reasons[col] = "Long text column"
                continue
            if series.isna().mean() > 0.9:
                excluded_reasons[col] = "Too many missing values (>90%)"
                continue
            if series.nunique(dropna=True) <= 1:
                excluded_reasons[col] = "Constant column"
                continue

            visualizable.append(col)

        return {
            "visualizable_columns": visualizable,
            "excluded_columns": excluded_reasons,
        }

    def extract_ml_features(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: Optional[str] = None,
        precomputed_profile: Optional[Dict] = None,
    ) -> Dict:
        """
        Extract ML features for chart recommendation model training.

        Generates 50+ features capturing dataset properties, column statistics,
        and pairwise relationships.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        col1 : str
            First column name
        col2 : str, optional
            Second column name for pairwise features
        precomputed_profile : Dict, optional
            Pre-computed profile to avoid recomputation

        Returns
        -------
        Dict
            Feature dictionary suitable for ML models
        """
        processed_df = self._preprocess_dataframe(df)
        features = {}

        profile = precomputed_profile or self.profile(processed_df)

        col1_profile = next((c for c in profile["columns"] if c["name"] == col1), None)
        col2_profile = (
            next((c for c in profile["columns"] if c["name"] == col2), None)
            if col2
            else None
        )

        if not col1_profile:
            return {}

        # Dataset-level features
        features.update(
            {
                "dataset_rows": np.log1p(profile["dataset_overview"]["rows"]),
                "dataset_cols": profile["dataset_overview"]["columns"],
                "dataset_density": np.log1p(profile["dataset_overview"]["density"]),
                "dataset_missing_pct": profile["dataset_overview"][
                    "missing_overall_percent"
                ],
                "dataset_is_wide": int(profile["dataset_overview"]["is_wide"]),
                "numeric_col_ratio": profile["column_type_summary"]["numeric_count"]
                / max(1, len(processed_df.columns)),
                "categorical_col_ratio": profile["column_type_summary"][
                    "categorical_count"
                ]
                / max(1, len(processed_df.columns)),
                "datetime_col_ratio": profile["column_type_summary"]["datetime_count"]
                / max(1, len(processed_df.columns)),
            }
        )

        # Column 1 features
        col1_dtype = col1_profile["dtype"]
        features.update(
            {
                "col1_is_numeric": int(col1_dtype == "numeric"),
                "col1_is_categorical": int(col1_dtype == "categorical"),
                "col1_is_datetime": int(col1_dtype == "datetime"),
                "col1_is_text": int(col1_dtype == "text"),
                "col1_missing_pct": col1_profile["missing_percent"],
            }
        )

        # Cardinality features
        if "cardinality" in col1_profile:
            card = col1_profile["cardinality"]
            features.update(
                {
                    "col1_unique_count": np.log1p(card["unique_count"]),
                    "col1_unique_ratio": card["unique_ratio"],
                    "col1_is_binary": int(card["cardinality_category"] == "binary"),
                    "col1_is_low_card": int(card["cardinality_category"] == "low"),
                    "col1_is_medium_card": int(
                        card["cardinality_category"] == "medium"
                    ),
                    "col1_is_high_card": int(card["cardinality_category"] == "high"),
                    "col1_suitable_pie": int(card.get("is_suitable_for_pie", False)),
                    "col1_suitable_bar": int(card.get("is_suitable_for_bar", False)),
                }
            )

        # Distribution features (numeric only)
        if col1_dtype == "numeric" and "distribution" in col1_profile:
            dist = col1_profile["distribution"]
            features.update(
                {
                    "col1_skewness": dist.get("skewness", 0),
                    "col1_kurtosis": dist.get("kurtosis", 0),
                    "col1_is_normal": int(dist.get("is_normal", False)),
                    "col1_has_outliers": int(dist.get("has_outliers", False)),
                    "col1_outlier_ratio": dist.get("outlier_ratio", 0),
                    "col1_coef_variation": dist.get("coefficient_of_variation", 0),
                    "col1_zero_ratio": dist.get("zero_ratio", 0),
                    "col1_negative_ratio": dist.get("negative_ratio", 0),
                }
            )
        else:
            features.update(
                {
                    "col1_skewness": 0,
                    "col1_kurtosis": 0,
                    "col1_is_normal": 0,
                    "col1_has_outliers": 0,
                    "col1_outlier_ratio": 0,
                    "col1_coef_variation": 0,
                    "col1_zero_ratio": 0,
                    "col1_negative_ratio": 0,
                }
            )

        # Balance features (categorical only)
        if col1_dtype == "categorical" and "balance" in col1_profile:
            balance = col1_profile["balance"]
            features.update(
                {
                    "col1_entropy": balance.get("normalized_entropy", 0),
                    "col1_gini": balance.get("gini_coefficient", 0),
                    "col1_is_balanced": int(balance.get("is_balanced", False)),
                    "col1_is_imbalanced": int(balance.get("is_imbalanced", False)),
                    "col1_dominant_pct": balance.get("dominant_category_percent", 0),
                }
            )
            features["col1_is_ordinal"] = int(col1_profile.get("is_ordinal", False))
        else:
            features.update(
                {
                    "col1_entropy": 0,
                    "col1_gini": 0,
                    "col1_is_balanced": 0,
                    "col1_is_imbalanced": 0,
                    "col1_dominant_pct": 0,
                    "col1_is_ordinal": 0,
                }
            )

        # Time series features (datetime only)
        if col1_dtype == "datetime" and "time_series" in col1_profile:
            ts = col1_profile["time_series"]
            features.update(
                {
                    "col1_is_sequential": int(ts.get("is_sequential", False)),
                    "col1_has_gaps": int(ts.get("has_gaps", False)),
                    "col1_time_span_days": np.log1p(ts.get("time_span_days", 0)),
                }
            )
        else:
            features.update(
                {
                    "col1_is_sequential": 0,
                    "col1_has_gaps": 0,
                    "col1_time_span_days": 0,
                }
            )

        # Semantic flags
        flags = col1_profile.get("flags", {})
        features.update(
            {
                "col1_is_currency": int(flags.get("is_currency", False)),
                "col1_is_percentage": int(flags.get("is_percentage", False)),
                "col1_is_geospatial": int(flags.get("is_geospatial", False)),
            }
        )

        # Column 2 features (if provided)
        if col2_profile:
            col2_dtype = col2_profile["dtype"]
            features.update(
                {
                    "col2_is_numeric": int(col2_dtype == "numeric"),
                    "col2_is_categorical": int(col2_dtype == "categorical"),
                    "col2_is_datetime": int(col2_dtype == "datetime"),
                    "col2_is_text": int(col2_dtype == "text"),
                    "col2_missing_pct": col2_profile["missing_percent"],
                }
            )

            if "cardinality" in col2_profile:
                card = col2_profile["cardinality"]
                features.update(
                    {
                        "col2_unique_count": np.log1p(card["unique_count"]),
                        "col2_unique_ratio": card["unique_ratio"],
                    }
                )

            features.update(
                self._extract_pair_features(
                    processed_df, col1, col2, col1_profile, col2_profile
                )
            )
        else:
            features.update(
                {
                    "col2_is_numeric": 0,
                    "col2_is_categorical": 0,
                    "col2_is_datetime": 0,
                    "col2_is_text": 0,
                    "col2_missing_pct": 0,
                    "col2_unique_count": 0,
                    "col2_unique_ratio": 0,
                    "has_col2": 0,
                    "pair_correlation": 0,
                    "pair_anova_f": 0,
                    "pair_chi2": 0,
                }
            )

        features["has_col2"] = int(col2 is not None)
        return features

    def _extract_pair_features(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str,
        col1_profile: Dict,
        col2_profile: Dict,
    ) -> Dict:
        """Extract statistical features for column pairs."""
        features = {}
        col1_dtype = col1_profile["dtype"]
        col2_dtype = col2_profile["dtype"]

        defaults = {
            "pair_correlation": 0,
            "pair_correlation_abs": 0,
            "pair_is_strongly_correlated": 0,
            "pair_type_num_num": 0,
            "pair_type_cat_num": 0,
            "pair_type_cat_cat": 0,
            "pair_type_time_num": 0,
            "pair_anova_f": 0,
            "pair_anova_p": 1,
            "pair_anova_significant": 0,
            "pair_category_count": 0,
            "pair_chi2": 0,
            "pair_chi2_p": 1,
            "pair_cramers_v": 0,
            "pair_is_time_series": 0,
        }
        features.update(defaults)

        # Numeric-Numeric: Pearson correlation
        if col1_dtype == "numeric" and col2_dtype == "numeric":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr = df[[col1, col2]].corr().iloc[0, 1]
                    if pd.isna(corr) or np.isinf(corr):
                        corr = 0
                    else:
                        corr = np.clip(corr, -1.0, 1.0)
                    features.update(
                        {
                            "pair_correlation": corr,
                            "pair_correlation_abs": abs(corr),
                            "pair_is_strongly_correlated": int(abs(corr) > 0.7),
                            "pair_type_num_num": 1,
                        }
                    )
            except Exception as e:
                self._log(f"Correlation computation failed: {e}")

        # Categorical-Numeric: ANOVA F-test
        elif (col1_dtype == "categorical" and col2_dtype == "numeric") or (
            col1_dtype == "numeric" and col2_dtype == "categorical"
        ):
            cat_col = col1 if col1_dtype == "categorical" else col2
            num_col = col2 if col1_dtype == "categorical" else col1
            features["pair_type_cat_num"] = 1
            features["pair_category_count"] = df[cat_col].nunique()

            try:
                groups = df.groupby(cat_col)[num_col].apply(list)
                clean_groups = [g for g in groups if len(g) > 1]

                if len(clean_groups) > 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f_stat, p_value = stats.f_oneway(*clean_groups)

                    if pd.isna(f_stat):
                        f_stat, p_value = 0, 1

                    if not (np.isinf(f_stat) or np.isnan(f_stat)):
                        f_stat = min(f_stat, 1e6)
                    else:
                        f_stat = 0

                    features.update(
                        {
                            "pair_anova_f": np.log1p(f_stat),
                            "pair_anova_p": p_value,
                            "pair_anova_significant": int(p_value < 0.05),
                        }
                    )
            except Exception as e:
                self._log(f"ANOVA computation failed: {e}")

        # Categorical-Categorical: Chi-squared test
        elif col1_dtype == "categorical" and col2_dtype == "categorical":
            features["pair_type_cat_cat"] = 1
            try:
                contingency = pd.crosstab(df[col1], df[col2])
                if contingency.size > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        chi2, p_value, dof, expected = stats.chi2_contingency(
                            contingency
                        )

                    n = contingency.sum().sum()
                    min_dim = min(contingency.shape) - 1
                    cramers_v = (
                        np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0
                    )

                    if not (np.isinf(chi2) or np.isnan(chi2)):
                        chi2 = min(chi2, 1e6)
                    else:
                        chi2 = 0

                    features.update(
                        {
                            "pair_chi2": np.log1p(chi2),
                            "pair_chi2_p": p_value,
                            "pair_cramers_v": cramers_v,
                        }
                    )
            except Exception as e:
                self._log(f"Chi-squared computation failed: {e}")

        # Datetime-Numeric: Time series marker
        elif (col1_dtype == "datetime" and col2_dtype == "numeric") or (
            col1_dtype == "numeric" and col2_dtype == "datetime"
        ):
            features["pair_type_time_num"] = 1
            features["pair_is_time_series"] = 1

        return features
