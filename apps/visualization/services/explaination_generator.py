from typing import Dict


class ExplanationGenerator:
    def generate(
        self, chart_type: str, fields: Dict, score: int, profile: Dict
    ) -> Dict:
        columns_info = {col["name"]: col for col in profile["columns"]}
        corr_analysis = profile.get("correlation_analysis", {})

        explanation = {
            "chart_type": chart_type,
            "reason": "",
            "insights": [],
            "warnings": [],
            "suitability_score": score,
            "data_quality": 100
            - profile["dataset_overview"].get("missing_overall_percent", 0),
        }

        if chart_type == "scatter":
            explanation = self._explain_scatter(
                fields, columns_info, corr_analysis, explanation
            )
        elif chart_type in ["line", "area"]:
            explanation = self._explain_line(fields, columns_info, explanation)
        elif chart_type == "pie":
            explanation = self._explain_pie(fields, columns_info, explanation)
        elif chart_type == "histogram":
            explanation = self._explain_histogram(fields, columns_info, explanation)
        elif chart_type == "treemap":
            explanation = self._explain_treemap(fields, columns_info, explanation)
        else:
            explanation[
                "reason"
            ] = f"{chart_type.replace('_', ' ').title()} recommended based on data structure"

        for field_val in fields.values():
            if isinstance(field_val, str) and field_val in columns_info:
                col_info = columns_info[field_val]
                missing_pct = col_info.get("missing_percent", 0)
                if missing_pct > 10:
                    explanation["warnings"].append(
                        f"{field_val}: {missing_pct:.1f}% missing values"
                    )

        return explanation

    def _explain_scatter(self, fields, columns_info, corr_analysis, explanation):
        x, y = fields.get("x"), fields.get("y")
        if x and y:
            corr_val = None
            for pair in corr_analysis.get("correlation_pairs", []):
                if (pair["col1"] == x and pair["col2"] == y) or (
                    pair["col1"] == y and pair["col2"] == x
                ):
                    corr_val = pair["correlation"]
                    break

            if corr_val:
                explanation[
                    "reason"
                ] = f"Strong correlation ({corr_val:.2f}) between {x} and {y}"
                explanation["insights"].append(
                    f"{'Positive' if corr_val > 0 else 'Negative'} relationship"
                )
            else:
                explanation["reason"] = f"Explore relationship between {x} and {y}"

        return explanation

    def _explain_line(self, fields, columns_info, explanation):
        x_col = fields.get("x")
        if x_col:
            x_info = columns_info.get(x_col, {})
            time_info = x_info.get("time_series", {})
            if time_info.get("is_sequential", False):
                span = time_info.get("time_span_days", 0)
                explanation["reason"] = "Sequential time series detected"
                explanation["insights"].append(f"Time span: {span} days")
                if time_info.get("has_gaps", False):
                    explanation["warnings"].append("Time series has gaps")
            else:
                explanation["reason"] = "Trend visualization over ordered data"

        return explanation

    def _explain_pie(self, fields, columns_info, explanation):
        cat_col = fields.get("category")
        if cat_col:
            cat_info = columns_info.get(cat_col, {})
            card = cat_info.get("cardinality", {})
            balance = cat_info.get("balance", {})

            explanation[
                "reason"
            ] = f"Proportional breakdown ({card.get('unique_count', 0)} categories)"

            if balance.get("is_imbalanced", False):
                dom_pct = balance.get("dominant_category_percent", 0)
                explanation["warnings"].append(
                    f"Data is imbalanced ({dom_pct:.1f}% in one category)"
                )

        return explanation

    def _explain_histogram(self, fields, columns_info, explanation):
        x_col = fields.get("x")
        if x_col:
            x_info = columns_info.get(x_col, {})
            dist = x_info.get("distribution", {})
            dist_type = dist.get("distribution_type", "unknown")

            explanation["reason"] = f"Distribution analysis ({dist_type})"

            if dist.get("has_outliers", False):
                outlier_count = dist.get("outlier_count", 0)
                explanation["insights"].append(f"{outlier_count} outliers detected")

            skew = dist.get("skewness", 0)
            if abs(skew) > 1:
                explanation["insights"].append(
                    f"{'Right' if skew > 0 else 'Left'}-skewed distribution"
                )

        return explanation

    def _explain_treemap(self, fields, columns_info, explanation):
        cat_col = fields.get("category")
        if cat_col:
            cat_info = columns_info.get(cat_col, {})
            card = cat_info.get("cardinality", {})
            explanation[
                "reason"
            ] = f"Hierarchical view of {card.get('unique_count', 0)} categories"

        return explanation
