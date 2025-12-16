from typing import Dict

from apps.visualization.services.analyzer import DatasetAnalyzer


class FieldSelector:
    def select_fields(self, chart_type: str, profile: Dict) -> Dict[str, str]:
        """
        Select the best fields for a given chart type
        :param chart_type:
        :param profile:
        :return: A dictionary of fields
        """
        columns_info = {col["name"]: col for col in profile["columns"]}
        corr_analysis = profile.get("correlation_analysis", {})
        hier_analysis = profile.get("hierarchical_analysis", {})

        analyzer = DatasetAnalyzer()
        column_types = analyzer.get_column_types(profile)

        num_cols = column_types["numeric"]
        cat_cols = column_types["categorical"]
        dt_cols = column_types["datetime"]

        def best_numeric(exclude=None):
            exclude = exclude or []
            candidates = [c for c in num_cols if c not in exclude]
            if not candidates:
                return None

            scored = []
            for col in candidates:
                info = columns_info.get(col, {})
                cv = info.get("distribution", {}).get("coefficient_of_variation", 0)
                scored.append((col, cv))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0] if scored else candidates[0]

        def best_categorical(exclude=None, low_card=True):
            exclude = exclude or []
            candidates = [c for c in cat_cols if c not in exclude]
            if not candidates:
                return None

            if low_card:
                scored = []
                for col in candidates:
                    info = columns_info.get(col, {})
                    unique = info.get("cardinality", {}).get("unique_count", 999)
                    balance_bonus = (
                        -5 if info.get("balance", {}).get("is_balanced", False) else 0
                    )
                    scored.append((col, unique + balance_bonus))
                scored.sort(key=lambda x: x[1])
                return scored[0][0] if scored else candidates[0]
            else:
                scored = []
                for col in candidates:
                    info = columns_info.get(col, {})
                    unique = info.get("cardinality", {}).get("unique_count", 0)
                    scored.append((col, unique))
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[0][0] if scored else candidates[0]

        def best_datetime():
            if not dt_cols:
                return None
            for dt_col in dt_cols:
                col_info = columns_info.get(dt_col, {})
                if col_info.get("time_series", {}).get("is_sequential", False):
                    return dt_col
            return dt_cols[0]

        fields = {}

        if chart_type == "scatter":
            pairs = corr_analysis.get("correlation_pairs", [])
            if pairs:
                best_pair = max(pairs, key=lambda x: x["correlation"])
                fields["x"] = best_pair["col1"]
                fields["y"] = best_pair["col2"]
            elif len(num_cols) >= 2:
                fields["x"] = num_cols[0]
                fields["y"] = num_cols[1]

        elif chart_type in ["line", "area"]:
            fields["x"] = best_datetime() or best_categorical()
            fields["y"] = best_numeric()

        elif chart_type == "bar":
            fields["x"] = best_categorical(low_card=True)
            fields["y"] = best_numeric()

        elif chart_type == "grouped_bar":
            fields["x"] = best_categorical(low_card=True)
            fields["group"] = best_categorical(exclude=[fields["x"]], low_card=True)
            fields["y"] = best_numeric()

        elif chart_type == "pie":
            pie_col = None
            for cat_col in cat_cols:
                if (
                    columns_info.get(cat_col, {})
                    .get("cardinality", {})
                    .get("is_suitable_for_pie", False)
                ):
                    pie_col = cat_col
                    break
            fields["category"] = pie_col or best_categorical(low_card=True)
            fields["value"] = best_numeric()

        elif chart_type == "histogram":
            fields["x"] = best_numeric()

        elif chart_type in ["boxplot", "violin"]:
            fields["x"] = best_categorical(low_card=True)
            fields["y"] = best_numeric()

        elif chart_type == "treemap":
            pairs = hier_analysis.get("hierarchical_pairs", [])
            if pairs:
                best_pair = max(pairs, key=lambda x: x["strength"])
                fields["category"] = best_pair["parent"]
                fields["subcategory"] = best_pair["child"]
            else:
                fields["category"] = best_categorical(low_card=False)
            fields["value"] = best_numeric()

        elif chart_type == "bubble":
            if len(num_cols) >= 3:
                fields["x"] = num_cols[0]
                fields["y"] = num_cols[1]
                fields["size"] = num_cols[2]

        elif chart_type == "heatmap":
            corr_cols = set()
            for pair in corr_analysis.get("correlation_pairs", [])[:10]:
                corr_cols.add(pair["col1"])
                corr_cols.add(pair["col2"])
            fields["columns"] = list(corr_cols) if corr_cols else num_cols[:10]

        else:
            fields["x"] = best_categorical() or best_numeric()
            fields["y"] = best_numeric(exclude=[fields.get("x")])

        return fields
