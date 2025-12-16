# import pandas as pd
# import numpy as np
# from typing import Dict, List, Any, Optional, Tuple
# import json
#
# from apps.pages.utils import prepare_plotly_charts_from_recommendations
#
#
# class SimpleChartSelector:
#     """
#     Lightweight rule-based chart selector (no LLM needed).
#     Uses profiler metadata + keyword matching.
#     """
#
#     # Intent keywords
#     INTENT_KEYWORDS = {
#         'trend': ['trend', 'over time', 'time', 'history', 'change', 'growth'],
#         'compare': ['compare', 'vs', 'versus', 'difference', 'between'],
#         'distribution': ['distribution', 'spread', 'range', 'histogram'],
#         'correlation': ['correlation', 'relationship', 'related', 'affect'],
#         'composition': ['share', 'proportion', 'breakdown', 'composition', 'percent'],
#         'ranking': ['top', 'bottom', 'highest', 'lowest', 'rank', 'best', 'worst']
#     }
#
#     def detect_intent(self, query: str) -> str:
#         """Detect user intent from query."""
#         query_lower = query.lower()
#
#         for intent, keywords in self.INTENT_KEYWORDS.items():
#             if any(kw in query_lower for kw in keywords):
#                 return intent
#
#         return 'explore'  # Default
#
#     def select_chart(
#             self,
#             user_query: str,
#             profiler_metadata: dict,
#             max_charts: int = 2
#     ) -> dict:
#         """
#         Select charts using rules (no LLM).
#
#         Parameters
#         ----------
#         user_query : str
#             User's question
#         profiler_metadata : dict
#             From profiler.profile()
#         max_charts : int
#             Maximum charts to return
#
#         Returns
#         -------
#         dict
#             Chart recommendations in LLM response format
#         """
#         intent = self.detect_intent(user_query)
#         columns = profiler_metadata.get("columns", [])
#         hints = profiler_metadata.get("chart_hints", {})
#         suitable_charts = hints.get("suitable_charts", [])
#         reasons = hints.get("reasons", {})
#
#         # Get column types
#         numeric_cols = [c for c in columns if c["dtype"] == "numeric"]
#         categorical_cols = [c for c in columns if c["dtype"] == "categorical"]
#         datetime_cols = [c for c in columns if c["dtype"] == "datetime"]
#
#         recommendations = []
#
#         # ===== RULE-BASED SELECTION =====
#
#         # Intent: TREND (datetime required)
#         if intent == 'trend' and datetime_cols and numeric_cols:
#             dt_col = datetime_cols[0]["name"]
#             num_col = numeric_cols[0]["name"]
#
#             recommendations.append({
#                 "chart_type": "line",
#                 "axes": {"x": dt_col, "y": num_col, "x_type": "datetime"},
#                 "aggregation": None,
#                 "title": f"{num_col} Over Time",
#                 "summary": f"Trend analysis of {num_col} over {dt_col}"
#             })
#
#         # Intent: COMPOSITION
#         elif intent == 'composition' and categorical_cols and numeric_cols:
#             cat_col = categorical_cols[0]
#             num_col = numeric_cols[0]["name"]
#             cat_name = cat_col["name"]
#             unique_count = cat_col.get("cardinality", {}).get("unique_count", 999)
#
#             if unique_count <= 8:
#                 chart_type = "pie"
#             elif unique_count <= 30:
#                 chart_type = "bar"
#             else:
#                 chart_type = "treemap"
#
#             recommendations.append({
#                 "chart_type": chart_type,
#                 "axes": {"category": cat_name, "value": num_col, "x_type": "category"},
#                 "aggregation": {"method": "sum", "group_by": [cat_name]},
#                 "title": f"{num_col} Breakdown by {cat_name}",
#                 "summary": f"Composition showing share of {num_col} across {cat_name}"
#             })
#
#         # Intent: COMPARISON
#         elif intent == 'compare' and categorical_cols and numeric_cols:
#             cat_col = categorical_cols[0]["name"]
#             num_col = numeric_cols[0]["name"]
#
#             recommendations.append({
#                 "chart_type": "bar",
#                 "axes": {"x": cat_col, "y": num_col, "x_type": "category"},
#                 "aggregation": {"method": "sum", "group_by": [cat_col]},
#                 "title": f"Compare {num_col} by {cat_col}",
#                 "summary": f"Side-by-side comparison of {num_col} across {cat_col}"
#             })
#
#         # Intent: CORRELATION
#         elif intent == 'correlation' and len(numeric_cols) >= 2:
#             col1 = numeric_cols[0]["name"]
#             col2 = numeric_cols[1]["name"]
#
#             recommendations.append({
#                 "chart_type": "scatter",
#                 "axes": {"x": col1, "y": col2, "x_type": "numeric"},
#                 "aggregation": None,
#                 "title": f"{col2} vs {col1}",
#                 "summary": f"Relationship between {col1} and {col2}"
#             })
#
#         # Intent: DISTRIBUTION
#         elif intent == 'distribution' and numeric_cols:
#             num_col = numeric_cols[0]["name"]
#
#             recommendations.append({
#                 "chart_type": "histogram",
#                 "axes": {"x": num_col, "x_type": "numeric"},
#                 "aggregation": None,
#                 "title": f"Distribution of {num_col}",
#                 "summary": f"Frequency distribution showing spread of {num_col}"
#             })
#
#         # FALLBACK: Use profiler's top suggestion
#         else:
#             if suitable_charts and suitable_charts[0] in ['line', 'bar', 'scatter', 'pie', 'histogram']:
#                 top_chart = suitable_charts[0]
#
#                 # Build axes based on chart type
#                 axes = {}
#                 if top_chart in ['bar', 'pie'] and categorical_cols and numeric_cols:
#                     axes = {
#                         "x": categorical_cols[0]["name"],
#                         "y": numeric_cols[0]["name"],
#                         "x_type": "category"
#                     }
#                     agg = {"method": "sum", "group_by": [categorical_cols[0]["name"]]}
#                 elif top_chart == 'line' and datetime_cols and numeric_cols:
#                     axes = {
#                         "x": datetime_cols[0]["name"],
#                         "y": numeric_cols[0]["name"],
#                         "x_type": "datetime"
#                     }
#                     agg = None
#                 elif top_chart == 'scatter' and len(numeric_cols) >= 2:
#                     axes = {
#                         "x": numeric_cols[0]["name"],
#                         "y": numeric_cols[1]["name"],
#                         "x_type": "numeric"
#                     }
#                     agg = None
#                 elif top_chart == 'histogram' and numeric_cols:
#                     axes = {"x": numeric_cols[0]["name"], "x_type": "numeric"}
#                     agg = None
#                 else:
#                     axes = {"x": columns[0]["name"]}
#                     agg = None
#
#                 recommendations.append({
#                     "chart_type": top_chart,
#                     "axes": axes,
#                     "aggregation": agg,
#                     "title": f"{top_chart.title()} Chart",
#                     "summary": reasons.get(top_chart, f"Profiler recommended {top_chart}")
#                 })
#
#         # Limit to max_charts
#         recommendations = recommendations[:max_charts]
#
#         # Ensure we have at least one recommendation
#         if not recommendations and columns:
#             # Ultimate fallback: bar chart of first two columns
#             if len(columns) >= 2:
#                 recommendations.append({
#                     "chart_type": "bar",
#                     "axes": {"x": columns[0]["name"], "y": columns[1]["name"]},
#                     "aggregation": {"method": "count", "group_by": [columns[0]["name"]]},
#                     "title": "Data Overview",
#                     "summary": "Basic visualization of available data"
#                 })
#
#         return {
#             "visualizations": recommendations,
#             "intent_detected": intent,
#             "profiler_hints_used": suitable_charts[:3]
#         }
#
#
# class DataAggregator:
#     """Your existing aggregator (unchanged)"""
#
#     def aggregate(self, df: pd.DataFrame, chart_type: str, fields: Dict) -> Tuple[pd.DataFrame, Dict]:
#         """
#         Aggregate data for chart type.
#         Returns (aggregated_df, extra_info)
#         """
#         extra_info = {}
#
#         if chart_type == 'histogram':
#             x = fields.get('x')
#             if x and x in df.columns:
#                 clean_df = df[[x]].dropna()
#                 extra_info['bins'] = self._smart_bins(clean_df[x])
#                 return clean_df, extra_info
#
#         if chart_type in ['scatter', 'bubble']:
#             cols = [fields.get(c) for c in ['x', 'y', 'size'] if fields.get(c) and fields.get(c) in df.columns]
#             return df[cols].dropna(), extra_info
#
#         if chart_type in ['boxplot', 'violin']:
#             cols = [fields.get(c) for c in ['x', 'y'] if fields.get(c) and fields.get(c) in df.columns]
#             return df[cols].dropna(), extra_info
#
#         if chart_type == 'heatmap':
#             cols = fields.get('columns', [])
#             cols = [c for c in cols if c in df.columns]
#             if cols:
#                 return df[cols].corr(), extra_info
#             return df, extra_info
#
#         group_cols = [fields.get(k) for k in ['x', 'group', 'category', 'subcategory'] if fields.get(k)]
#         group_cols = [c for c in group_cols if c in df.columns]
#         value_col = fields.get('y') or fields.get('value')
#
#         if not group_cols or not value_col or value_col not in df.columns:
#             return df, extra_info
#
#         agg_method = self._choose_agg(df, value_col)
#         grouped = df.groupby(group_cols, dropna=False)[value_col].agg(agg_method).reset_index()
#         extra_info['aggregation'] = agg_method
#
#         if chart_type in ['line', 'area'] and fields.get('x') in grouped.columns:
#             grouped = grouped.sort_values(by=fields['x'])
#
#         return grouped, extra_info
#
#     def _smart_bins(self, series: pd.Series) -> int:
#         n = len(series)
#         if n <= 20:
#             return 5
#         if n <= 100:
#             return int(np.sqrt(n))
#         IQR = series.quantile(0.75) - series.quantile(0.25)
#         if IQR == 0:
#             return 10
#         h = 2 * IQR / (n ** (1 / 3))
#         return max(10, int((series.max() - series.min()) / h))
#
#     def _choose_agg(self, df: pd.DataFrame, col: str) -> str:
#         if col not in df.columns:
#             return 'mean'
#         unique_ratio = df[col].nunique() / len(df)
#         if unique_ratio < 0.1:
#             return 'sum'
#         if unique_ratio < 0.5:
#             return 'mean'
#         return 'median'
#
#
# class RecommendationToFieldsAdapter:
#     """
#     Converts chart recommendations to the format expected by your DataAggregator.
#     """
#
#     @staticmethod
#     def convert_recommendation_to_fields(recommendation: Dict) -> Dict:
#         """
#         Convert selector/LLM recommendation to DataAggregator fields format.
#
#         Input (from selector/LLM):
#         {
#             "chart_type": "bar",
#             "axes": {"x": "region", "y": "sales", "group": "product"},
#             "aggregation": {"method": "sum", "group_by": ["region"]}
#         }
#
#         Output (for DataAggregator):
#         {
#             "x": "region",
#             "y": "sales",
#             "group": "product"
#         }
#         """
#         axes = recommendation.get("axes", {})
#
#         # Direct mapping
#         fields = {}
#         for key in ["x", "y", "group", "size", "color", "category", "value", "subcategory"]:
#             if key in axes and axes[key]:
#                 fields[key] = axes[key]
#
#         # Handle special cases for different chart types
#         chart_type = recommendation.get("chart_type")
#
#         # Pie/Donut charts use "category" and "value"
#         if chart_type in ["pie", "donut"]:
#             if "category" in axes:
#                 fields["category"] = axes["category"]
#             elif "x" in axes:
#                 fields["category"] = axes["x"]
#
#             if "value" in axes:
#                 fields["value"] = axes["value"]
#             elif "y" in axes:
#                 fields["value"] = axes["y"]
#
#         # Heatmap needs columns list
#         if chart_type == "heatmap":
#             if "columns" in axes:
#                 fields["columns"] = axes["columns"]
#             elif "x" in axes and "y" in axes:
#                 # For correlation heatmap, we might want all numeric columns
#                 # This will be handled by the aggregator
#                 pass
#
#         return fields
#
#
# class AdaptedChartPipeline:
#     """
#     Complete pipeline using your existing DataAggregator.
#     """
#
#     def __init__(self, df: pd.DataFrame, profiler_metadata: Dict):
#         """
#         Initialize pipeline with your DataAggregator.
#
#         Parameters
#         ----------
#         df : pd.DataFrame
#             Source data
#         profiler_metadata : dict
#             Output from DatasetProfiler.profile()
#         """
#         self.df = df
#         self.profiler_metadata = profiler_metadata
#         self.aggregator = DataAggregator()  # Your existing aggregator
#         self.adapter = RecommendationToFieldsAdapter()
#
#     def generate_charts(
#             self,
#             user_query: str,
#             use_llm: bool = False,
#             llm_response: Optional[str] = None
#     ) -> List[Dict]:
#         """
#         Generate Plotly charts from user query using your DataAggregator.
#
#         Parameters
#         ----------
#         user_query : str
#             User's natural language query
#         use_llm : bool
#             Whether to use LLM (if False, uses rule-based selector)
#         llm_response : str, optional
#             Pre-computed LLM response JSON string
#
#         Returns
#         -------
#         list of dict
#             Plotly chart configurations ready for frontend
#         """
#         # Step 1: Get chart recommendations
#         if llm_response:
#             try:
#                 recommendations = json.loads(llm_response)
#             except json.JSONDecodeError as e:
#                 print(f"JSON decode error: {e}")
#                 return []
#         elif use_llm:
#             raise NotImplementedError("LLM integration required. Pass llm_response instead.")
#         else:
#             # Use rule-based selector
#             selector = SimpleChartSelector()
#             recommendations = selector.select_chart(
#                 user_query=user_query,
#                 profiler_metadata=self.profiler_metadata,
#                 max_charts=3
#             )
#
#         # Extract visualizations
#         visualizations = recommendations.get("visualizations", [])
#         if not visualizations:
#             return []
#
#         # Step 2: Process each recommendation with your DataAggregator
#         chart_requests = []
#         for viz in visualizations:
#             chart_type = viz.get("chart_type")
#
#             # Convert recommendation to fields format
#             fields = self.adapter.convert_recommendation_to_fields(viz)
#
#             # Use YOUR DataAggregator
#             try:
#                 aggregated_df, extra_info = self.aggregator.aggregate(
#                     df=self.df,
#                     chart_type=chart_type,
#                     fields=fields
#                 )
#
#                 if aggregated_df.empty:
#                     print(f"Warning: No data after aggregation for {chart_type}")
#                     continue
#
#                 # Convert DataFrame to list of dicts for PlotlyChartGenerator
#                 aggregated_data = aggregated_df.to_dict(orient="records")
#
#                 # Build chart request
#                 chart_request = {
#                     "chart_type": chart_type,
#                     "fields": fields,
#                     "aggregated_data": aggregated_data,
#                     "title": viz.get("title", f"{chart_type.title()} Chart"),
#                     "summary": viz.get("summary", ""),
#                     "reason": viz.get("reason", ""),
#                     "extra_info": extra_info  # Include bins, aggregation method, etc.
#                 }
#                 chart_requests.append(chart_request)
#
#             except Exception as e:
#                 print(f"Error aggregating data for {chart_type}: {e}")
#                 continue
#
#         # Step 3: Generate Plotly charts
#         plotly_charts = prepare_plotly_charts_from_recommendations(
#             recommendations=chart_requests,
#             context_id=f"query_{hash(user_query)}"
#         )
#
#         # Step 4: Add extra_info to chart metadata
#         for i, chart in enumerate(plotly_charts):
#             if i < len(chart_requests):
#                 extra = chart_requests[i].get("extra_info", {})
#                 chart["aggregation_method"] = extra.get("aggregation", "none")
#                 chart["bins"] = extra.get("bins")
#
#         return plotly_charts
#
#

from typing import Dict, Tuple

import numpy as np
import pandas as pd


class DataAggregator:
    def aggregate(
        self, df: pd.DataFrame, chart_type: str, fields: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Aggregate data for chart type.
        Returns (aggregated_df, extra_info)
        """
        extra_info = {}

        if chart_type == "histogram":
            x = fields.get("x")
            if x and x in df.columns:
                clean_df = df[[x]].dropna()
                extra_info["bins"] = self._smart_bins(clean_df[x])
                return clean_df, extra_info

        if chart_type in ["scatter", "bubble"]:
            cols = [
                fields.get(c)
                for c in ["x", "y", "size"]
                if fields.get(c) and fields.get(c) in df.columns
            ]
            return df[cols].dropna(), extra_info

        if chart_type in ["boxplot", "violin"]:
            cols = [
                fields.get(c)
                for c in ["x", "y"]
                if fields.get(c) and fields.get(c) in df.columns
            ]
            return df[cols].dropna(), extra_info

        if chart_type == "heatmap":
            cols = fields.get("columns", [])
            cols = [c for c in cols if c in df.columns]
            if cols:
                return df[cols].corr(), extra_info
            return df, extra_info

        group_cols = [
            fields.get(k)
            for k in ["x", "group", "category", "subcategory"]
            if fields.get(k)
        ]
        group_cols = [c for c in group_cols if c in df.columns]
        value_col = fields.get("y") or fields.get("value")

        if not group_cols or not value_col or value_col not in df.columns:
            return df, extra_info

        agg_method = self._choose_agg(df, value_col)
        grouped = (
            df.groupby(group_cols, dropna=False)[value_col]
            .agg(agg_method)
            .reset_index()
        )
        extra_info["aggregation"] = agg_method

        if chart_type in ["line", "area"] and fields.get("x") in grouped.columns:
            grouped = grouped.sort_values(by=fields["x"])

        return grouped, extra_info

    def _smart_bins(self, series: pd.Series) -> int:
        n = len(series)
        if n <= 20:
            return 5
        if n <= 100:
            return int(np.sqrt(n))
        IQR = series.quantile(0.75) - series.quantile(0.25)
        if IQR == 0:
            return 10
        h = 2 * IQR / (n ** (1 / 3))
        return max(10, int((series.max() - series.min()) / h))

    def _choose_agg(self, df: pd.DataFrame, col: str) -> str:
        if col not in df.columns:
            return "mean"
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.1:
            return "sum"
        if unique_ratio < 0.5:
            return "mean"
        return "median"
