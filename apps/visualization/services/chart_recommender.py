from typing import Dict, List

from apps.visualization.services.analyzer import DatasetAnalyzer


class ChartRecommender:
    def __init__(self, analyzer: DatasetAnalyzer):
        self.analyzer = analyzer

    def recommend(self, profile: Dict, top_k: int = 3) -> List[Dict]:
        """
        Generate top k recommendations with detailed analysis
        :param profile:
        :param top_k:
        :return: A list of recommendations[Dict]
        """
        chart_hints = profile.get("chart_hints", {})
        suitable_charts = chart_hints.get("suitable_charts", [])

        if not suitable_charts:
            suitable_charts = self._get_fallback_charts(profile)

        ranked_charts = self._rank_charts(profile, suitable_charts[:5])

        return ranked_charts

    def _get_fallback_charts(self, profile: Dict) -> List[str]:
        column_types = self.analyzer.get_column_types(profile)
        charts = []

        num_cols = column_types["numeric"]
        cat_cols = column_types["categorical"]
        dt_cols = column_types["datetime"]

        if dt_cols and num_cols:
            charts.extend(["line", "area"])
        if cat_cols and num_cols:
            charts.append("bar")
        if len(num_cols) >= 2:
            charts.append("scatter")
        if num_cols:
            charts.append("histogram")
        if len(cat_cols) >= 2 and num_cols:
            charts.append("grouped_bar")

        return charts[:5]

    def _rank_charts(self, profile: Dict, chart_types: List[str]) -> List[Dict]:
        columns_info = {col["name"]: col for col in profile["columns"]}
        corr_analysis = profile.get("correlation_analysis", {})
        hier_analysis = profile.get("hierarchical_analysis", {})

        column_types = self.analyzer.get_column_types(profile)
        num_cols = column_types["numeric"]
        cat_cols = column_types["categorical"]
        dt_cols = column_types["datetime"]

        ranked = []

        for chart_type in chart_types:
            score = 10

            if chart_type == "scatter":
                if corr_analysis.get("has_strong_correlations", False):
                    score += 30
                    score += len(corr_analysis.get("correlation_pairs", [])) * 5
                if len(num_cols) >= 2:
                    score += 10

            elif chart_type == "line":
                if dt_cols:
                    score += 25
                    for dt_col in dt_cols:
                        col_info = columns_info.get(dt_col, {})
                        if col_info.get("time_series", {}).get("is_sequential", False):
                            score += 20

            elif chart_type == "bar":
                for cat_col in cat_cols:
                    col_info = columns_info.get(cat_col, {})
                    if col_info.get("cardinality", {}).get(
                        "is_suitable_for_bar", False
                    ):
                        score += 20
                        if col_info.get("balance", {}).get("is_balanced", False):
                            score += 5

            elif chart_type == "pie":
                suitable = any(
                    columns_info.get(c, {})
                    .get("cardinality", {})
                    .get("is_suitable_for_pie", False)
                    for c in cat_cols
                )
                score += 30 if suitable else -10

            elif chart_type == "treemap":
                if hier_analysis.get("has_hierarchy", False):
                    score += 35
                if any(
                    columns_info.get(c, {})
                    .get("cardinality", {})
                    .get("is_suitable_for_treemap", False)
                    for c in cat_cols
                ):
                    score += 25

            elif chart_type == "histogram":
                if num_cols:
                    score += 15
                for num_col in num_cols:
                    if (
                        columns_info.get(num_col, {})
                        .get("distribution", {})
                        .get("has_outliers", False)
                    ):
                        score += 10
                        break

            elif chart_type in ["boxplot", "violin"]:
                if cat_cols and num_cols:
                    score += 15
                    if any(
                        columns_info.get(c, {})
                        .get("distribution", {})
                        .get("has_outliers", False)
                        for c in num_cols
                    ):
                        score += 15

            ranked.append({"type": chart_type, "score": score})

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked


import hashlib
from typing import Dict, List

import plotly.graph_objects as go


class PlotlyChartGenerator:
    """Generate Plotly chart configurations with consistent styling."""

    # Curated color palettes for consistency
    CATEGORICAL_COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]

    SEQUENTIAL_COLORS = "Blues"  # For heatmaps, treemaps
    DIVERGING_COLORS = "RdBu"  # For correlations

    def __init__(self):
        """Initialize the chart generator."""
        self.color_cache = {}  # Cache colors for consistency

    def _get_consistent_color(self, key: str, palette: List[str] = None) -> str:
        """
        Get consistent color for a given key using hash-based assignment.
        Same key always gets same color across different charts.
        """
        if palette is None:
            palette = self.CATEGORICAL_COLORS

        if key not in self.color_cache:
            # Hash the key to get consistent index
            hash_val = int(hashlib.md5(str(key).encode()).hexdigest(), 16)
            idx = hash_val % len(palette)
            self.color_cache[key] = palette[idx]

        return self.color_cache[key]

    def _get_color_palette(self, n: int, palette: List[str] = None) -> List[str]:
        """Get n colors from palette with wrapping."""
        if palette is None:
            palette = self.CATEGORICAL_COLORS
        return [palette[i % len(palette)] for i in range(n)]

    def generate_chart(
        self,
        chart_type: str,
        fields: Dict[str, str],
        aggregated_data: List[Dict],
        title: str = None,
        summary: str = None,
        reason: str = None,
        context_id: str = None,
    ) -> Dict:
        """
        Generate Plotly chart configuration.

        Parameters
        ----------
        chart_type : str
            Type of chart (e.g., 'bar', 'scatter', 'line')
        fields : dict
            Mapping of field roles (e.g., {'x': 'date', 'y': 'sales'})
        aggregated_data : list of dict
            Data rows for the chart
        title : str, optional
            Chart title
        summary : str, optional
            Chart summary/description
        reason : str, optional
            Reason for chart recommendation
        context_id : str, optional
            Context identifier

        Returns
        -------
        dict
            Plotly chart configuration
        """
        if not aggregated_data:
            return self._create_empty_chart(title or "No Data")

        # Dispatch to appropriate handler
        handler = getattr(self, f"_create_{chart_type}", None)
        if handler is None:
            return self._create_fallback_chart(
                chart_type, fields, aggregated_data, title
            )

        try:
            fig = handler(fields, aggregated_data, title)

            # Apply common styling
            self._apply_common_styling(fig, title)

            return {
                "context_id": context_id,
                "chart_config": fig.to_dict(),
                "summary": summary or "",
                "reason": reason or "",
                "chart_type": chart_type,
            }
        except Exception as e:
            print(f"Error generating {chart_type}: {e}")
            return self._create_fallback_chart(
                chart_type, fields, aggregated_data, title
            )

    def _apply_common_styling(self, fig: go.Figure, title: str = None):
        """Apply consistent styling to all charts."""
        fig.update_layout(
            title={
                "text": title or "",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 18, "family": "Arial, sans-serif"},
            },
            plot_bgcolor="rgba(240, 240, 240, 0.5)",
            paper_bgcolor="white",
            font={"family": "Arial, sans-serif", "size": 12},
            hovermode="closest",
            margin={"l": 60, "r": 40, "t": 80, "b": 60},
            autosize=True,
        )

    # ========== UNIVARIATE CHARTS ==========

    def _create_histogram(
        self, fields: Dict, data: List[Dict], title: str
    ) -> go.Figure:
        """Create histogram with proper binning."""
        value_col = fields.get("x") or fields.get("value") or fields.get("y")
        values = [
            row.get(value_col)
            for row in data
            if value_col in row and row.get(value_col) is not None
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=values,
                name=value_col,
                marker={"color": "#1f77b4", "line": {"color": "#0d3d5c", "width": 1}},
                opacity=0.75,
                nbinsx=min(30, max(10, len(values) // 10)),  # Smart binning
            )
        )

        fig.update_xaxes(title_text=value_col)
        fig.update_yaxes(title_text="Frequency")

        return fig

    def _create_pie(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create pie chart with consistent colors."""
        category_col = fields.get("category") or fields.get("labels") or fields.get("x")
        value_col = fields.get("value") or fields.get("y")

        labels = [
            str(row.get(category_col, "Unknown")) for row in data if category_col in row
        ]
        values = [row.get(value_col, 0) for row in data if value_col in row]

        # Get consistent colors for categories
        colors = [self._get_consistent_color(label) for label in labels]

        fig = go.Figure()
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker={"colors": colors, "line": {"color": "white", "width": 2}},
                textinfo="label+percent",
                textposition="auto",
                hovertemplate="<b>%{label}</b><br>Value: %{value}<br>Percent: %{percent}<extra></extra>",
            )
        )

        return fig

    def _create_donut(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create donut chart (pie with hole)."""
        fig = self._create_pie(fields, data, title)
        fig.update_traces(hole=0.4)
        return fig

    # ========== BIVARIATE CHARTS ==========

    def _create_bar(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create bar chart with consistent colors."""
        x_col = fields.get("x") or fields.get("category")
        y_col = fields.get("y") or fields.get("value")

        x_data = [str(row.get(x_col, "")) for row in data if x_col in row]
        y_data = [row.get(y_col, 0) for row in data if y_col in row]

        # Consistent colors for categories
        colors = [self._get_consistent_color(x) for x in x_data]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=x_data,
                y=y_data,
                marker={"color": colors, "line": {"color": "white", "width": 1}},
                text=y_data,
                texttemplate="%{text:.2s}",
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>",
            )
        )

        fig.update_xaxes(title_text=x_col, tickangle=-45)
        fig.update_yaxes(title_text=y_col)

        return fig

    def _create_line(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create line chart for time series."""
        x_col = fields.get("x") or fields.get("date") or fields.get("time")
        y_col = fields.get("y") or fields.get("value")

        x_data = [row.get(x_col) for row in data if x_col in row]
        y_data = [row.get(y_col, 0) for row in data if y_col in row]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="lines+markers",
                line={"color": "#1f77b4", "width": 2},
                marker={"size": 6, "color": "#1f77b4"},
                name=y_col,
                hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>",
            )
        )

        fig.update_xaxes(title_text=x_col)
        fig.update_yaxes(title_text=y_col)

        return fig

    def _create_area(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create area chart (filled line)."""
        fig = self._create_line(fields, data, title)
        fig.update_traces(fill="tozeroy", fillcolor="rgba(31, 119, 180, 0.3)")
        return fig

    def _create_scatter(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create PROPER scatter plot with points only."""
        x_col = fields.get("x")
        y_col = fields.get("y")

        x_data = [
            row.get(x_col)
            for row in data
            if x_col in row and row.get(x_col) is not None
        ]
        y_data = [
            row.get(y_col)
            for row in data
            if y_col in row and row.get(y_col) is not None
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="markers",  # CRITICAL: markers only, no lines!
                marker={
                    "size": 8,
                    "color": "#1f77b4",
                    "opacity": 0.7,
                    "line": {"width": 0.5, "color": "white"},
                },
                name="Data Points",
                hovertemplate=f"<b>{x_col}: %{{x}}</b><br>{y_col}: %{{y}}<extra></extra>",
            )
        )

        fig.update_xaxes(title_text=x_col, showgrid=True, gridcolor="lightgray")
        fig.update_yaxes(title_text=y_col, showgrid=True, gridcolor="lightgray")

        return fig

    def _create_bubble(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create bubble chart (scatter with size dimension)."""
        x_col = fields.get("x")
        y_col = fields.get("y")
        size_col = fields.get("size") or y_col  # Use y as size if not specified

        x_data = [row.get(x_col) for row in data if x_col in row]
        y_data = [row.get(y_col) for row in data if y_col in row]
        size_data = [row.get(size_col, 10) for row in data if size_col in row]

        # Normalize sizes to 10-50 range
        if size_data:
            min_size, max_size = min(size_data), max(size_data)
            if max_size > min_size:
                size_normalized = [
                    10 + 40 * (s - min_size) / (max_size - min_size) for s in size_data
                ]
            else:
                size_normalized = [20] * len(size_data)
        else:
            size_normalized = [20] * len(x_data)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="markers",
                marker={
                    "size": size_normalized,
                    "color": "#ff7f0e",
                    "opacity": 0.6,
                    "line": {"width": 1, "color": "white"},
                },
                text=[f"Size: {s:.2f}" for s in size_data],
                hovertemplate=f"<b>{x_col}: %{{x}}</b><br>{y_col}: %{{y}}<br>%{{text}}<extra></extra>",
            )
        )

        fig.update_xaxes(title_text=x_col, showgrid=True)
        fig.update_yaxes(title_text=y_col, showgrid=True)

        return fig

    # ========== MULTI-CATEGORICAL CHARTS ==========

    def _create_grouped_bar(
        self, fields: Dict, data: List[Dict], title: str
    ) -> go.Figure:
        """Create grouped bar chart with color-coded groups."""
        x_col = fields.get("x") or fields.get("category")
        y_col = fields.get("y") or fields.get("value")
        group_col = fields.get("group") or fields.get("category2")

        if not group_col:
            # Fallback to regular bar if no group column
            return self._create_bar(fields, data, title)

        # Group data by category and group
        grouped = {}
        for row in data:
            group = str(row.get(group_col, "Unknown"))
            x_val = str(row.get(x_col, ""))
            y_val = row.get(y_col, 0)

            if group not in grouped:
                grouped[group] = {"x": [], "y": []}
            grouped[group]["x"].append(x_val)
            grouped[group]["y"].append(y_val)

        fig = go.Figure()

        for i, (group, values) in enumerate(grouped.items()):
            color = self._get_consistent_color(group)
            fig.add_trace(
                go.Bar(
                    name=group,
                    x=values["x"],
                    y=values["y"],
                    marker={"color": color, "line": {"color": "white", "width": 1}},
                    hovertemplate=f"<b>{group}</b><br>%{{x}}: %{{y}}<extra></extra>",
                )
            )

        fig.update_layout(
            barmode="group",
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend={"title": {"text": group_col}},
        )

        return fig

    def _create_stacked_bar(
        self, fields: Dict, data: List[Dict], title: str
    ) -> go.Figure:
        """Create stacked bar chart."""
        fig = self._create_grouped_bar(fields, data, title)
        fig.update_layout(barmode="stack")
        return fig

    def _create_boxplot(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create box plot for distribution across categories."""
        x_col = fields.get("x") or fields.get("category")
        y_col = fields.get("y") or fields.get("value")

        # Group values by category
        grouped = {}
        for row in data:
            category = str(row.get(x_col, "Unknown"))
            value = row.get(y_col)
            if value is not None:
                grouped.setdefault(category, []).append(value)

        fig = go.Figure()

        for category, values in grouped.items():
            color = self._get_consistent_color(category)
            fig.add_trace(
                go.Box(
                    y=values,
                    name=category,
                    marker={"color": color},
                    boxmean="sd",  # Show mean and std dev
                    hovertemplate=f"<b>{category}</b><br>Value: %{{y}}<extra></extra>",
                )
            )

        fig.update_xaxes(title_text=x_col)
        fig.update_yaxes(title_text=y_col)

        return fig

    def _create_violin(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create violin plot."""
        x_col = fields.get("x") or fields.get("category")
        y_col = fields.get("y") or fields.get("value")

        grouped = {}
        for row in data:
            category = str(row.get(x_col, "Unknown"))
            value = row.get(y_col)
            if value is not None:
                grouped.setdefault(category, []).append(value)

        fig = go.Figure()

        for category, values in grouped.items():
            color = self._get_consistent_color(category)
            fig.add_trace(
                go.Violin(
                    y=values,
                    name=category,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=color,
                    opacity=0.6,
                    line={"color": color},
                    hovertemplate=f"<b>{category}</b><br>Value: %{{y}}<extra></extra>",
                )
            )

        fig.update_xaxes(title_text=x_col)
        fig.update_yaxes(title_text=y_col)

        return fig

    # ========== HIERARCHICAL CHARTS ==========

    def _create_treemap(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create treemap for hierarchical data."""
        category_col = fields.get("category") or fields.get("labels") or fields.get("x")
        value_col = fields.get("value") or fields.get("y")

        labels = [
            str(row.get(category_col, "Unknown")) for row in data if category_col in row
        ]
        values = [row.get(value_col, 0) for row in data if value_col in row]
        parents = [""] * len(labels)  # All root level

        # Get consistent colors
        colors = [self._get_consistent_color(label) for label in labels]

        fig = go.Figure()
        fig.add_trace(
            go.Treemap(
                labels=labels,
                values=values,
                parents=parents,
                marker={"colors": colors, "line": {"width": 2}},
                textinfo="label+value+percent parent",
                hovertemplate="<b>%{label}</b><br>Value: %{value}<br>Percent: %{percentParent}<extra></extra>",
            )
        )

        return fig

    def _create_sunburst(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create sunburst chart (circular treemap)."""
        category_col = fields.get("category") or fields.get("labels") or fields.get("x")
        value_col = fields.get("value") or fields.get("y")

        labels = [
            str(row.get(category_col, "Unknown")) for row in data if category_col in row
        ]
        values = [row.get(value_col, 0) for row in data if value_col in row]
        parents = [""] * len(labels)

        colors = [self._get_consistent_color(label) for label in labels]

        fig = go.Figure()
        fig.add_trace(
            go.Sunburst(
                labels=labels,
                values=values,
                parents=parents,
                marker={"colors": colors, "line": {"width": 2}},
                hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
            )
        )

        return fig

    def _create_waterfall(
        self, fields: Dict, data: List[Dict], title: str
    ) -> go.Figure:
        """Create waterfall chart for cumulative changes."""
        x_col = fields.get("x") or fields.get("category")
        y_col = fields.get("y") or fields.get("value")

        x_data = [str(row.get(x_col, "")) for row in data if x_col in row]
        y_data = [row.get(y_col, 0) for row in data if y_col in row]

        # Determine measure types (relative or total)
        measures = ["relative"] * len(y_data)
        if len(measures) > 0:
            measures[-1] = "total"  # Last bar shows total

        fig = go.Figure()
        fig.add_trace(
            go.Waterfall(
                x=x_data,
                y=y_data,
                measure=measures,
                text=[f"{v:+.2f}" if v != 0 else "0" for v in y_data],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#2ca02c"}},
                decreasing={"marker": {"color": "#d62728"}},
                totals={"marker": {"color": "#1f77b4"}},
                hovertemplate="<b>%{x}</b><br>Change: %{y}<extra></extra>",
            )
        )

        fig.update_xaxes(title_text=x_col)
        fig.update_yaxes(title_text=y_col)

        return fig

    # ========== ADVANCED CHARTS ==========

    def _create_heatmap(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create heatmap for correlation/contingency matrices."""
        # Assuming data is in matrix form or needs to be pivoted
        x_col = fields.get("x")
        y_col = fields.get("y")
        z_col = fields.get("z") or fields.get("value")

        # Build matrix
        x_values = sorted(list(set(row.get(x_col) for row in data if x_col in row)))
        y_values = sorted(list(set(row.get(y_col) for row in data if y_col in row)))

        matrix = [[0] * len(x_values) for _ in range(len(y_values))]

        for row in data:
            try:
                x_idx = x_values.index(row.get(x_col))
                y_idx = y_values.index(row.get(y_col))
                matrix[y_idx][x_idx] = row.get(z_col, 0)
            except (ValueError, KeyError):
                continue

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=x_values,
                y=y_values,
                colorscale="Blues",
                hovertemplate="%{x}, %{y}: %{z}<extra></extra>",
                showscale=True,
            )
        )

        fig.update_xaxes(title_text=x_col)
        fig.update_yaxes(title_text=y_col)

        return fig

    def _create_timeline(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create timeline visualization."""
        date_col = fields.get("date") or fields.get("x") or fields.get("time")
        event_col = fields.get("event") or fields.get("y") or fields.get("label")

        dates = [row.get(date_col) for row in data if date_col in row]
        events = [str(row.get(event_col, "Event")) for row in data if event_col in row]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=list(range(len(dates))),
                mode="markers+text",
                marker={"size": 15, "color": "#1f77b4"},
                text=events,
                textposition="top center",
                hovertemplate="<b>%{text}</b><br>%{x}<extra></extra>",
            )
        )

        fig.update_xaxes(title_text=date_col)
        fig.update_yaxes(visible=False)

        return fig

    def _create_map(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create geographic map."""
        lat_col = fields.get("lat") or fields.get("latitude")
        lon_col = fields.get("lon") or fields.get("longitude")
        value_col = fields.get("value") or fields.get("size")

        lats = [row.get(lat_col) for row in data if lat_col in row]
        lons = [row.get(lon_col) for row in data if lon_col in row]
        values = [row.get(value_col, 10) for row in data if value_col in row]

        fig = go.Figure()
        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="markers",
                marker={"size": values, "color": "#1f77b4", "opacity": 0.6},
                hovertemplate="Lat: %{lat}<br>Lon: %{lon}<extra></extra>",
            )
        )

        fig.update_geos(projection_type="natural earth")

        return fig

    def _create_parallel_coordinates(
        self, fields: Dict, data: List[Dict], title: str
    ) -> go.Figure:
        """Create parallel coordinates plot."""
        # Extract all numeric columns
        numeric_cols = []
        for key in fields.values():
            if key and any(isinstance(row.get(key), (int, float)) for row in data):
                numeric_cols.append(key)

        if not numeric_cols:
            return self._create_empty_chart(
                "No numeric columns for parallel coordinates"
            )

        dimensions = []
        for col in numeric_cols:
            values = [row.get(col, 0) for row in data if col in row]
            dimensions.append(dict(label=col, values=values))

        fig = go.Figure()
        fig.add_trace(go.Parcoords(line={"color": "#1f77b4"}, dimensions=dimensions))

        return fig

    def _create_sankey(self, fields: Dict, data: List[Dict], title: str) -> go.Figure:
        """Create Sankey diagram for flow data."""
        source_col = fields.get("source") or fields.get("from")
        target_col = fields.get("target") or fields.get("to")
        value_col = fields.get("value") or fields.get("weight")

        # Build node list
        nodes = list(
            set(
                [str(row.get(source_col, "")) for row in data if source_col in row]
                + [str(row.get(target_col, "")) for row in data if target_col in row]
            )
        )
        node_dict = {node: idx for idx, node in enumerate(nodes)}

        # Build links
        sources = [
            node_dict[str(row.get(source_col, ""))] for row in data if source_col in row
        ]
        targets = [
            node_dict[str(row.get(target_col, ""))] for row in data if target_col in row
        ]
        values = [row.get(value_col, 1) for row in data if value_col in row]

        # Assign colors to nodes
        node_colors = [self._get_consistent_color(node) for node in nodes]

        fig = go.Figure()
        fig.add_trace(
            go.Sankey(
                node={"label": nodes, "color": node_colors, "pad": 15},
                link={"source": sources, "target": targets, "value": values},
            )
        )

        return fig

    # ========== FALLBACK & UTILITY ==========

    def _create_fallback_chart(
        self, chart_type: str, fields: Dict, data: List[Dict], title: str
    ) -> Dict:
        """Fallback to bar chart for unsupported types."""
        print(f"Warning: {chart_type} not implemented, using bar chart fallback")
        fig = self._create_bar(fields, data, title or f"Fallback: {chart_type}")
        self._apply_common_styling(fig, title)
        return {
            "chart_config": fig.to_dict(),
            "summary": f"Fallback visualization for {chart_type}",
            "reason": "Chart type not fully implemented",
            "chart_type": "bar",
        }

    def _create_empty_chart(self, message: str) -> Dict:
        """Create placeholder chart for empty data."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
        fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False})
        return {
            "chart_config": fig.to_dict(),
            "summary": message,
            "reason": "",
            "chart_type": "empty",
        }
