import html
from typing import Dict, List

from bs4 import BeautifulSoup

from apps.visualization.services.chart_recommender import PlotlyChartGenerator


def generate_styled_table(df, limit=20):
    def truncate_with_tooltip(value):
        if isinstance(value, str) and len(value) > limit:
            short = value[:limit] + "..."
            # Escape quotes so HTML stays valid
            safe_full = html.escape(value, quote=True)
            return f'<span title="{safe_full}">{html.escape(short)}</span>'
        elif isinstance(value, str):
            return html.escape(value)
        return value

    for col in df.columns:
        df[col] = df[col].apply(truncate_with_tooltip)

    html_table = df.to_html(
        index=False,
        classes="min-w-full text-xs md:text-sm border-collapse",
        border=0,
        escape=False,
    )

    soup = BeautifulSoup(html_table, "html.parser")

    for tag in soup.find_all(["th", "td"]):
        tag.attrs.pop("style", None)
        tag["class"] = "px-4 py-2 text-gray-800 text-center"

    thead = soup.find("thead")
    if thead:
        thead["class"] = (
            "bg-gradient-to-r from-blue-50 to-purple-50 text-blue-700 "
            "uppercase tracking-wide text-xs font-semibold"
        )

    tbody = soup.find("tbody")
    if tbody:
        tbody["class"] = "divide-y divide-gray-100"

    table = soup.find("table")
    if table:
        table["class"] = (
            "min-w-full overflow-hidden shadow ring-1 ring-black/5 "
            "text-sm bg-white rounded-xl"
        )

    return str(soup)


# def prepare_chart_data(df, viz):
#     x_col = viz["axes"].get("x") or viz["axes"].get("category")
#     y_col = viz["axes"].get("y") or viz["axes"].get("value")
#     is_count = False
#     if not y_col and viz["axes"].get("count"):
#         is_count = True
#
#     agg = viz.get("aggregation")
#
#     if viz.get("filter"):
#         for k, v in viz["filter"].items():
#             df = df[df[k] == v]
#
#     if agg:
#         df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
#         df[y_col] = df[y_col].fillna(0)
#
#         group_by = agg.get("group_by", [x_col])
#         agg_func = agg.get("y", "sum")
#         grouped = df.groupby(group_by, as_index=False)[y_col].agg(agg_func)
#         labels = grouped[group_by[0]].astype(str).tolist()
#         data = grouped[y_col].tolist()
#     elif is_count:
#         labels, data = prepare_histogram_data(df, x_col)
#         labels_cat, data_cat = prepare_histogram_data(df, x_col)
#     else:
#         labels = df[x_col].astype(str).tolist()
#         data = df[y_col].tolist()
#     return labels, data


def prepare_chart_data(df, viz):
    """
    Prepare chart-ready data for Plotly from a single viz dict.
    Returns a uniform dict with 'traces' (list) ready for Plotly.
    """
    chart_type = viz.get("chart_type")
    axes = viz.get("axes", {})
    agg = viz.get("aggregation")
    filt = viz.get("filter")
    group = axes.get("group")
    parent = axes.get("parent")

    x_col = axes.get("x") or axes.get("category")
    y_col = axes.get("y") or axes.get("value")

    # -----------------------------
    # APPLY FILTERS
    # -----------------------------
    if filt:
        for col, condition in filt.items():
            op = condition.get("op")
            val = condition.get("value")
            if op == "==":
                df = df[df[col] == val]
            elif op == "!=":
                df = df[df[col] != val]
            elif op in [">", ">=", "<", "<="]:
                df = df.query(f"{col} {op} @val")

    # -----------------------------
    # APPLY AGGREGATION
    # -----------------------------
    if agg and y_col:
        method = agg.get("method", "sum")
        group_by = agg.get("group_by") or x_col
        grouping_cols = [c for c in [group_by, group] if c]
        df = df.groupby(grouping_cols, as_index=False)[y_col].agg(method)

    # -----------------------------
    # ROUTE CHART TYPES
    # -----------------------------
    traces = []

    if chart_type in ["bar", "column", "line", "scatter", "area"]:
        if group and group in df.columns:
            # grouped series
            for grp in df[group].unique():
                d = df[df[group] == grp]
                trace = {
                    "x": d[x_col].tolist(),
                    "y": d[y_col].tolist(),
                    "name": str(grp),
                    "type": "bar" if chart_type in ["bar", "column"] else "scatter",
                }
                if chart_type in ["line", "area"]:
                    trace["mode"] = "lines+markers"
                    if chart_type == "area":
                        trace["fill"] = "tozeroy"
                traces.append(trace)
        else:
            # single series
            trace = {
                "x": df[x_col].tolist() if x_col in df.columns else [],
                "y": df[y_col].tolist() if y_col in df.columns else [],
                "name": viz.get("title", chart_type.title()),
                "type": "bar" if chart_type in ["bar", "column"] else "scatter",
            }
            if chart_type in ["line", "area"]:
                trace["mode"] = "lines+markers"
                if chart_type == "area":
                    trace["fill"] = "tozeroy"
            traces.append(trace)

    elif chart_type == "pie":
        trace = {
            "labels": df[x_col].tolist() if x_col in df.columns else [],
            "values": df[y_col].tolist() if y_col in df.columns else [],
            "type": "pie",
            "name": viz.get("title", "Pie Chart"),
        }
        traces.append(trace)

    elif chart_type == "histogram":
        trace = {
            "x": df[y_col].dropna().tolist() if y_col in df.columns else [],
            "type": "histogram",
            "name": viz.get("title", "Histogram"),
        }
        traces.append(trace)

    elif chart_type in ["box", "boxplot", "violin"]:
        if group and group in df.columns:
            for grp in df[group].unique():
                trace = {
                    "y": df[df[group] == grp][y_col].dropna().tolist()
                    if y_col in df.columns
                    else [],
                    "name": str(grp),
                    "type": "box" if chart_type in ["box", "boxplot"] else "violin",
                }
                traces.append(trace)
        else:
            trace = {
                "y": df[y_col].dropna().tolist() if y_col in df.columns else [],
                "name": viz.get("title", chart_type.title()),
                "type": "box" if chart_type in ["box", "boxplot"] else "violin",
            }
            traces.append(trace)

    elif chart_type == "treemap":
        trace = {
            "labels": df[x_col].tolist() if x_col in df.columns else [],
            "values": df[y_col].tolist() if y_col in df.columns else [],
            "parents": df[parent].tolist()
            if parent and parent in df.columns
            else [""] * len(df),
            "type": "treemap",
            "name": viz.get("title", "Treemap"),
        }
        traces.append(trace)

    else:
        # fallback bar
        trace = {
            "x": df[x_col].tolist() if x_col in df.columns else [],
            "y": df[y_col].tolist() if y_col in df.columns else [],
            "name": viz.get("title", chart_type.title()),
            "type": "bar",
        }
        traces.append(trace)

    return traces


def prepare_plotly_charts_from_recommendations(
    recommendations: List[Dict], context_id: str = None
) -> List[Dict]:
    generator = PlotlyChartGenerator()
    charts_data = []

    print(recommendations)

    for rec in recommendations:
        chart_type = rec.get("chart_type", "bar")
        fields = rec.get("fields", {})
        aggregated_data = rec.get("aggregated_data", [])
        title = rec.get("title", chart_type.replace("_", " ").title())
        summary = rec.get("summary", "")
        reason = rec.get("reason", "")

        chart_config = generator.generate_chart(
            chart_type=chart_type,
            fields=fields,
            aggregated_data=aggregated_data,
            title=title,
            summary=summary,
            reason=reason,
            context_id=context_id or rec.get("context_id"),
        )

        charts_data.append(chart_config)

    return charts_data
