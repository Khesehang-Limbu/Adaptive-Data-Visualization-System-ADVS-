import re

from core.settings import BASE_DIR

MODEL_PATH = BASE_DIR / "apps/visualization/ml/models/chart_recommend_model.pkl"

CHART_TYPES = [
    "line",
    "area",
    "bar",
    "pie",
    "scatter",
    "histogram",
    "boxplot",
    "heatmap",
    "bubble",
    "grouped_bar",
    "stacked_bar",
    "violin",
    "treemap",
    "donut",
    "waterfall",
    "timeline",
    "map",
    "sunburst",
    "parallel_coordinates",
    "sankey",
]

CHART_FAMILIES = {
    "categorical_numeric": [
        "bar",
        "grouped_bar",
        "boxplot",
        "violin",
        "line",
        "treemap",
    ],
    "numeric_numeric": ["scatter", "bubble", "hexbin", "line"],
    "datetime_numeric": ["line", "area", "bar", "timeline"],
    "categorical_categorical": ["stacked_bar", "grouped_bar", "heatmap", "sunburst"],
    "univariate_numeric": ["histogram", "boxplot", "violin"],
    "univariate_categorical": ["pie", "donut"],
    "univariate_datetime": ["timeline"],
    "geospatial": ["map"],
}

LLM_API_ENDPOINT = "http://localhost:11434/api/generate"
LLM_PROMPT = """
You are a data visualization expert. Your task is to evaluate chart recommendations and suggest improvements.

CRITICAL: Return ONLY a single valid JSON object. No markdown, no code blocks, no explanatory text before or after.

## Response Structure
{{
  "are_initial_recommendations_appropriate": boolean,
  "feedback": "string (1-3 sentences explaining your assessment)",
  "visualizations": [
    {{
      "chart_type": "string",
      "axes": {{
            "x": <COL_NAME>,
            "y": <COL_NAME>,
            "group": <COL> (optional),
            "parent": <COL> (optional, for treemap),
            "x_type": <"datetime"|"category"|"numeric"> (optional)
      }},
      "aggregation": {{
            "method": <AGGREGATION_FUNCTION_NAME>,
            "group_by": axes["y"],
            "notes": <summary on what this aggregation logic means>
      }} or null,
      "filter": {{}} or null,
      "title": "string",
      "summary": "string (a short analysis about the chart that can be obtained from the recommendation)"
    }}
  ]
}}


## Aggregation Rules
1. Always use quoted strings: "sum", "count", "avg", "min", "max", "median"
2. Only aggregate when grouping by a dimension (e.g., sum sales by region)
3. Set to null when showing individual data points (scatter plots, raw time series)
4. Common patterns:
   - Bar charts: Usually need aggregation (e.g., {{"y": "sum"}})
   - Line charts: May or may not need aggregation depending on granularity
   - Scatter plots: Typically no aggregation (null)
5. If the data contains more than one row per x-axis group, always specify an aggregation function (such as 'sum', 'count', or 'average'). Do not assume the data is already aggregated

## Validation Checklist
Before finalizing your response, verify:
- [ ] Chart type matches data types (categorical vs. continuous)
- [ ] X-axis is appropriate for the chart type (categorical for bar, temporal for line)
- [ ] Y-axis is numeric when aggregation is applied
- [ ] Aggregation is quoted and valid: "sum", "count", "avg", "min", "max", "median"
- [ ] No trailing commas in JSON
- [ ] Title clearly describes what the chart shows
- [ ] Summary explains the insight or purpose

## Decision Framework
1. **Assess appropriateness**: Do current recommendations match user's analytical goal?
2. **Check data compatibility**: Do recommended axes exist in the dataset?
3. **Evaluate chart choice**: Is the chart type optimal for the data types and question?
4. **Verify aggregation logic**: Does the aggregation make sense for the analysis?

## Edge Cases
- If user context is vague, keep initial recommendations but note in feedback
- If dataset has < 3 rows, avoid aggregation
- If time column exists and user asks about trends, prioritize line charts
- If user asks for comparison, prioritize bar charts
- Maximum 3 visualization suggestions to avoid overwhelming the user

---

User context: "{user_context}"

System recommendations:
{current_recs}

Dataset preview (columns and first few rows):
{data_head}
"""

LLM_PROMPT_V1 = """
You are a senior data visualization expert. Your role is to choose the *best analytical visualizations* based on:
1) the user's intent
2) the structure of the dataset
3) correctness rules
4) modern analytics best practices

CRITICAL: Return ONLY a single valid JSON object. No markdown, code fences, or text before/after.
## Validation Checklist
Before finalizing your response, verify:
- [ ] Chart type matches data types (categorical vs. continuous)
- [ ] X-axis is appropriate for the chart type (categorical for bar, temporal for line)
- [ ] Y-axis is numeric when aggregation is applied
- [ ] Aggregation is quoted and valid: "sum", "count", "avg", "min", "max", "median"
- [ ] No trailing commas in JSON
- [ ] Title clearly describes what the chart shows
- [ ] Summary explains the insight or purpose

---------------------------
## Step 1 — Infer User Intent
Infer the user’s analytical goal from their message.
Possible intent types include (but are not limited to):
- comparison (rank categories, compare values)
- composition (market share, parts of a whole)
- distribution (spread of numeric data)
- trend (change over time)
- correlation (relationship between variables)
- outlier detection
- segmentation / grouping

Write your inference internally and use it when designing visualizations.

---------------------------
## Step 2 — Select Best Chart Types Based on Intent
**Univariate Charts:**
- histogram: Distribution of a single numeric variable
- pie: Proportions of low-cardinality categorical (2-8 categories)
- donut: Alternative to pie with emphasis on total
- boxplot: Distribution summary with quartiles (can be univariate or by category)

**Bivariate Charts:**
- bar: Categorical vs numeric comparison
- line: Temporal trends (datetime × numeric)
- area: Cumulative temporal trends
- scatter: Relationship between two numeric variables
- bubble: Three dimensions (x, y, size)

**Multi-Categorical Charts:**
- grouped_bar: Multiple categories with numeric values, grouped comparison
- stacked_bar: Multiple categories with numeric values, showing composition
- violin: Distribution density across categories

**Hierarchical & Advanced:**
- treemap: High-cardinality hierarchical data with size encoding
- sunburst: Circular hierarchical visualization
- waterfall: Sequential cumulative changes
- heatmap: Correlation matrix or contingency table (numeric × numeric or cat × cat)
- timeline: Event sequence visualization (datetime only)
- map: Geographic data with lat/lon coordinates
- parallel_coordinates: Multivariate numeric comparison (4+ variables)
- sankey: Flow between categorical hierarchies

---------------------------
## Step 3 — Apply Correct Aggregation Logic
1. Aggregation only when grouping a categorical axis.
2. Valid values: "sum", "count", "avg", "min", "max", "median"
3. Raw numeric vs. aggregated:
   - Scatter → aggregation: null
   - Raw time series with a timestamp column → aggregation: null
4. If >1 row per category: MUST aggregate.

---------------------------
## Step 4 — Response Schema

{{
  "visualizations": [
    {{
      "chart_type": "string",
      "axes": {{
            "x": <COL_NAME>,
            "y": <COL_NAME>,
            "group": <COL> (optional),
            "parent": <COL> (optional, for treemap),
            "x_type": <"datetime"|"category"|"numeric"> (optional)
      }},
      "aggregation": {{
            "method": <AGGREGATION_FUNCTION_NAME>,
            "group_by": axes["y"],
            "notes": <summary on what this aggregation logic means>
      }} or null,
      "filter": {{}} or null,
      "title": "string",
      "summary": "string (a short analysis about the chart that can be obtained from the recommendation)"
    }}
  ]
}}

Rules:
- Max 3 visualizations.
- Choose diverse charts unless the user explicitly wants one type.
- Titles must be descriptive.
- Summaries must reflect analytical purpose.
- JSON must be clean: no trailing commas.

---------------------------
User context:
"{user_context}"

Dataset metadata:
{metadata}
"""

LLM_PROMPT_V2 = """You are a data visualization expert. Analyze the user's request and dataset to recommend charts.

RETURN ONLY VALID JSON. No explanations, markdown, or code fences.

═══════════════════════════════════════════
ALLOWED CHARTS
═══════════════════════════════════════════
histogram, pie, donut, bar, line, area, scatter, bubble, boxplot, violin, grouped_bar, stacked_bar, treemap, sunburst, waterfall, heatmap, timeline, map, parallel_coordinates, sankey

═══════════════════════════════════════════
QUICK SELECTION GUIDE
═══════════════════════════════════════════
• 1 numeric → histogram or boxplot
• 1 categorical (2-8 items) → pie or bar
• 1 categorical (9+ items) → bar or treemap
• datetime + numeric → line or area
• categorical + numeric → bar or boxplot
• 2 numerics → scatter
• 2 categoricals + numeric → grouped_bar or stacked_bar

Use profiler's "suitable_charts" as starting point.

═══════════════════════════════════════════
AGGREGATION
═══════════════════════════════════════════
Methods: sum, count, avg, min, max, median

When to aggregate:
• Multiple rows per category → YES
• Scatter/histogram → NO (use null)
• Time series → depends on granularity

═══════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════
{{
  "visualizations": [
    {{
      "chart_type": "<chart from allowed list>",
      "axes": {{"x": "<col>", "y": "<col>", "group": "<col>"}},
      "aggregation": {{"method": "<method>", "group_by": ["<col>"]}} or null,
      "title": "<clear title>",
      "summary": "<what insight this shows>"
    }}
  ]
}}

Max 2 charts. Use columns from metadata.

═══════════════════════════════════════════
INPUT
═══════════════════════════════════════════
User: {user_context}

Available columns: {column_summary}

Profiler suggests: {chart_hints}

Generate JSON now:
"""


LLM_DATASET_SUMMARY_PROMPT_STREAM = """
You are a data summarization assistant. You are given the metadata of a CSV file. Your task is to generate a **concise one-paragraph summary** of the dataset using the metadata, describing:

- The main structure of the data (columns, types, and general content)
- Any notable patterns, trends, or key statistics
- The purpose or usage insights that can be inferred from the dataset
- No need to highlight the words with markups. Just text is necessary.

Make the summary informative, concise, and easy to read.

Here is the entire metadata of the csv data:
{dataset_summary}

Now summarize as instructed above. Do not include any text aside from the summary of the data. No leading sentences before the summary. Strictly follow these rules.
"""

LLM_MINISTRAL = "ministral-3:8b"
LLM_QWEN = "qwen3-vl:4b"
LLM_LLAMA = "llama3.1:latest"

MAX_LLM_RETRIES = 3
LLM_TIMEOUT = 30
MAX_GROUPS_FOR_AGGREGATION = 100
SAMPLE_SIZE_FOR_FEATURES = 10000
UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
CURRENCY_PATTERN = re.compile(
    r"^(?:\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?(?:USD|EUR|GBP))$"
)
URL_PATTERN = re.compile(r"http[s]?://|www\.", re.IGNORECASE)


ALL_FEATURE_KEYS = [
    # Dataset-level (8 features)
    "dataset_rows",
    "dataset_cols",
    "dataset_density",
    "dataset_missing_pct",
    "dataset_is_wide",
    "numeric_col_ratio",
    "categorical_col_ratio",
    "datetime_col_ratio",
    # Column 1 basic (5 features)
    "col1_is_numeric",
    "col1_is_categorical",
    "col1_is_datetime",
    "col1_is_text",
    "col1_missing_pct",
    # Column 1 cardinality (8 features)
    "col1_unique_count",
    "col1_unique_ratio",
    "col1_is_binary",
    "col1_is_low_card",
    "col1_is_medium_card",
    "col1_is_high_card",
    "col1_suitable_pie",
    "col1_suitable_bar",
    # Column 1 distribution (8 features - numeric only)
    "col1_skewness",
    "col1_kurtosis",
    "col1_is_normal",
    "col1_has_outliers",
    "col1_outlier_ratio",
    "col1_coef_variation",
    "col1_zero_ratio",
    "col1_negative_ratio",
    # Column 1 balance (6 features - categorical only)
    "col1_entropy",
    "col1_gini",
    "col1_is_balanced",
    "col1_is_imbalanced",
    "col1_dominant_pct",
    "col1_is_ordinal",
    # Column 1 time series (3 features - datetime only)
    "col1_is_sequential",
    "col1_has_gaps",
    "col1_time_span_days",
    # Column 1 flags (3 features)
    "col1_is_currency",
    "col1_is_percentage",
    "col1_is_geospatial",
    # Column 2 basic (6 features)
    "col2_is_numeric",
    "col2_is_categorical",
    "col2_is_datetime",
    "col2_is_text",
    "col2_missing_pct",
    "col2_unique_count",
    "col2_unique_ratio",
    # Pairwise features (13 features)
    "has_col2",
    "pair_correlation",
    "pair_correlation_abs",
    "pair_is_strongly_correlated",
    "pair_type_num_num",
    "pair_type_cat_num",
    "pair_type_cat_cat",
    "pair_type_time_num",
    "pair_anova_f",
    "pair_anova_p",
    "pair_anova_significant",
    "pair_category_count",
    "pair_chi2",
    "pair_chi2_p",
    "pair_cramers_v",
    "pair_is_time_series",
]
