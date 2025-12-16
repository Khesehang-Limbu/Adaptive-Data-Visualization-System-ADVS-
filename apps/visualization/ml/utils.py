import datetime
import json
import time
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import requests

from apps.visualization.ml.constants import (
    LLM_API_ENDPOINT,
    LLM_LLAMA,
    LLM_PROMPT_V2,
    MAX_LLM_RETRIES,
    MODEL_PATH,
)


@lru_cache(maxsize=1)
def load_chart_recommend_model():
    return joblib.load(MODEL_PATH)


def format_lightweight_prompt(user_query: str, profiler_metadata: dict) -> str:
    """
    Create minimal prompt for local LLMs.

    Parameters
    ----------
    user_query : str
        User's question
    profiler_metadata : dict
        Full profiler.profile() output

    Returns
    -------
    str
        Compact prompt (<1000 tokens)
    """
    # Extract only essential info
    columns = profiler_metadata.get("columns", [])

    # Simplified column summary
    col_summary = []
    for col in columns:
        name = col["name"]
        dtype = col["dtype"]

        # Add key properties
        extras = []
        if dtype == "categorical":
            card = col.get("cardinality", {})
            extras.append(f"{card.get('unique_count', 0)} categories")
        elif dtype == "datetime":
            ts = col.get("time_series", {})
            if ts.get("is_sequential"):
                extras.append("sequential")

        col_info = f"{name} ({dtype})"
        if extras:
            col_info += f" - {', '.join(extras)}"
        col_summary.append(col_info)

    column_summary_str = "\n".join(col_summary)

    # Extract chart hints
    hints = profiler_metadata.get("chart_hints", {})
    suitable = hints.get("suitable_charts", [])[:5]  # Top 5 only
    reasons = hints.get("reasons", {})

    chart_hints_str = ""
    if suitable:
        chart_hints_str = f"Recommended: {', '.join(suitable)}\n"
        # Add top reason
        if reasons and suitable:
            top_chart = suitable[0]
            chart_hints_str += f"Best: {top_chart} - {reasons.get(top_chart, '')}"
    else:
        chart_hints_str = "No specific recommendations"

    # Format prompt
    prompt = LLM_PROMPT_V2.format(
        user_context=user_query,
        column_summary=column_summary_str,
        chart_hints=chart_hints_str,
    )

    return prompt


def ask_llm_for_viz(context, metadata, df):
    prompt = format_lightweight_prompt(context, metadata)

    data_complexity = len(df.columns) * min(len(df), 20)

    if data_complexity < 100:
        read_timeout = 60
    elif data_complexity < 500:
        read_timeout = 120
    else:
        read_timeout = 180

    for attempt in range(MAX_LLM_RETRIES):
        try:
            response = requests.post(
                LLM_API_ENDPOINT,
                json={"model": LLM_LLAMA, "prompt": prompt, "stream": False},
            )

            json_response = json.loads(response.json().get("response"))

            response.raise_for_status()

            json_response = json.loads(response.json().get("response"))

            if "visualizations" not in json_response:
                print(
                    f"LLM response missing 'visualizations' field (attempt {attempt + 1})"
                )
                continue

            visualizations = json_response.get("visualizations", [])
            is_valid, error_msg = validate_visualization(
                visualizations, list(df.columns)
            )

            if is_valid:
                if "feedback" not in json_response:
                    json_response[
                        "feedback"
                    ] = "Recommendations generated successfully."

                return json_response
            else:
                print(f"LLM validation failed (attempt {attempt + 1}): {error_msg}")

        except requests.Timeout:
            print(f"LLM timeout after {read_timeout}s (attempt {attempt + 1})")
        except requests.RequestException as e:
            print(f"LLM request error (attempt {attempt + 1}): {e}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"LLM response parsing error (attempt {attempt + 1}): {e}")
        except Exception as e:
            print(f"Unexpected error in LLM call (attempt {attempt + 1}): {e}")

        if attempt < MAX_LLM_RETRIES - 1:
            wait_time = 2**attempt  # 1s, 2s, 4s
            print(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

    print(f"All {MAX_LLM_RETRIES} LLM attempts failed, using fallback recommendations")
    return None


def validate_visualization(viz, dataset_columns):
    """Enhanced validation with better error messages"""
    if not viz or not isinstance(viz, list):
        return False, "Visualizations must be a non-empty list"

    for i, v in enumerate(viz):
        if not isinstance(v, dict):
            return False, f"Visualization {i} is not a dictionary"

        # Check required fields
        if "chart_type" not in v:
            return False, f"Visualization {i} missing 'chart_type'"

        if "axes" not in v:
            return False, f"Visualization {i} missing 'axes'"

        axes = v["axes"]

        # Validate axis columns
        if axes.get("x") and axes["x"] not in dataset_columns:
            return False, f"X-axis column '{axes['x']}' not found in dataset"

        if axes.get("y") and axes["y"] not in dataset_columns:
            return False, f"Y-axis column '{axes['y']}' not found in dataset"

        if axes.get("category") and axes["category"] not in dataset_columns:
            return False, f"Category column '{axes['category']}' not found in dataset"

        # Validate aggregation
        if v.get("aggregation"):
            valid_aggs = ["sum", "count", "avg", "mean", "min", "max", "median"]
            agg = v["aggregation"]
            if isinstance(agg, dict):
                for key, val in agg.items():
                    if val not in valid_aggs:
                        return (
                            False,
                            f"Invalid aggregation '{val}' in visualization {i}",
                        )
            elif isinstance(agg, str) and agg not in valid_aggs:
                return False, f"Invalid aggregation '{agg}' in visualization {i}"

    return True, "Valid"


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]

    elif isinstance(obj, tuple):
        return [convert_numpy(v) for v in obj]  # tuples → lists (JSON-safe)

    elif isinstance(obj, np.ndarray):
        return convert_numpy(obj.tolist())

    elif isinstance(obj, (np.integer,)):
        return int(obj)

    elif isinstance(obj, (np.floating,)):
        return float(obj)

    elif isinstance(obj, (np.bool_,)):
        return bool(obj)

    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    elif obj is pd.NaT:
        return None

    else:
        return obj
