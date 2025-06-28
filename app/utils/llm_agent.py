import json
import pandas as pd  # Import pandas for dtype checks


def suggest_cleaning(column_profile: dict, model=None) -> list[dict]:
    """
    Simulates LLM suggestions for cleaning actions based on column profile.
    In a real application, this would involve calling a local or remote LLM.
    """
    column_name = column_profile.get("column", "Unknown Column")
    dtype = column_profile.get("dtype", "unknown")
    missing_pct = column_profile.get("missing_pct", 0)
    outliers_count = column_profile.get("outliers_count", 0)
    skewness = column_profile.get("skewness", 0)
    unique_count = column_profile.get("unique_count", 0)
    total_count = column_profile.get("total_count", 1)
    avg_word_count = column_profile.get("avg_word_count", 0)

    # Get the type suggestion from the profile if available
    type_suggestion = column_profile.get("type_suggestion")

    suggestions = []

    # --- Missing Values ---
    if missing_pct > 0:
        if "float" in dtype or "int" in dtype:
            suggestions.append(
                {
                    "method": "Impute with median",
                    "pro": f"Keeps all rows; robust to outliers for numeric column '{column_name}'.",
                    "con": "May hide real data issues; can reduce variance.",
                }
            )
            suggestions.append(
                {
                    "method": "Impute with mean",
                    "pro": f"Keeps all rows; simple for numeric column '{column_name}'.",
                    "con": "Sensitive to outliers; can reduce variance.",
                }
            )
            suggestions.append(
                {
                    "method": "Drop rows with missing values",
                    "pro": f"Ensures clean training data for '{column_name}'; no imputation bias.",
                    "con": f"Loss of data ({missing_pct}% of rows).",
                }
            )
        else:  # For categorical/object types
            suggestions.append(
                {
                    "method": "Impute with mode",
                    "pro": f"Keeps all rows; suitable for categorical column '{column_name}'.",
                    "con": "Can introduce bias if mode is very dominant.",
                }
            )
            suggestions.append(
                {
                    "method": "Drop rows with missing values",
                    "pro": f"Ensures clean training data for '{column_name}'.",
                    "con": f"Loss of data ({missing_pct}% of rows).",
                }
            )

    # --- Type Conversion (General) ---
    # Only suggest if it's currently object/string and a specific suggestion exists from profiling
    if ("object" in dtype or "string" in dtype) and type_suggestion == "numeric":
        suggestions.append(
            {
                "method": "Convert to numeric",
                "pro": f"Enables numerical analysis for '{column_name}'; handles mixed types.",
                "con": "Non-convertible values will become NaN, potentially increasing missing data.",
            }
        )
    elif ("object" in dtype or "string" in dtype) and type_suggestion == "datetime":
        suggestions.append(
            {
                "method": "Convert to datetime",
                "pro": f"Enables date/time-based analysis for '{column_name}'.",
                "con": "Invalid date strings will become NaT, potentially increasing missing data.",
            }
        )

    # --- Text Cleaning Specific ---
    if "object" in dtype or "string" in dtype:
        if avg_word_count > 2:  # Heuristic: if it looks like a sentence/phrase
            suggestions.append(
                {
                    "method": "Remove Stopwords",
                    "pro": f"Removes common words from '{column_name}'; useful for text analysis (e.g., topic modeling).",
                    "con": "Can alter sentence meaning; not always suitable for all text tasks.",
                }
            )

        # Suggest regex replace for general text standardization
        suggestions.append(
            {
                "method": "Regex Replace",
                "pro": f"Highly flexible for complex text standardization, cleaning special characters, or extracting patterns.",
                "con": "Requires knowledge of regex; can lead to unintended changes if pattern is incorrect.",
            }
        )

        suggestions.append(
            {
                "method": "Remove leading/trailing spaces",
                "pro": f"Cleans up string data in '{column_name}'; ensures consistent comparisons.",
                "con": "No significant cons.",
            }
        )
        suggestions.append(
            {
                "method": "Convert to lowercase",
                "pro": f"Standardizes text data in '{column_name}'; reduces unique categories.",
                "con": "Loss of original casing information.",
            }
        )

        # Categorical Encoding Suggestions based on cardinality (retained)
        if unique_count > 1 and unique_count < min(
            50, total_count * 0.5
        ):  # Not too many unique values
            suggestions.append(
                {
                    "method": "One-Hot Encode",
                    "pro": f"Converts '{column_name}' into numerical format suitable for many ML models; avoids ordinal bias.",
                    "con": "Can create many new columns, increasing dataset dimensionality, especially for high cardinality.",
                }
            )
            suggestions.append(
                {
                    "method": "Label Encode",
                    "pro": f"Converts '{column_name}' into a single numerical column, preserving dimensionality.",
                    "con": "Introduces arbitrary ordinal relationship between categories, which can mislead ML models.",
                }
            )

    # --- Date/Time Specific (from pandas dtype) ---
    if "datetime" in dtype:  # More robust check for datetime dtype after conversion
        suggestions.append(
            {
                "method": "Standardize Date Format",
                "pro": f"Ensures consistent date string representation in '{column_name}'.",
                "con": "Converts datetime objects back to strings; may lose precision.",
            }
        )
        suggestions.append(
            {
                "method": "Extract Date Part",
                "pro": f"Creates new features (e.g., year, month) from '{column_name}' for time-series analysis.",
                "con": "Increases column count; requires datetime column to be clean.",
            }
        )

    # --- Outlier Handling & Transformation ---
    if "float" in dtype or "int" in dtype:
        if outliers_count > 0:
            suggestions.append(
                {
                    "method": "Cap outliers (IQR)",
                    "pro": f"Reduces the impact of extreme values in '{column_name}' without removing rows.",
                    "con": "Distorts the original distribution; values are artificially limited.",
                }
            )
            suggestions.append(
                {
                    "method": "Remove Outliers (IQR)",
                    "pro": f"Completely removes extreme values from '{column_name}'.",
                    "con": f"Leads to loss of data (potentially {outliers_count} rows).",
                }
            )
        if skewness and abs(skewness) > 1:  # High skewness
            suggestions.append(
                {
                    "method": "Log Transform",
                    "pro": f"Reduces skewness in '{column_name}' and can normalize data distribution, beneficial for some ML models.",
                    "con": "Cannot be applied to zero or negative values; alters the interpretability of the original scale.",
                }
            )

    # --- Duplicate Rows (Dataset-wide, but can be suggested per column if column is an identifier) ---
    if (
        unique_count < total_count
        and column_profile.get("total_count", 0) > 0
        and column_profile.get("missing_count", 0) == 0
    ):
        if column_profile["total_count"] - column_profile["unique_count"] > 0:
            suggestions.append(
                {
                    "method": "Remove duplicate rows",
                    "pro": f"Ensures unique entries for '{column_name}' if it's an identifier; cleans up redundant data.",
                    "con": "Loss of data if duplicates are legitimate; should be used carefully.",
                }
            )

    # --- Generic Fallback ---
    if not suggestions:
        suggestions.append(
            {
                "method": "Review manually",
                "pro": f"Provides granular control over data quality for '{column_name}'.",
                "con": "Time-consuming for large datasets.",
            }
        )

    return suggestions
