import json

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

    suggestions = []

    # Handle missing values
    if missing_pct > 0:
        if "float" in dtype or "int" in dtype:
            suggestions.append({
                "method": "Impute with median",
                "pro": f"Keeps all rows; robust to outliers for numeric column '{column_name}'.",
                "con": "May hide real data issues; can reduce variance."
            })
            suggestions.append({
                "method": "Impute with mean",
                "pro": f"Keeps all rows; simple for numeric column '{column_name}'.",
                "con": "Sensitive to outliers; can reduce variance."
            })
            suggestions.append({
                "method": "Drop rows with missing values",
                "pro": f"Ensures clean training data for '{column_name}'; no imputation bias.",
                "con": f"Loss of data ({missing_pct}% of rows)."
            })
        else: # For categorical/object types
            suggestions.append({
                "method": "Impute with mode",
                "pro": f"Keeps all rows; suitable for categorical column '{column_name}'.",
                "con": "Can introduce bias if mode is very dominant."
            })
            suggestions.append({
                "method": "Drop rows with missing values",
                "pro": f"Ensures clean training data for '{column_name}'.",
                "con": f"Loss of data ({missing_pct}% of rows)."
            })

    # Handle numeric type conversion issues (simulated)
    if ("object" in dtype or "string" in dtype) and any(char.isdigit() for char in column_name) and total_count > 0 and (unique_count / total_count > 0.5):
        # Heuristic: if column name might imply numeric (e.g., 'price_str', 'age_text')
        # and has many unique values (might contain non-numeric chars)
        suggestions.append({
            "method": "Convert to numeric",
            "pro": f"Enables numerical analysis for '{column_name}'; handles mixed types.",
            "con": "Non-convertible values will become NaN, potentially increasing missing data."
        })

    # Handle text/string specific issues
    if "object" in dtype or "string" in dtype:
        suggestions.append({
            "method": "Remove leading/trailing spaces",
            "pro": f"Cleans up string data in '{column_name}'; ensures consistent comparisons.",
            "con": "No significant cons."
        })
        suggestions.append({
            "method": "Convert to lowercase",
            "pro": f"Standardizes text data in '{column_name}'; reduces unique categories.",
            "con": "Loss of original casing information."
        })

    # Handle outliers in numeric columns
    if "float" in dtype or "int" in dtype:
        if outliers_count > 0:
            suggestions.append({
                "method": "Cap outliers (IQR)",
                "pro": f"Reduces the impact of extreme values in '{column_name}' without removing rows.",
                "con": "Distorts the original distribution; values are artificially limited."
            })

    # Handle duplicates (if applicable to a single column's unique values)
    # This is more a dataset-level action but can be suggested per column if column is an identifier
    if unique_count < total_count and column_profile.get("total_count", 0) > 0 and column_profile.get("missing_count", 0) == 0:
        if column_profile["total_count"] - column_profile["unique_count"] > 0:
            suggestions.append({
                "method": "Remove duplicate rows",
                "pro": f"Ensures unique entries for '{column_name}' if it's an identifier; cleans up redundant data.",
                "con": "Loss of data if duplicates are legitimate; should be used carefully."
            })


    # If no specific suggestions, provide a generic one
    if not suggestions:
        suggestions.append({
            "method": "Review manually",
            "pro": f"Provides granular control over data quality for '{column_name}'.",
            "con": "Time-consuming for large datasets."
        })

    return suggestions