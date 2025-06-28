import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import IsolationForest


def infer_data_type_suggestions(df: pd.DataFrame) -> dict:
    """
    Infers potential data type conversions for columns that are currently 'object' or 'string'.
    Returns a dictionary of {column_name: suggested_type}.
    """
    suggestions = {}
    for col in df.columns:
        series = df[col]

        # Check if the column is currently an object/string type
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Try converting to numeric
            try:
                # Use regex to replace common non-numeric characters before conversion
                cleaned_series = (
                    series.astype(str)
                    .str.replace(r"[$,%€]", "", regex=True)
                    .str.strip()
                )
                # Check if it can be purely numeric (ignoring NaN)
                if (
                    pd.to_numeric(cleaned_series, errors="coerce").notnull().sum()
                    / len(series.dropna())
                    > 0.9
                ):
                    # If more than 90% of non-null values can be converted to numeric
                    suggestions[col] = "numeric"
                    continue  # Move to next column if numeric conversion is strong

            except Exception:
                pass  # Not numeric

            # Try converting to datetime
            try:
                # Check if it can be purely datetime (ignoring NaN)
                if (
                    pd.to_datetime(series, errors="coerce", dayfirst=False)
                    .notnull()
                    .sum()
                    / len(series.dropna())
                    > 0.9
                ):
                    # If more than 90% of non-null values can be converted to datetime
                    suggestions[col] = "datetime"
                    continue  # Move to next column if datetime conversion is strong
            except Exception:
                pass  # Not datetime

    return suggestions


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generates a profile for each column in the DataFrame.
    """
    profile = {}
    for col in df.columns:
        series = df[col]
        col_profile = {}

        # Basic statistics
        col_profile["column"] = col
        col_profile["dtype"] = str(series.dtype)
        col_profile["unique_count"] = series.nunique()
        col_profile["total_count"] = len(series)
        col_profile["missing_count"] = series.isnull().sum()
        col_profile["missing_pct"] = (
            round((series.isnull().sum() / len(series)) * 100, 2)
            if len(series) > 0
            else 0.0
        )

        if pd.api.types.is_numeric_dtype(series):
            col_profile["mean"] = (
                round(series.mean(), 2) if series.notnull().any() else None
            )
            col_profile["median"] = (
                round(series.median(), 2) if series.notnull().any() else None
            )
            col_profile["std"] = (
                round(series.std(), 2) if series.notnull().any() else None
            )
            col_profile["min"] = (
                round(series.min(), 2) if series.notnull().any() else None
            )
            col_profile["max"] = (
                round(series.max(), 2) if series.notnull().any() else None
            )
            col_profile["skewness"] = (
                round(skew(series.dropna()), 2) if series.notnull().any() else None
            )
            col_profile["kurtosis"] = (
                round(kurtosis(series.dropna()), 2) if series.notnull().any() else None
            )

            # Outlier detection using Isolation Forest
            outliers = detect_outliers(series.dropna().to_frame())
            col_profile["outliers_count"] = len(outliers)
            # col_profile["outlier_indices"] = outliers.tolist() # Can be large, maybe don't store in profile dict
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(
            series
        ):
            col_profile["most_frequent"] = (
                series.mode().iloc[0] if not series.mode().empty else None
            )
            col_profile["top_5_values"] = series.value_counts().nlargest(5).to_dict()
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_profile["first_date"] = (
                series.min().strftime("%Y-%m-%d") if series.notnull().any() else None
            )
            col_profile["last_date"] = (
                series.max().strftime("%Y-%m-%d") if series.notnull().any() else None
            )

        profile[col] = col_profile
    return profile


def detect_outliers(df: pd.DataFrame) -> np.ndarray:
    """
    Detects outliers in a DataFrame using Isolation Forest.
    Assumes numerical data.
    """
    if df.empty or df.isnull().all().all():
        return np.array([])

    # IsolationForest needs at least 2 samples if n_estimators is default (100)
    # Adjusting contamination if there are too few non-null data points
    n_samples = df.shape[0]
    if n_samples < 2:
        return np.array([])  # Not enough data for outlier detection

    # Heuristic for contamination: 0.01 for small datasets, 0.1 for larger, max 0.5
    contamination_val = min(0.1, max(0.01, (df.isnull().sum().sum() + 1) / n_samples))

    try:
        model = IsolationForest(random_state=42, contamination=contamination_val)
        model.fit(df)
        predictions = model.predict(df)
        outliers_indices = df.index[predictions == -1].values
        return outliers_indices
    except Exception as e:
        # Handle cases where IsolationForest might fail (e.g., all values are same)
        print(f"Warning: Could not detect outliers for column. Error: {e}")
        return np.array([])


def generate_statistics(df: pd.DataFrame) -> dict:
    """
    Generates overall dataset statistics.
    """
    stats = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "total_missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }
    return stats
