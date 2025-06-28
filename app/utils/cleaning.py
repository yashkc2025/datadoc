import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from datetime import datetime

# Ensure NLTK stopwords are downloaded
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")


def apply_one_hot_encoding(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Applies One-Hot Encoding to a specified column.
    Handles potential issues if the column is not suitable.
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found for One-Hot Encoding.")
        return df

    if pd.api.types.is_numeric_dtype(
        df[column]
    ) or pd.api.types.is_datetime64_any_dtype(df[column]):
        print(
            f"Warning: Column '{column}' is {df[column].dtype}. One-Hot Encoding might not be appropriate."
        )
        return df

    try:
        # Create a OneHotEncoder instance
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        # Fit and transform the column. Reshape to 2D array as required by OneHotEncoder.
        encoded_data = encoder.fit_transform(df[[column]])

        # Create a DataFrame from the encoded data with meaningful column names
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoder.get_feature_names_out([column]),
            index=df.index,
        )

        # Drop the original column and concatenate the new one-hot encoded columns
        df_encoded = pd.concat([df.drop(columns=[column]), encoded_df], axis=1)
        print(
            f"Applied One-Hot Encoding to column '{column}'. New columns created: {', '.join(encoded_df.columns)}"
        )
        return df_encoded
    except Exception as e:
        print(f"Error applying One-Hot Encoding to column '{column}': {e}")
        return df


def apply_label_encoding(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Applies Label Encoding to a specified column.
    Handles potential issues if the column is not suitable.
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found for Label Encoding.")
        return df

    if pd.api.types.is_numeric_dtype(
        df[column]
    ) or pd.api.types.is_datetime64_any_dtype(df[column]):
        print(
            f"Warning: Column '{column}' is {df[column].dtype}. Label Encoding might not be appropriate."
        )
        return df

    try:
        # Create a LabelEncoder instance
        encoder = LabelEncoder()

        # Fit and transform the column. Handle NaNs if present.
        series_no_nan = df[column].astype(str).fillna("__NaN__")  # Temporary fill NaNs
        encoded_series = encoder.fit_transform(series_no_nan)

        df_encoded = df.copy()
        df_encoded[column] = encoded_series
        print(f"Applied Label Encoding to column '{column}'.")
        return df_encoded
    except Exception as e:
        print(f"Error applying Label Encoding to column '{column}': {e}")
        return df


def remove_stopwords(df: pd.DataFrame, column: str, language="english") -> pd.DataFrame:
    """
    Removes common stop words from a text column.
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found for stop word removal.")
        return df
    if not (
        pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_string_dtype(df[column])
    ):
        print(
            f"Skipping: Stop word removal not applicable for non-string column '{column}'."
        )
        return df

    stop_words = set(stopwords.words(language))
    cleaned_df = df.copy()

    def filter_words(text):
        if pd.isna(text):
            return text
        words = str(text).lower().split()
        filtered_words = [word for word in words if word not in stop_words]
        return " ".join(filtered_words)

    cleaned_df[column] = cleaned_df[column].apply(filter_words)
    print(f"Applied: Removed stop words from '{column}'.")
    return cleaned_df


def apply_regex_replace(
    df: pd.DataFrame, column: str, pattern: str, replacement: str
) -> pd.DataFrame:
    """
    Applies a regex find and replace to a string column.
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found for regex replace.")
        return df
    if not (
        pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_string_dtype(df[column])
    ):
        print(
            f"Skipping: Regex replace not applicable for non-string column '{column}'."
        )
        return df

    cleaned_df = df.copy()
    try:
        cleaned_df[column] = (
            cleaned_df[column]
            .astype(str)
            .str.replace(pattern, replacement, regex=True, na=False)
        )
        print(
            f"Applied: Regex replace (pattern='{pattern}', replacement='{replacement}') to '{column}'."
        )
        return cleaned_df
    except re.error as e:
        print(f"Error: Invalid regex pattern '{pattern}' for column '{column}': {e}")
        return df
    except Exception as e:
        print(f"Error applying regex replace to column '{column}': {e}")
        return df


def standardize_datetime_format(
    df: pd.DataFrame, column: str, target_format: str = "%Y-%m-%d %H:%M:%S"
) -> pd.DataFrame:
    """
    Converts a datetime column to a standardized string format.
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found for date format standardization.")
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        print(
            f"Skipping: Date format standardization not applicable for non-datetime column '{column}'."
        )
        return df

    cleaned_df = df.copy()
    cleaned_df[column] = cleaned_df[column].dt.strftime(target_format)
    print(f"Applied: Standardized date format of '{column}' to '{target_format}'.")
    return cleaned_df


def extract_date_part(df: pd.DataFrame, column: str, part: str) -> pd.DataFrame:
    """
    Extracts a specific part (year, month, day, hour, minute, second) from a datetime column
    and creates a new column.
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found for date part extraction.")
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        print(
            f"Skipping: Date part extraction not applicable for non-datetime column '{column}'."
        )
        return df

    cleaned_df = df.copy()
    new_col_name = f"{column}_{part}"
    try:
        if part == "year":
            cleaned_df[new_col_name] = cleaned_df[column].dt.year
        elif part == "month":
            cleaned_df[new_col_name] = cleaned_df[column].dt.month
        elif part == "day":
            cleaned_df[new_col_name] = cleaned_df[column].dt.day
        elif part == "hour":
            cleaned_df[new_col_name] = cleaned_df[column].dt.hour
        elif part == "minute":
            cleaned_df[new_col_name] = cleaned_df[column].dt.minute
        elif part == "second":
            cleaned_df[new_col_name] = cleaned_df[column].dt.second
        else:
            print(f"Warning: Unknown date part '{part}'. Skipping extraction.")
            return df
        print(
            f"Applied: Extracted '{part}' from '{column}' into new column '{new_col_name}'."
        )
        return cleaned_df
    except Exception as e:
        print(f"Error extracting date part '{part}' from column '{column}': {e}")
        return df


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Removes rows where the specified numeric column has outliers based on IQR method.
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found for outlier removal.")
        return df
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(
            f"Skipping: Outlier removal not applicable for non-numeric column '{column}'."
        )
        return df

    cleaned_df = df.copy()
    Q1 = cleaned_df[column].quantile(0.25)
    Q3 = cleaned_df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df[
        (cleaned_df[column] >= lower_bound) | (cleaned_df[column].isnull())
    ]
    cleaned_df = cleaned_df[
        (cleaned_df[column] <= upper_bound) | (cleaned_df[column].isnull())
    ]
    rows_removed = initial_rows - len(cleaned_df)
    print(
        f"Applied: Removed {rows_removed} outliers from '{column}' using IQR method (bounds: [{lower_bound}, {upper_bound}])."
    )
    return cleaned_df


def apply_log_transform(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Applies a natural logarithm transformation to a numeric column.
    Handles non-positive values by replacing them with NaN.
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found for log transform.")
        return df
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(
            f"Skipping: Log transform not applicable for non-numeric column '{column}'."
        )
        return df

    cleaned_df = df.copy()
    # Add a small constant to handle zero values, or replace non-positive with NaN
    # Here, we'll replace non-positive values with NaN, as log(0) is undefined and log(negative) is complex.
    cleaned_df[column] = np.log(
        cleaned_df[column].where(cleaned_df[column] > 0, np.nan)
    )
    print(
        f"Applied: Log transformation to '{column}'. Non-positive values converted to NaN."
    )
    return cleaned_df


def apply_cleaning_actions(df: pd.DataFrame, actions: list[dict]) -> pd.DataFrame:
    """
    Applies a list of cleaning actions to the DataFrame.
    Each action is a dictionary with 'column', 'method', and 'parameters'.
    """
    cleaned_df = df.copy()
    for action in actions:
        col = action.get("column")
        method = action.get("method")
        params = action.get("parameters", {})

        # Special handling for actions that might change column names or are dataset-wide
        if method in ["One-Hot Encode", "Extract Date Part"]:
            # These methods handle column existence internally or create new columns
            pass
        elif col not in cleaned_df.columns:
            print(
                f"Warning: Column '{col}' not found for action '{method}'. Skipping action: {action}"
            )
            continue

        try:
            if method == "Impute with median":
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    median_val = cleaned_df[col].median()
                    cleaned_df[col].fillna(median_val, inplace=True)
                    print(
                        f"Applied: Imputed missing values in '{col}' with median ({median_val})."
                    )
                else:
                    print(
                        f"Skipping: Impute with median not applicable for non-numeric column '{col}'."
                    )
            elif method == "Impute with mean":
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    mean_val = cleaned_df[col].mean()
                    cleaned_df[col].fillna(mean_val, inplace=True)
                    print(
                        f"Applied: Imputed missing values in '{col}' with mean ({mean_val})."
                    )
                else:
                    print(
                        f"Skipping: Impute with mean not applicable for non-numeric column '{col}'."
                    )
            elif method == "Impute with mode":
                mode_val = (
                    cleaned_df[col].mode()[0]
                    if not cleaned_df[col].mode().empty
                    else None
                )
                if mode_val is not None:
                    cleaned_df[col].fillna(mode_val, inplace=True)
                    print(
                        f"Applied: Imputed missing values in '{col}' with mode ({mode_val})."
                    )
                else:
                    print(
                        f"Skipping: Could not impute mode for '{col}' (no mode found)."
                    )
            elif method == "Drop rows with missing values":
                initial_rows = len(cleaned_df)
                cleaned_df.dropna(subset=[col], inplace=True)
                rows_dropped = initial_rows - len(cleaned_df)
                print(
                    f"Applied: Dropped {rows_dropped} rows with missing values in '{col}'."
                )
            elif method == "Convert to numeric":
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
                print(
                    f"Applied: Converted '{col}' to numeric. Non-convertible values are now NaN."
                )
            elif method == "Convert to datetime":
                cleaned_df[col] = pd.to_datetime(
                    cleaned_df[col], errors="coerce", infer_datetime_format=True
                )
                print(
                    f"Applied: Converted '{col}' to datetime. Invalid dates are now NaT."
                )
            elif method == "One-Hot Encode":
                cleaned_df = apply_one_hot_encoding(cleaned_df, col)
            elif method == "Label Encode":
                cleaned_df = apply_label_encoding(cleaned_df, col)
            elif method == "Remove Stopwords":
                cleaned_df = remove_stopwords(cleaned_df, col)
            elif method == "Regex Replace":
                cleaned_df = apply_regex_replace(
                    cleaned_df, col, params.get("pattern"), params.get("replacement")
                )
            elif method == "Standardize Date Format":
                cleaned_df = standardize_datetime_format(
                    cleaned_df, col, params.get("format")
                )
            elif method == "Extract Date Part":
                cleaned_df = extract_date_part(cleaned_df, col, params.get("part"))
            elif method == "Remove Outliers (IQR)":
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            elif method == "Log Transform":
                cleaned_df = apply_log_transform(cleaned_df, col)
            elif method == "Remove leading/trailing spaces":
                if pd.api.types.is_string_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                    print(f"Applied: Removed leading/trailing spaces from '{col}'.")
                else:
                    print(
                        f"Skipping: Remove spaces not applicable for non-string column '{col}'."
                    )
            elif method == "Convert to lowercase":
                if pd.api.types.is_string_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].astype(str).str.lower()
                    print(f"Applied: Converted '{col}' to lowercase.")
                else:
                    print(
                        f"Skipping: Convert to lowercase not applicable for non-string column '{col}'."
                    )
            elif method == "Remove duplicate rows":
                initial_rows = len(cleaned_df)
                cleaned_df.drop_duplicates(subset=[col], inplace=True)
                rows_dropped = initial_rows - len(cleaned_df)
                print(
                    f"Applied: Removed {rows_dropped} duplicate rows based on column '{col}'."
                )
            elif method == "Cap outliers (IQR)":
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    original_count = cleaned_df[col].count()
                    cleaned_df[col] = np.where(
                        cleaned_df[col] < lower_bound, lower_bound, cleaned_df[col]
                    )
                    cleaned_df[col] = np.where(
                        cleaned_df[col] > upper_bound, upper_bound, cleaned_df[col]
                    )
                    capped_count = (
                        original_count - cleaned_df[col].count()
                    )  # This won't work for capping, but rather for dropping
                    print(
                        f"Applied: Capped outliers in '{col}' using IQR method (lower: {lower_bound}, upper: {upper_bound})."
                    )
                else:
                    print(
                        f"Skipping: Cap outliers not applicable for non-numeric column '{col}'."
                    )
            else:
                print(
                    f"Warning: Unknown cleaning method '{method}' for column '{col}'. Skipping."
                )
        except Exception as e:
            print(f"Error applying action '{method}' to column '{col}': {e}")
    return cleaned_df


def generate_script(actions: list[dict]) -> str:
    """
    Generates a Python script based on the applied cleaning actions.
    """
    script_lines = [
        "import pandas as pd",
        "import numpy as np",
        "import re",
        "import nltk",
        "from nltk.corpus import stopwords",
        "from sklearn.preprocessing import OneHotEncoder, LabelEncoder",
        "",
        "# Ensure NLTK stopwords are downloaded (run once if needed)",
        "# try:",
        "#     nltk.data.find('corpora/stopwords')",
        "# except nltk.downloader.DownloadError:",
        "#     nltk.download('stopwords')",
        "",
        "# Load your dataset (modify path as needed)",
        "# df = pd.read_csv('your_dataset.csv')",
        "",
        "def clean_data(df: pd.DataFrame) -> pd.DataFrame:",
        "    cleaned_df = df.copy()",
        "",
    ]

    for action in actions:
        col = action.get("column")
        method = action.get("method")
        params = action.get("parameters", {})

        script_lines.append(f"    # Action for column: '{col}' - Method: '{method}'")
        if method == "Impute with median":
            script_lines.append(
                f"    if pd.api.types.is_numeric_dtype(cleaned_df['{col}']):"
            )
            script_lines.append(f"        median_val = cleaned_df['{col}'].median()")
            script_lines.append(
                f"        cleaned_df['{col}'].fillna(median_val, inplace=True)"
            )
        elif method == "Impute with mean":
            script_lines.append(
                f"    if pd.api.types.is_numeric_dtype(cleaned_df['{col}']):"
            )
            script_lines.append(f"        mean_val = cleaned_df['{col}'].mean()")
            script_lines.append(
                f"        cleaned_df['{col}'].fillna(mean_val, inplace=True)"
            )
        elif method == "Impute with mode":
            script_lines.append(
                f"    mode_val = cleaned_df['{col}'].mode()[0] if not cleaned_df['{col}'].mode().empty else None"
            )
            script_lines.append(f"    if mode_val is not None:")
            script_lines.append(
                f"        cleaned_df['{col}'].fillna(mode_val, inplace=True)"
            )
        elif method == "Drop rows with missing values":
            script_lines.append(
                f"    cleaned_df.dropna(subset=['{col}'], inplace=True)"
            )
        elif method == "Convert to numeric":
            script_lines.append(
                f"    cleaned_df['{col}'] = pd.to_numeric(cleaned_df['{col}'], errors='coerce')"
            )
        elif method == "Convert to datetime":
            script_lines.append(
                f"    cleaned_df['{col}'] = pd.to_datetime(cleaned_df['{col}'], errors='coerce', infer_datetime_format=True)"
            )
        elif method == "One-Hot Encode":
            script_lines.append(f"    # One-Hot Encoding for '{col}'")
            script_lines.append(
                f"    encoder_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)"
            )
            script_lines.append(
                f"    encoded_data_ohe = encoder_ohe.fit_transform(cleaned_df[['{col}']])"
            )
            script_lines.append(
                f"    encoded_df_ohe = pd.DataFrame(encoded_data_ohe, columns=encoder_ohe.get_feature_names_out(['{col}']), index=cleaned_df.index)"
            )
            script_lines.append(
                f"    cleaned_df = pd.concat([cleaned_df.drop(columns=['{col}']), encoded_df_ohe], axis=1)"
            )
        elif method == "Label Encode":
            script_lines.append(f"    # Label Encoding for '{col}'")
            script_lines.append(f"    encoder_le = LabelEncoder()")
            script_lines.append(
                f"    series_no_nan_le = cleaned_df['{col}'].astype(str).fillna('__NaN__')"
            )
            script_lines.append(
                f"    cleaned_df['{col}'] = encoder_le.fit_transform(series_no_nan_le)"
            )
        elif method == "Remove Stopwords":
            script_lines.append(f"    stop_words = set(stopwords.words('english'))")
            script_lines.append(
                f"    cleaned_df['{col}'] = cleaned_df['{col}'].astype(str).apply(lambda text: ' '.join([word for word in str(text).lower().split() if word not in stop_words]))"
            )
        elif method == "Regex Replace":
            pattern_str = repr(
                params.get("pattern")
            )  # Use repr for string representation in script
            replacement_str = repr(params.get("replacement"))
            script_lines.append(
                f"    cleaned_df['{col}'] = cleaned_df['{col}'].astype(str).str.replace({pattern_str}, {replacement_str}, regex=True, na=False)"
            )
        elif method == "Standardize Date Format":
            target_format = repr(params.get("format"))
            script_lines.append(
                f"    cleaned_df['{col}'] = pd.to_datetime(cleaned_df['{col}'], errors='coerce', infer_datetime_format=True).dt.strftime({target_format})"
            )
        elif method == "Extract Date Part":
            part = params.get("part")
            new_col_name = f"{col}_{part}"
            if part == "year":
                script_lines.append(
                    f"    cleaned_df['{new_col_name}'] = pd.to_datetime(cleaned_df['{col}'], errors='coerce', infer_datetime_format=True).dt.year"
                )
            elif part == "month":
                script_lines.append(
                    f"    cleaned_df['{new_col_name}'] = pd.to_datetime(cleaned_df['{col}'], errors='coerce', infer_datetime_format=True).dt.month"
                )
            elif part == "day":
                script_lines.append(
                    f"    cleaned_df['{new_col_name}'] = pd.to_datetime(cleaned_df['{col}'], errors='coerce', infer_datetime_format=True).dt.day"
                )
            elif part == "hour":
                script_lines.append(
                    f"    cleaned_df['{new_col_name}'] = pd.to_datetime(cleaned_df['{col}'], errors='coerce', infer_datetime_format=True).dt.hour"
                )
            elif part == "minute":
                script_lines.append(
                    f"    cleaned_df['{new_col_name}'] = pd.to_datetime(cleaned_df['{col}'], errors='coerce', infer_datetime_format=True).dt.minute"
                )
            elif part == "second":
                script_lines.append(
                    f"    cleaned_df['{new_col_name}'] = pd.to_datetime(cleaned_df['{col}'], errors='coerce', infer_datetime_format=True).dt.second"
                )
        elif method == "Remove Outliers (IQR)":
            script_lines.append(f"    Q1 = cleaned_df['{col}'].quantile(0.25)")
            script_lines.append(f"    Q3 = cleaned_df['{col}'].quantile(0.75)")
            script_lines.append(f"    IQR = Q3 - Q1")
            script_lines.append(f"    lower_bound = Q1 - 1.5 * IQR")
            script_lines.append(f"    upper_bound = Q3 + 1.5 * IQR")
            script_lines.append(
                f"    cleaned_df = cleaned_df[ (cleaned_df['{col}'] >= lower_bound) | (cleaned_df['{col}'].isnull()) ]"
            )
            script_lines.append(
                f"    cleaned_df = cleaned_df[ (cleaned_df['{col}'] <= upper_bound) | (cleaned_df['{col}'].isnull()) ]"
            )
        elif method == "Log Transform":
            script_lines.append(
                f"    cleaned_df['{col}'] = np.log(cleaned_df['{col}'].where(cleaned_df['{col}'] > 0, np.nan))"
            )
        elif method == "Remove leading/trailing spaces":
            script_lines.append(
                f"    if pd.api.types.is_string_dtype(cleaned_df['{col}']):"
            )
            script_lines.append(
                f"        cleaned_df['{col}'] = cleaned_df['{col}'].astype(str).str.strip()"
            )
        elif method == "Convert to lowercase":
            script_lines.append(
                f"    if pd.api.types.is_string_dtype(cleaned_df['{col}']):"
            )
            script_lines.append(
                f"        cleaned_df['{col}'] = cleaned_df['{col}'].astype(str).str.lower()"
            )
        elif method == "Remove duplicate rows":
            script_lines.append(
                f"    cleaned_df.drop_duplicates(subset=['{col}'], inplace=True)"
            )
        elif method == "Cap outliers (IQR)":
            script_lines.append(
                f"    if pd.api.types.is_numeric_dtype(cleaned_df['{col}']):"
            )
            script_lines.append(f"        Q1 = cleaned_df['{col}'].quantile(0.25)")
            script_lines.append(f"        Q3 = cleaned_df['{col}'].quantile(0.75)")
            script_lines.append(f"        IQR = Q3 - Q1")
            script_lines.append(f"        lower_bound = Q1 - 1.5 * IQR")
            script_lines.append(f"        upper_bound = Q3 + 1.5 * IQR")
            script_lines.append(
                f"        cleaned_df['{col}'] = np.where(cleaned_df['{col}'] < lower_bound, lower_bound, cleaned_df['{col}'])"
            )
            script_lines.append(
                f"        cleaned_df['{col}'] = np.where(cleaned_df['{col}'] > upper_bound, upper_bound, cleaned_df['{col}'])"
            )
        script_lines.append("")  # Add an empty line for readability

    script_lines.append("    return cleaned_df")
    script_lines.append("")
    script_lines.append("# Example usage:")
    script_lines.append("# df_cleaned = clean_data(df.copy())")
    script_lines.append("# print(df_cleaned.head())")

    return "\\n".join(script_lines)


def validate_data(df: pd.DataFrame, rules: list[dict]) -> dict:
    """
    Applies a list of custom validation rules to the DataFrame.
    Each rule is a dictionary with 'column', 'rule_type', and 'value'.
    Returns a dictionary of validation results.
    """
    validation_results = {}
    for rule in rules:
        col = rule.get("column")
        rule_type = rule.get("rule_type")
        value = rule.get("value")
        rule_id = rule.get("id")  # Use the unique ID generated in the UI

        if col not in df.columns:
            validation_results[rule_id] = {
                "rule_description": f"Column '{col}' not found for rule '{rule_type}'",
                "violating_rows_count": 0,
                "violating_rows_sample": pd.DataFrame(),
            }
            continue

        violations = pd.Series(
            [False] * len(df), index=df.index
        )  # Initialize all to False

        try:
            # Handle null checks
            if rule_type == "is_null":
                violations = df[col].isnull()
            elif rule_type == "is_not_null":
                violations = df[col].notnull()

            # Numeric comparisons
            elif pd.api.types.is_numeric_dtype(df[col]) and isinstance(
                value, (int, float)
            ):
                if rule_type == "greater_than":
                    violations = df[col] <= value
                elif rule_type == "less_than":
                    violations = df[col] >= value
                elif rule_type == "equals":
                    violations = df[col] != value
                elif rule_type == "not_equals":
                    violations = df[col] == value

            # String comparisons
            elif (
                pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])
            ) and isinstance(value, str):
                df_col_str = (
                    df[col].astype(str).fillna("")
                )  # Convert to string for comparison
                if rule_type == "equals":
                    violations = df_col_str != value
                elif rule_type == "not_equals":
                    violations = df_col_str == value
                elif rule_type == "contains_string":
                    violations = ~df_col_str.str.contains(value, na=False)
                elif rule_type == "regex_match":
                    # Ensure value is treated as a raw string for regex
                    try:
                        violations = ~df_col_str.str.contains(
                            value, regex=True, na=False
                        )
                    except re.error as e:
                        print(f"Invalid regex for column '{col}': {value} - {e}")
                        violations = pd.Series(
                            [False] * len(df), index=df.index
                        )  # No violations if regex is invalid
            else:
                print(
                    f"Warning: Rule type '{rule_type}' or value type mismatch for column '{col}'. Skipping."
                )
                continue  # Skip if rule type or value type is not applicable

        except Exception as e:
            print(
                f"Error applying validation rule '{rule_type}' on column '{col}': {e}"
            )
            violations = pd.Series(
                [False] * len(df), index=df.index
            )  # Treat as no violations on error

        violating_rows = df[violations]
        validation_results[rule_id] = {
            "rule_description": f"{col} {rule_type} {value}",
            "violating_rows_count": len(violating_rows),
            "violating_rows_sample": violating_rows.head(
                5
            ),  # Show first 5 violating rows
        }
    return validation_results
