import pandas as pd
import numpy as np

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

        if col not in cleaned_df.columns:
            print(f"Warning: Column '{col}' not found. Skipping action: {action}")
            continue

        try:
            if method == "Impute with median":
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    median_val = cleaned_df[col].median()
                    cleaned_df[col].fillna(median_val, inplace=True)
                    print(f"Applied: Imputed missing values in '{col}' with median ({median_val}).")
                else:
                    print(f"Skipping: Impute with median not applicable for non-numeric column '{col}'.")
            elif method == "Impute with mean":
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    mean_val = cleaned_df[col].mean()
                    cleaned_df[col].fillna(mean_val, inplace=True)
                    print(f"Applied: Imputed missing values in '{col}' with mean ({mean_val}).")
                else:
                    print(f"Skipping: Impute with mean not applicable for non-numeric column '{col}'.")
            elif method == "Impute with mode":
                mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None
                if mode_val is not None:
                    cleaned_df[col].fillna(mode_val, inplace=True)
                    print(f"Applied: Imputed missing values in '{col}' with mode ({mode_val}).")
                else:
                    print(f"Skipping: Could not impute mode for '{col}' (no mode found).")
            elif method == "Drop rows with missing values":
                initial_rows = len(cleaned_df)
                cleaned_df.dropna(subset=[col], inplace=True)
                rows_dropped = initial_rows - len(cleaned_df)
                print(f"Applied: Dropped {rows_dropped} rows with missing values in '{col}'.")
            elif method == "Convert to numeric":
                # Errors='coerce' will turn non-convertible values into NaN
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                print(f"Applied: Converted '{col}' to numeric. Non-convertible values are now NaN.")
            elif method == "Remove leading/trailing spaces":
                if pd.api.types.is_string_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                    print(f"Applied: Removed leading/trailing spaces from '{col}'.")
                else:
                    print(f"Skipping: Remove spaces not applicable for non-string column '{col}'.")
            elif method == "Convert to lowercase":
                if pd.api.types.is_string_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].astype(str).str.lower()
                    print(f"Applied: Converted '{col}' to lowercase.")
                else:
                    print(f"Skipping: Convert to lowercase not applicable for non-string column '{col}'.")
            elif method == "Remove duplicate rows":
                initial_rows = len(cleaned_df)
                cleaned_df.drop_duplicates(subset=[col], inplace=True)
                rows_dropped = initial_rows - len(cleaned_df)
                print(f"Applied: Removed {rows_dropped} duplicate rows based on column '{col}'.")
            elif method == "Cap outliers (IQR)":
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    original_count = cleaned_df[col].count()
                    cleaned_df[col] = np.where(cleaned_df[col] < lower_bound, lower_bound, cleaned_df[col])
                    cleaned_df[col] = np.where(cleaned_df[col] > upper_bound, upper_bound, cleaned_df[col])
                    capped_count = original_count - cleaned_df[col].count() # This won't work for capping, but rather for dropping
                    print(f"Applied: Capped outliers in '{col}' using IQR method (lower: {lower_bound}, upper: {upper_bound}).")
                else:
                    print(f"Skipping: Cap outliers not applicable for non-numeric column '{col}'.")
            else:
                print(f"Warning: Unknown cleaning method '{method}' for column '{col}'. Skipping.")
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
        "",
        "# Load your dataset (modify path as needed)",
        "# df = pd.read_csv('your_dataset.csv')",
        "",
        "def clean_data(df: pd.DataFrame) -> pd.DataFrame:",
        "    cleaned_df = df.copy()",
        ""
    ]

    for action in actions:
        col = action.get("column")
        method = action.get("method")
        # params = action.get("parameters", {}) # Currently not used explicitly in script, but can be added

        script_lines.append(f"    # Action for column: '{col}' - Method: '{method}'")
        if method == "Impute with median":
            script_lines.append(f"    if pd.api.types.is_numeric_dtype(cleaned_df['{col}']):")
            script_lines.append(f"        median_val = cleaned_df['{col}'].median()")
            script_lines.append(f"        cleaned_df['{col}'].fillna(median_val, inplace=True)")
        elif method == "Impute with mean":
            script_lines.append(f"    if pd.api.types.is_numeric_dtype(cleaned_df['{col}']):")
            script_lines.append(f"        mean_val = cleaned_df['{col}'].mean()")
            script_lines.append(f"        cleaned_df['{col}'].fillna(mean_val, inplace=True)")
        elif method == "Impute with mode":
            script_lines.append(f"    mode_val = cleaned_df['{col}'].mode()[0] if not cleaned_df['{col}'].mode().empty else None")
            script_lines.append(f"    if mode_val is not None:")
            script_lines.append(f"        cleaned_df['{col}'].fillna(mode_val, inplace=True)")
        elif method == "Drop rows with missing values":
            script_lines.append(f"    cleaned_df.dropna(subset=['{col}'], inplace=True)")
        elif method == "Convert to numeric":
            script_lines.append(f"    cleaned_df['{col}'] = pd.to_numeric(cleaned_df['{col}'], errors='coerce')")
        elif method == "Remove leading/trailing spaces":
            script_lines.append(f"    if pd.api.types.is_string_dtype(cleaned_df['{col}']):")
            script_lines.append(f"        cleaned_df['{col}'] = cleaned_df['{col}'].astype(str).str.strip()")
        elif method == "Convert to lowercase":
            script_lines.append(f"    if pd.api.types.is_string_dtype(cleaned_df['{col}']):")
            script_lines.append(f"        cleaned_df['{col}'] = cleaned_df['{col}'].astype(str).str.lower()")
        elif method == "Remove duplicate rows":
            script_lines.append(f"    cleaned_df.drop_duplicates(subset=['{col}'], inplace=True)")
        elif method == "Cap outliers (IQR)":
            script_lines.append(f"    if pd.api.types.is_numeric_dtype(cleaned_df['{col}']):")
            script_lines.append(f"        Q1 = cleaned_df['{col}'].quantile(0.25)")
            script_lines.append(f"        Q3 = cleaned_df['{col}'].quantile(0.75)")
            script_lines.append(f"        IQR = Q3 - Q1")
            script_lines.append(f"        lower_bound = Q1 - 1.5 * IQR")
            script_lines.append(f"        upper_bound = Q3 + 1.5 * IQR")
            script_lines.append(f"        cleaned_df['{col}'] = np.where(cleaned_df['{col}'] < lower_bound, lower_bound, cleaned_df['{col}'])")
            script_lines.append(f"        cleaned_df['{col}'] = np.where(cleaned_df['{col}'] > upper_bound, upper_bound, cleaned_df['{col}'])")
        script_lines.append("") # Add an empty line for readability

    script_lines.append("    return cleaned_df")
    script_lines.append("")
    script_lines.append("# Example usage:")
    script_lines.append("# df_cleaned = clean_data(df.copy())")
    script_lines.append("# print(df_cleaned.head())")

    return "\\n".join(script_lines)