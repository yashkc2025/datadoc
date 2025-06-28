import streamlit as st
import pandas as pd
from app.utils.llm_agent import suggest_cleaning
from app.utils.cleaning import (
    apply_cleaning_actions,
    validate_data,
    apply_one_hot_encoding,
    apply_label_encoding,
    remove_stopwords,
    apply_regex_replace,
    standardize_datetime_format,
    extract_date_part,
    remove_outliers_iqr,
    apply_log_transform,  # New imports
)
import json  # For handling cleaning actions
import re  # For regex validation
import nltk  # For NLTK download check
from datetime import datetime

st.set_page_config(page_title="Clean Dataset", page_icon="🧹", layout="wide")

st.title("🧹 Clean Dataset")
st.markdown(
    "Apply intelligent cleaning suggestions and define custom validation rules for your data."
)

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    st.info(
        "Downloading NLTK stopwords (first time setup)... This might take a moment."
    )
    nltk.download("stopwords")
    st.success("NLTK stopwords downloaded successfully!")
    st.experimental_rerun()


if "current_df" not in st.session_state or not st.session_state["current_df"]:
    st.warning(
        "No dataset loaded. Please go to the 'Upload' page to upload or select a dataset, and then 'Profile' it."
    )
elif "column_profiles" not in st.session_state:
    st.warning("Please generate the dataset profile on the 'Profile' page first.")
else:
    df = pd.read_json(st.session_state["current_df"])
    column_profiles = st.session_state["column_profiles"]
    dataset_name = st.session_state.get("current_dataset_name", "Unnamed Dataset")
    st.subheader(f"Cleaning & Validation for: {dataset_name}")

    # Initialize session state for cleaning actions and validation rules
    if "cleaning_actions" not in st.session_state:
        st.session_state["cleaning_actions"] = []
    if "validation_rules" not in st.session_state:
        st.session_state["validation_rules"] = []
    if "validation_results" not in st.session_state:
        st.session_state["validation_results"] = {}

    st.markdown("---")
    st.subheader("AI-Simulated Cleaning Suggestions")
    selected_column_clean = st.selectbox(
        "Select a column to get cleaning suggestions for:",
        df.columns,
        key="select_col_clean",
    )

    if selected_column_clean:
        col_profile = column_profiles.get(selected_column_clean, {})
        st.write(f"**Current Profile for '{selected_column_clean}':**")
        st.json(col_profile)

        # Type conversion suggestion
        if "type_suggestion" in col_profile and col_profile["type_suggestion"]:
            st.info(
                f"**Type Conversion Suggestion:** This column could be converted to **{col_profile['type_suggestion']}**."
            )
            if st.button(
                f"Apply Type Conversion to {selected_column_clean} (to {col_profile['type_suggestion']})",
                key=f"apply_type_conversion_{selected_column_clean}",
            ):
                action = {
                    "column": selected_column_clean,
                    "method": f"Convert to {col_profile['type_suggestion']}",
                    "parameters": {},
                }
                st.session_state["cleaning_actions"].append(action)
                st.success(
                    f"Action 'Convert to {col_profile['type_suggestion']}' added for column '{selected_column_clean}'."
                )
                st.rerun()

        # Categorical Encoding Options (Direct UI)
        if (
            col_profile.get("dtype") in ["object", "string"]
            and col_profile.get("unique_count", 0) > 1
        ):
            st.info(
                f"**Categorical Encoding Options:** Column '{selected_column_clean}' is categorical."
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    f"Apply One-Hot Encoding to {selected_column_clean}",
                    key=f"apply_ohe_{selected_column_clean}",
                ):
                    action = {
                        "column": selected_column_clean,
                        "method": "One-Hot Encode",
                        "parameters": {},
                    }
                    st.session_state["cleaning_actions"].append(action)
                    st.success(
                        f"Action 'One-Hot Encode' added for column '{selected_column_clean}'."
                    )
                    st.rerun()
            with col2:
                if st.button(
                    f"Apply Label Encoding to {selected_column_clean}",
                    key=f"apply_le_{selected_column_clean}",
                ):
                    action = {
                        "column": selected_column_clean,
                        "method": "Label Encode",
                        "parameters": {},
                    }
                    st.session_state["cleaning_actions"].append(action)
                    st.success(
                        f"Action 'Label Encode' added for column '{selected_column_clean}'."
                    )
                    st.rerun()

        # New: Text Cleaning Options (Direct UI)
        if col_profile.get("dtype") in ["object", "string"]:
            st.info(
                f"**Text Cleaning Options:** Column '{selected_column_clean}' is text-based."
            )
            if st.button(
                f"Remove Stop Words from {selected_column_clean}",
                key=f"remove_stopwords_{selected_column_clean}",
            ):
                action = {
                    "column": selected_column_clean,
                    "method": "Remove Stopwords",
                    "parameters": {},
                }
                st.session_state["cleaning_actions"].append(action)
                st.success(
                    f"Action 'Remove Stopwords' added for column '{selected_column_clean}'."
                )
                st.rerun()

            with st.expander(f"Apply Regex Find & Replace in {selected_column_clean}"):
                regex_pattern = st.text_input(
                    "Regex Pattern:", key=f"regex_pattern_{selected_column_clean}"
                )
                regex_replacement = st.text_input(
                    "Replacement String:",
                    key=f"regex_replacement_{selected_column_clean}",
                )
                if st.button(
                    "Add Regex Replace Action",
                    key=f"add_regex_replace_{selected_column_clean}",
                ):
                    if regex_pattern:
                        action = {
                            "column": selected_column_clean,
                            "method": "Regex Replace",
                            "parameters": {
                                "pattern": regex_pattern,
                                "replacement": regex_replacement,
                            },
                        }
                        st.session_state["cleaning_actions"].append(action)
                        st.success(
                            f"Action 'Regex Replace' added for column '{selected_column_clean}'."
                        )
                        st.rerun()
                    else:
                        st.warning("Please provide a regex pattern.")

        # New: Date/Time Specific Options (Direct UI)
        if (
            col_profile.get("dtype") == "datetime64[ns]"
        ):  # Check for inferred datetime type
            st.info(
                f"**Date/Time Options:** Column '{selected_column_clean}' is a datetime type."
            )
            target_date_format = st.text_input(
                "Target Date Format (e.g., %Y-%m-%d):",
                value="%Y-%m-%d",
                key=f"date_format_{selected_column_clean}",
            )
            if st.button(
                f"Standardize Date Format for {selected_column_clean}",
                key=f"standardize_date_format_{selected_column_clean}",
            ):
                action = {
                    "column": selected_column_clean,
                    "method": "Standardize Date Format",
                    "parameters": {"format": target_date_format},
                }
                st.session_state["cleaning_actions"].append(action)
                st.success(
                    f"Action 'Standardize Date Format' added for column '{selected_column_clean}'."
                )
                st.rerun()

            date_part_to_extract = st.selectbox(
                "Extract Date Part:",
                ["None", "Year", "Month", "Day", "Hour", "Minute", "Second"],
                key=f"extract_date_part_{selected_column_clean}",
            )
            if date_part_to_extract != "None" and st.button(
                f"Extract {date_part_to_extract} from {selected_column_clean}",
                key=f"extract_date_part_btn_{selected_column_clean}",
            ):
                action = {
                    "column": selected_column_clean,
                    "method": "Extract Date Part",
                    "parameters": {"part": date_part_to_extract.lower()},
                }
                st.session_state["cleaning_actions"].append(action)
                st.success(
                    f"Action 'Extract {date_part_to_extract}' added for column '{selected_column_clean}'."
                )
                st.rerun()

        # New: Outlier Handling Options (Direct UI)
        if col_profile.get("dtype") in ["int64", "float64"]:  # Check for numeric type
            st.info(
                f"**Outlier Handling & Transformation Options:** Column '{selected_column_clean}' is numeric."
            )
            col_outlier1, col_outlier2 = st.columns(2)
            with col_outlier1:
                if st.button(
                    f"Remove Outliers (IQR) from {selected_column_clean}",
                    key=f"remove_outliers_iqr_{selected_column_clean}",
                ):
                    action = {
                        "column": selected_column_clean,
                        "method": "Remove Outliers (IQR)",
                        "parameters": {},
                    }
                    st.session_state["cleaning_actions"].append(action)
                    st.success(
                        f"Action 'Remove Outliers (IQR)' added for column '{selected_column_clean}'."
                    )
                    st.rerun()
            with col_outlier2:
                if st.button(
                    f"Apply Log Transform to {selected_column_clean}",
                    key=f"apply_log_transform_{selected_column_clean}",
                ):
                    action = {
                        "column": selected_column_clean,
                        "method": "Log Transform",
                        "parameters": {},
                    }
                    st.session_state["cleaning_actions"].append(action)
                    st.success(
                        f"Action 'Log Transform' added for column '{selected_column_clean}'."
                    )
                    st.rerun()

        if st.button(
            f"Get General AI-Simulated Suggestions for {selected_column_clean}",
            key="get_suggestions_btn",
        ):
            with st.spinner("Getting AI-simulated suggestions..."):
                suggestions = suggest_cleaning(col_profile, model="simulated_model")
                st.session_state["current_suggestions"] = suggestions
                st.success("Suggestions loaded!")

        if (
            "current_suggestions" in st.session_state
            and st.session_state["current_suggestions"]
        ):
            st.markdown("---")
            st.write("##### AI-Simulated Cleaning Suggestions:")
            for i, suggestion in enumerate(st.session_state["current_suggestions"]):
                expander_title = (
                    f"Suggestion {i+1}: {suggestion.get('method', 'Unknown Method')}"
                )
                with st.expander(expander_title):
                    st.write(f"**Method:** `{suggestion.get('method')}`")
                    st.write(f"**Pros:** {suggestion.get('pro')}")
                    st.write(f"**Cons:** {suggestion.get('con')}")

                    if st.button(
                        f"Apply '{suggestion.get('method')}' to {selected_column_clean}",
                        key=f"apply_clean_{selected_column_clean}_{i}",
                    ):
                        action = {
                            "column": selected_column_clean,
                            "method": suggestion.get("method"),
                            "parameters": suggestion.get("parameters", {}),
                        }
                        st.session_state["cleaning_actions"].append(action)
                        st.success(
                            f"Action '{suggestion.get('method')}' added for column '{selected_column_clean}'."
                        )
                        st.session_state["current_suggestions"] = (
                            []
                        )  # Clear suggestions after adding
                        st.rerun()

    st.markdown("---")
    st.subheader("Applied Cleaning Actions")
    if not st.session_state["cleaning_actions"]:
        st.info("No cleaning actions applied yet.")
    else:
        for idx, action in enumerate(st.session_state["cleaning_actions"]):
            st.write(
                f"**{idx+1}. Column:** `{action['column']}` | **Method:** `{action['method']}`"
            )
            if action["parameters"]:
                st.write(
                    f"   **Parameters:** {json.dumps(action['parameters'])}"
                )  # Display parameters cleanly
        st.markdown("---")
        if st.button(
            "Apply All Selected Cleaning Actions and Preview", key="apply_all_clean_btn"
        ):
            with st.spinner("Applying cleaning actions..."):
                try:
                    df_current_state = pd.read_json(st.session_state["current_df"])
                    cleaned_df = apply_cleaning_actions(
                        df_current_state.copy(), st.session_state["cleaning_actions"]
                    )

                    st.session_state["cleaned_df"] = cleaned_df.to_json()
                    st.session_state["current_df"] = (
                        cleaned_df.to_json()
                    )  # Update current_df for chaining operations
                    st.session_state["last_cleaned_time"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    st.success("Cleaning actions applied!")
                    st.subheader("Preview of Cleaned Data (First 5 rows)")
                    st.dataframe(cleaned_df.head())
                    st.download_button(
                        label="Download Cleaned Data (CSV)",
                        data=cleaned_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"cleaned_{dataset_name.split('.')[0]}.csv",
                        mime="text/csv",
                        key="download_cleaned_preview",
                    )
                    st.info(
                        "You can download the cleaned data from here, or go to the 'Export' page for more options."
                    )
                except Exception as e:
                    st.error(f"Error applying cleaning actions: {e}")

    st.markdown("---")
    st.subheader("Custom Validation Rules")

    with st.expander("Define a new validation rule"):
        col_to_validate = st.selectbox(
            "Select column to validate:", df.columns, key="select_col_validate"
        )
        rule_type = st.selectbox(
            "Select rule type:",
            [
                "greater_than",
                "less_than",
                "equals",
                "not_equals",
                "is_null",
                "is_not_null",
                "contains_string",
                "regex_match",
            ],
            key="select_rule_type",
        )

        value_input = None
        if rule_type not in ["is_null", "is_not_null"]:
            if pd.api.types.is_string_dtype(df[col_to_validate]):
                value_input = st.text_input(
                    "Value/Regex for comparison:", key="rule_value_text"
                )
            elif pd.api.types.is_numeric_dtype(df[col_to_validate]):
                value_input = st.number_input(
                    "Value for comparison:", key="rule_value_number", value=0.0
                )
            else:
                value_input = st.text_input(
                    "Value for comparison:", key="rule_value_general"
                )

        if st.button("Add Validation Rule", key="add_validation_rule_btn"):
            if col_to_validate:
                rule_id = f"{col_to_validate}_{rule_type}_{len(st.session_state['validation_rules'])}"
                new_rule = {
                    "id": rule_id,
                    "column": col_to_validate,
                    "rule_type": rule_type,
                    "value": value_input if value_input is not None else None,
                }
                st.session_state["validation_rules"].append(new_rule)
                st.success(
                    f"Rule added: {col_to_validate} {rule_type} {value_input if value_input is not None else ''}"
                )
                st.rerun()

    st.write("##### Defined Validation Rules:")
    if not st.session_state["validation_rules"]:
        st.info("No validation rules defined yet.")
    else:
        for idx, rule in enumerate(st.session_state["validation_rules"]):
            display_value = f"'{rule['value']}'" if rule["value"] is not None else ""
            st.write(
                f"**{idx+1}. Column:** `{rule['column']}` | **Rule:** `{rule['rule_type']}` {display_value}"
            )
            if st.button(f"Remove Rule {idx+1}", key=f"remove_rule_{idx}"):
                st.session_state["validation_rules"].pop(idx)
                st.success(f"Rule {idx+1} removed.")
                st.rerun()

    if st.button("Run Custom Validation", key="run_validation_btn"):
        if not st.session_state["validation_rules"]:
            st.warning("Please define at least one validation rule first.")
        else:
            with st.spinner("Running custom validations..."):
                df_to_validate = pd.read_json(st.session_state["current_df"])
                validation_results = validate_data(
                    df_to_validate, st.session_state["validation_rules"]
                )
                st.session_state["validation_results"] = validation_results
                st.success("Validation complete!")
                st.rerun()

    if st.session_state["validation_results"]:
        st.write("##### Validation Results:")
        for rule_id, result in st.session_state["validation_results"].items():
            rule_info = next(
                (r for r in st.session_state["validation_rules"] if r["id"] == rule_id),
                {"column": "N/A", "rule_type": "N/A", "value": "N/A"},
            )
            display_value = (
                f"'{rule_info['value']}'" if rule_info["value"] is not None else ""
            )
            st.markdown(
                f"**Rule: `{rule_info['column']} {rule_info['rule_type']} {display_value}`**"
            )
            st.write(f"- **Violating Rows Count:** `{result['violating_rows_count']}`")
            if result["violating_rows_count"] > 0:
                st.write("###### Sample of Violating Rows:")
                st.dataframe(result["violating_rows_sample"])
            else:
                st.info("No violations found for this rule.")
            st.markdown("---")

    st.markdown("---")
    st.info(
        "The custom validation helps ensure your data meets specific business or quality requirements."
    )
