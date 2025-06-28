import streamlit as st
import pandas as pd
import plotly.express as px
from app.utils.profiling import (
    profile_dataset,
    generate_statistics,
    infer_data_type_suggestions,
)
import json

st.set_page_config(page_title="Profile Dataset", page_icon="🔍", layout="wide")

st.title("🔍 Profile Dataset")
st.markdown(
    "Get a comprehensive overview of your dataset's quality and characteristics."
)

if "current_df" not in st.session_state or not st.session_state["current_df"]:
    st.warning(
        "No dataset loaded. Please go to the 'Upload' page to upload or select a dataset."
    )
else:
    df_current = pd.read_json(st.session_state["current_df"])
    dataset_name = st.session_state.get("current_dataset_name", "Unnamed Dataset")
    st.subheader(f"Profiling: {dataset_name}")

    if st.button("Generate Full Profile"):
        with st.spinner("Generating dataset profile... This might take a moment."):
            # Generate overall statistics
            overall_stats = generate_statistics(df_current)
            st.session_state["overall_stats"] = overall_stats

            # Generate column-level profiles with type suggestions
            column_profiles = {}
            type_suggestions = infer_data_type_suggestions(
                df_current
            )  # Get overall type suggestions
            for col in df_current.columns:
                col_profile = profile_dataset(
                    df_current[[col]]
                )  # Pass single column DataFrame
                if col in col_profile:
                    col_profile[col]["type_suggestion"] = type_suggestions.get(
                        col
                    )  # Add type suggestion to profile
                    column_profiles[col] = col_profile[col]
            st.session_state["column_profiles"] = column_profiles
            st.success("Dataset profile generated!")
            st.rerun()  # Rerun to display the results below

    if "overall_stats" in st.session_state:
        st.subheader("Overall Dataset Statistics")
        stats_df = pd.DataFrame([st.session_state["overall_stats"]]).T
        stats_df.columns = ["Value"]
        st.dataframe(stats_df)

    if "column_profiles" in st.session_state:
        st.subheader("Column-level Profiles")
        profiles_df = pd.DataFrame.from_dict(
            st.session_state["column_profiles"], orient="index"
        )

        # Reorder columns to put 'type_suggestion' near 'dtype'
        cols = profiles_df.columns.tolist()
        if "type_suggestion" in cols:
            cols.remove("type_suggestion")
            try:
                dtype_idx = cols.index("dtype")
                cols.insert(dtype_idx + 1, "type_suggestion")
            except ValueError:
                cols.insert(
                    0, "type_suggestion"
                )  # If dtype isn't found, put it at the beginning
        profiles_df = profiles_df[cols]

        st.dataframe(profiles_df)

        st.subheader("Visual Diagnostics")
        selected_column = st.selectbox(
            "Select a column to visualize:", df_current.columns
        )

        if selected_column:
            column_data = df_current[selected_column]
            col_profile = st.session_state["column_profiles"].get(selected_column, {})
            st.write(f"#### Details for column: **{selected_column}**")

            # Display type suggestion specifically
            if "type_suggestion" in col_profile and col_profile["type_suggestion"]:
                st.info(
                    f"**Type Conversion Suggestion:** This column could be converted to **{col_profile['type_suggestion']}**."
                )

            st.json(col_profile)  # Display raw profile data

            # Basic visualizations based on column type
            if pd.api.types.is_numeric_dtype(column_data):
                st.write("##### Distribution (Histogram)")
                fig = px.histogram(
                    df_current,
                    x=selected_column,
                    nbins=30,
                    title=f"Distribution of {selected_column}",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.write("##### Box Plot (for Outliers)")
                fig_box = px.box(
                    df_current,
                    y=selected_column,
                    title=f"Box Plot of {selected_column}",
                )
                st.plotly_chart(fig_box, use_container_width=True)

            elif (
                pd.api.types.is_categorical_dtype(column_data)
                or column_data.nunique() < 20
            ):  # Treat low-cardinality as categorical
                st.write("##### Value Counts (Bar Chart)")
                value_counts = column_data.value_counts().reset_index()
                value_counts.columns = [selected_column, "Count"]
                fig_bar = px.bar(
                    value_counts,
                    x=selected_column,
                    y="Count",
                    title=f"Value Counts of {selected_column}",
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info(
                    "No specific visualization available for this column type or high cardinality (text)."
                )
    else:
        st.info(
            "Click 'Generate Full Profile' to see dataset statistics and column profiles."
        )

    st.markdown("---")
    st.subheader("Comparison: Original vs. Current Data")
    if (
        "original_df" in st.session_state
        and st.session_state["original_df"]
        and "current_df" in st.session_state
        and st.session_state["current_df"]
    ):
        df_original = pd.read_json(st.session_state["original_df"])

        st.info(
            "Select a column to compare its distribution and statistics before and after cleaning actions."
        )
        compare_column = st.selectbox(
            "Select a column to compare:", df_current.columns, key="compare_column"
        )

        if compare_column:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"##### Original Data: `{compare_column}`")
                st.dataframe(df_original[[compare_column]].describe().T)
                if pd.api.types.is_numeric_dtype(df_original[compare_column]):
                    fig_orig = px.histogram(
                        df_original,
                        x=compare_column,
                        nbins=30,
                        title=f"Original Distribution of {compare_column}",
                    )
                    st.plotly_chart(fig_orig, use_container_width=True)
                elif (
                    pd.api.types.is_categorical_dtype(df_original[compare_column])
                    or df_original[compare_column].nunique() < 20
                ):
                    value_counts_orig = (
                        df_original[compare_column].value_counts().reset_index()
                    )
                    value_counts_orig.columns = [compare_column, "Count"]
                    fig_orig_bar = px.bar(
                        value_counts_orig,
                        x=compare_column,
                        y="Count",
                        title=f"Original Value Counts of {compare_column}",
                    )
                    st.plotly_chart(fig_orig_bar, use_container_width=True)
                else:
                    st.write(
                        "No direct visualization for this column type in original data for comparison."
                    )

            with col2:
                st.markdown(f"##### Current Data: `{compare_column}`")
                st.dataframe(df_current[[compare_column]].describe().T)
                if pd.api.types.is_numeric_dtype(df_current[compare_column]):
                    fig_curr = px.histogram(
                        df_current,
                        x=compare_column,
                        nbins=30,
                        title=f"Current Distribution of {compare_column}",
                    )
                    st.plotly_chart(fig_curr, use_container_width=True)
                elif (
                    pd.api.types.is_categorical_dtype(df_current[compare_column])
                    or df_current[compare_column].nunique() < 20
                ):
                    value_counts_curr = (
                        df_current[compare_column].value_counts().reset_index()
                    )
                    value_counts_curr.columns = [compare_column, "Count"]
                    fig_curr_bar = px.bar(
                        value_counts_curr,
                        x=compare_column,
                        y="Count",
                        title=f"Current Value Counts of {compare_column}",
                    )
                    st.plotly_chart(fig_curr_bar, use_container_width=True)
                else:
                    st.write(
                        "No direct visualization for this column type in current data for comparison."
                    )
    else:
        st.info(
            "Load a dataset on the 'Upload' page and apply some cleaning actions on the 'Clean' page to enable data comparison."
        )

st.markdown("---")
st.info(
    "The profile helps you understand data quality issues like missing values, outliers, and data distribution."
)
