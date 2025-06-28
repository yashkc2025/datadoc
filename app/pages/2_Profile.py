import streamlit as st
import pandas as pd
import plotly.express as px
from app.utils.profiling import profile_dataset, generate_statistics
import json

st.set_page_config(
    page_title="Profile Dataset",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Profile Dataset")
st.markdown("Get a comprehensive overview of your dataset's quality and characteristics.")

if "current_df" not in st.session_state or not st.session_state["current_df"]:
    st.warning("No dataset loaded. Please go to the 'Upload' page to upload or select a dataset.")
else:
    df = pd.read_json(st.session_state["current_df"])
    dataset_name = st.session_state.get("current_dataset_name", "Unnamed Dataset")
    st.subheader(f"Profiling: {dataset_name}")

    if st.button("Generate Full Profile"):
        with st.spinner("Generating dataset profile... This might take a moment."):
            # Generate overall statistics
            overall_stats = generate_statistics(df)
            st.session_state["overall_stats"] = overall_stats

            # Generate column-level profiles
            column_profiles = {}
            for col in df.columns:
                col_profile = profile_dataset(df[[col]]) # Pass single column DataFrame
                if col in col_profile: # profile_dataset returns a dict of profiles
                    column_profiles[col] = col_profile[col]
            st.session_state["column_profiles"] = column_profiles
            st.success("Dataset profile generated!")
            st.rerun() # Rerun to display the results below

    if "overall_stats" in st.session_state:
        st.subheader("Overall Dataset Statistics")
        stats_df = pd.DataFrame([st.session_state["overall_stats"]]).T
        stats_df.columns = ["Value"]
        st.dataframe(stats_df)

    if "column_profiles" in st.session_state:
        st.subheader("Column-level Profiles")
        profiles_df = pd.DataFrame.from_dict(st.session_state["column_profiles"], orient="index")
        st.dataframe(profiles_df)

        st.subheader("Visual Diagnostics")
        selected_column = st.selectbox("Select a column to visualize:", df.columns)

        if selected_column:
            column_data = df[selected_column]
            col_profile = st.session_state["column_profiles"].get(selected_column, {})
            st.write(f"#### Details for column: **{selected_column}**")
            st.json(col_profile) # Display raw profile data

            # Basic visualizations based on column type
            if pd.api.types.is_numeric_dtype(column_data):
                st.write("##### Distribution (Histogram)")
                fig = px.histogram(df, x=selected_column, nbins=30, title=f"Distribution of {selected_column}")
                st.plotly_chart(fig, use_container_width=True)

                st.write("##### Box Plot (for Outliers)")
                fig_box = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
                st.plotly_chart(fig_box, use_container_width=True)

            elif pd.api.types.is_categorical_dtype(column_data) or column_data.nunique() < 20: # Treat low-cardinality as categorical
                st.write("##### Value Counts (Bar Chart)")
                value_counts = column_data.value_counts().reset_index()
                value_counts.columns = [selected_column, "Count"]
                fig_bar = px.bar(value_counts, x=selected_column, y="Count", title=f"Value Counts of {selected_column}")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No specific visualization available for this column type or high cardinality (text).")
    else:
        st.info("Click 'Generate Full Profile' to see dataset statistics and column profiles.")

st.markdown("---")
st.info("The profile helps you understand data quality issues like missing values, outliers, and data distribution.")