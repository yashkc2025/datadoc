import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff  # For correlation matrix
import numpy as np

st.set_page_config(page_title="Data Dashboard", page_icon="📈", layout="wide")

st.title("📈 Data Dashboard")
st.markdown("Create custom interactive visualizations of your current dataset.")

if "current_df" not in st.session_state or not st.session_state["current_df"]:
    st.warning(
        "No dataset loaded. Please go to the 'Upload' page to upload or select a dataset."
    )
else:
    df = pd.read_json(st.session_state["current_df"])
    dataset_name = st.session_state.get("current_dataset_name", "Unnamed Dataset")
    st.subheader(f"Dashboard for: {dataset_name}")

    st.markdown("---")
    st.subheader("Choose Your Visualization")

    plot_type = st.selectbox(
        "Select Plot Type:",
        [
            "Histogram",
            "Box Plot",
            "Scatter Plot",
            "Bar Chart (Categorical)",
            "Correlation Matrix",
        ],
    )

    if plot_type == "Histogram":
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            selected_column = st.selectbox("Select Numeric Column:", numeric_cols)
            if selected_column:
                fig = px.histogram(
                    df,
                    x=selected_column,
                    nbins=30,
                    title=f"Distribution of {selected_column}",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for a histogram.")

    elif plot_type == "Box Plot":
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            selected_column = st.selectbox("Select Numeric Column:", numeric_cols)
            if selected_column:
                fig = px.box(
                    df, y=selected_column, title=f"Box Plot of {selected_column}"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for a box plot.")

    elif plot_type == "Scatter Plot":
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            x_column = st.selectbox(
                "Select X-axis Column:", numeric_cols, key="scatter_x"
            )
            y_column = st.selectbox(
                "Select Y-axis Column:", numeric_cols, key="scatter_y"
            )
            color_column = st.selectbox(
                "Select Color by (Optional):",
                [None] + df.columns.tolist(),
                key="scatter_color",
            )
            if x_column and y_column:
                fig = px.scatter(
                    df,
                    x=x_column,
                    y=y_column,
                    color=color_column,
                    title=f"Scatter Plot: {x_column} vs {y_column}",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least two numeric columns for a scatter plot.")

    elif plot_type == "Bar Chart (Categorical)":
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if categorical_cols:
            selected_column = st.selectbox(
                "Select Categorical Column:", categorical_cols
            )
            if selected_column:
                value_counts = df[selected_column].value_counts().reset_index()
                value_counts.columns = [selected_column, "Count"]
                fig = px.bar(
                    value_counts,
                    x=selected_column,
                    y="Count",
                    title=f"Value Counts of {selected_column}",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns available for a bar chart.")

    elif plot_type == "Correlation Matrix":
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr(numeric_only=True)
            if not corr_matrix.empty:
                st.write("##### Correlation Matrix")
                fig = ff.create_annotated_heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    colorscale="Viridis",
                    showscale=True,
                    hoverinfo="z",  # Shows the correlation value on hover
                )
                fig.update_layout(title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "Not enough numeric columns with variance to compute a meaningful correlation matrix."
                )
        else:
            st.info("No numeric columns available to compute a correlation matrix.")

st.markdown("---")
st.info(
    "Use this dashboard to visually explore relationships and distributions in your cleaned data."
)
