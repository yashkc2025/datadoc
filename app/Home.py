import streamlit as st
import pandas as pd
from datetime import datetime
from app.utils.storage import list_saved_datasets

st.set_page_config(
    page_title="DataDoc: AI Data Profiler & Cleaner", page_icon="📊", layout="wide"
)

st.title("📊 DataDoc: AI-Powered Data Profiler & Cleaner")
st.markdown(
    """
    Welcome to DataDoc! This tool helps data scientists and analysts rapidly explore
    and clean raw datasets with intelligent suggestions.
    """
)

st.subheader("Recent Datasets")

try:
    datasets = list_saved_datasets()
    if not datasets:
        st.info("No datasets uploaded yet. Go to the 'Upload' page to get started!")
    else:
        st.dataframe(pd.DataFrame(datasets).set_index("Name"))
except Exception as e:
    st.error(f"Error loading recent datasets: {e}")

st.markdown("---")
st.subheader("How to use:")
st.markdown(
    """
    1.  **Upload**: Go to the 'Upload' page to upload your CSV or XLSX file.
    2.  **Profile**: Navigate to the 'Profile' page to get a comprehensive overview of your data. **You can also compare original and cleaned data here.**
    3.  **Clean**: On the 'Clean' page, receive AI-simulated suggestions for data cleaning and apply them. **You can also define and run custom validation rules, apply categorical encoding, perform text cleaning, standardize date formats, extract date parts, and handle outliers here.**
    4.  **Dashboard**: Explore your data visually with interactive charts.
    5.  **Export**: Download your cleaned dataset and a Python script of the applied transformations. **Now supports JSON export.**
    6.  **Report**: Generate a summarized HTML report of your data profiling and cleaning process.
    7.  **History**: (Basic) View a log of your past operations.
    """
)

st.markdown("---")
st.info(
    "This application operates fully offline and locally, ensuring your data privacy."
)
