import streamlit as st
import pandas as pd
from app.utils.storage import list_saved_datasets, list_saved_results

st.set_page_config(
    page_title="History",
    page_icon="📚",
    layout="wide"
)

st.title("📚 History")
st.markdown("Review your past uploaded datasets and cleaned results.")

st.subheader("Uploaded Datasets History")
try:
    datasets = list_saved_datasets()
    if not datasets:
        st.info("No datasets have been uploaded yet.")
    else:
        df_datasets = pd.DataFrame(datasets)
        st.dataframe(df_datasets.set_index("Name"))
except Exception as e:
    st.error(f"Error loading dataset history: {e}")


st.subheader("Cleaned Results History")
try:
    results = list_saved_results()
    if not results:
        st.info("No cleaned results have been saved yet.")
    else:
        df_results = pd.DataFrame(results)
        st.dataframe(df_results.set_index("Name"))
except Exception as e:
    st.error(f"Error loading results history: {e}")

st.markdown("---")
st.info("This page provides a basic log of your activities. Full version could include detailed diffs.")