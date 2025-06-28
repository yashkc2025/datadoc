import streamlit as st
import pandas as pd
from app.utils.cleaning import generate_script
from app.utils.storage import save_result
import os

st.set_page_config(
    page_title="Export Data & Script",
    page_icon="💾",
    layout="wide"
)

st.title("💾 Export Data & Script")
st.markdown("Download your cleaned dataset and the Python script that generated it.")

if "cleaned_df" not in st.session_state or not st.session_state["cleaned_df"]:
    st.warning("No cleaned dataset available. Please go to the 'Upload' page, then 'Profile', and finally 'Clean' your data.")
else:
    cleaned_df = pd.read_json(st.session_state["cleaned_df"])
    dataset_name = st.session_state.get("current_dataset_name", "Unnamed Dataset")
    original_file_name_base = os.path.splitext(dataset_name)[0]

    st.subheader(f"Cleaned Dataset: {original_file_name_base}.csv")
    st.dataframe(cleaned_df.head()) # Show a preview

    # Download cleaned data
    cleaned_csv = cleaned_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Data (CSV)",
        data=cleaned_csv,
        file_name=f"cleaned_{original_file_name_base}.csv",
        mime="text/csv",
    )

    if st.button("Save Cleaned Data to Results Folder"):
        try:
            save_path = save_result(cleaned_df, f"cleaned_{original_file_name_base}.csv")
            st.success(f"Cleaned data saved to {save_path}")
        except Exception as e:
            st.error(f"Error saving cleaned data: {e}")


    st.markdown("---")
    st.subheader("Python Cleaning Script")

    if "cleaning_actions" in st.session_state and st.session_state["cleaning_actions"]:
        cleaning_script = generate_script(st.session_state["cleaning_actions"])
        st.code(cleaning_script, language="python")

        # Download Python script
        st.download_button(
            label="Download Python Cleaning Script",
            data=cleaning_script.encode('utf-8'),
            file_name=f"clean_script_{original_file_name_base}.py",
            mime="text/x-python",
        )
    else:
        st.info("No cleaning actions were applied, so no script can be generated.")

st.markdown("---")
st.info("The exported Python script allows you to reproduce the cleaning steps programmatically.")