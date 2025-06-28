import streamlit as st
import pandas as pd
from app.utils.cleaning import generate_script
from app.utils.storage import save_result
import os

st.set_page_config(page_title="Export Data & Script", page_icon="💾", layout="wide")

st.title("💾 Export Data & Script")
st.markdown("Download your cleaned dataset and the Python script that generated it.")

if "current_df" not in st.session_state or not st.session_state["current_df"]:
    st.warning(
        "No dataset loaded. Please go to the 'Upload' page, then 'Profile', and finally 'Clean' your data."
    )
else:
    # Export the currently processed DF (either original or cleaned)
    df_to_export = pd.read_json(st.session_state["current_df"])
    dataset_name = st.session_state.get("current_dataset_name", "Unnamed Dataset")
    original_file_name_base = os.path.splitext(dataset_name)[0]

    st.subheader(f"Current Dataset for Export: {original_file_name_base}.csv")
    st.dataframe(df_to_export.head())  # Show a preview

    col_csv, col_json = st.columns(2)
    with col_csv:
        # Download current data as CSV
        current_csv = df_to_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Current Data (CSV)",
            data=current_csv,
            file_name=f"current_{original_file_name_base}.csv",
            mime="text/csv",
        )
    with col_json:
        # Download current data as JSON
        current_json = df_to_export.to_json(orient="records", indent=4).encode("utf-8")
        st.download_button(
            label="Download Current Data (JSON)",
            data=current_json,
            file_name=f"current_{original_file_name_base}.json",
            mime="application/json",
        )

    if st.button("Save Current Data to Results Folder"):
        try:
            save_path = save_result(
                df_to_export, f"processed_{original_file_name_base}.csv"
            )  # Always save CSV
            st.success(f"Current data saved to {save_path}")
        except Exception as e:
            st.error(f"Error saving current data: {e}")

    st.markdown("---")
    st.subheader("Python Cleaning Script")

    if "cleaning_actions" in st.session_state and st.session_state["cleaning_actions"]:
        cleaning_script = generate_script(st.session_state["cleaning_actions"])
        st.code(cleaning_script, language="python")

        # Download Python script
        st.download_button(
            label="Download Python Cleaning Script",
            data=cleaning_script.encode("utf-8"),
            file_name=f"clean_script_{original_file_name_base}.py",
            mime="text/x-python",
        )
    else:
        st.info("No cleaning actions were applied, so no script can be generated.")

st.markdown("---")
st.info(
    "The exported Python script allows you to reproduce the cleaning steps programmatically."
)
