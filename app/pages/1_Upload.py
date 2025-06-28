import streamlit as st
import pandas as pd
from app.utils.storage import save_dataset, list_saved_datasets
from datetime import datetime
import os

st.set_page_config(page_title="Upload Dataset", page_icon="⬆️", layout="wide")

st.title("⬆️ Upload Dataset")
st.markdown("Upload your CSV or XLSX file here to begin profiling and cleaning.")

uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    st.session_state["uploaded_file_name"] = file_name

    try:
        # Read the file into a DataFrame
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or XLSX file.")
            st.stop()

        st.success(
            f"Successfully loaded '{file_name}' with {len(df)} rows and {len(df.columns)} columns."
        )
        st.subheader("Data Preview (First 5 rows)")
        st.dataframe(df.head())

        if st.button("Save Dataset"):
            save_path = save_dataset(uploaded_file, file_name)
            st.session_state["current_dataset_path"] = save_path
            st.session_state["current_dataset_name"] = file_name
            st.session_state["last_uploaded_time"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            st.session_state["current_df"] = (
                df.to_json()
            )  # Store as JSON string for current working copy
            st.session_state["original_df"] = (
                df.to_json()
            )  # Store original for comparison purposes
            st.success(
                f"Dataset '{file_name}' saved successfully to {save_path}. You can now proceed to 'Profile'."
            )
            st.rerun()  # Rerun to update the session state and show current status
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.info(
            "Please ensure the file is not corrupted and is in a valid CSV or XLSX format."
        )

st.markdown("---")
st.subheader("Currently Loaded Dataset")
if (
    "current_dataset_name" in st.session_state
    and st.session_state["current_dataset_name"]
):
    st.write(f"**Name:** {st.session_state['current_dataset_name']}")
    st.write(f"**Path:** {st.session_state['current_dataset_path']}")
    st.write(
        f"**Last Uploaded/Saved:** {st.session_state.get('last_uploaded_time', 'N/A')}"
    )
    if "current_df" in st.session_state:
        st.write(f"**Rows:** {pd.read_json(st.session_state['current_df']).shape[0]}")
        st.write(
            f"**Columns:** {pd.read_json(st.session_state['current_df']).shape[1]}"
        )
else:
    st.info("No dataset is currently loaded. Please upload one above.")

st.markdown("---")
st.subheader("Recently Saved Datasets")
try:
    recent_datasets = list_saved_datasets()
    if recent_datasets:
        st.dataframe(pd.DataFrame(recent_datasets).set_index("Name"))
        if (
            "current_dataset_name" not in st.session_state
            or not st.session_state["current_dataset_name"]
        ):
            st.info(
                "You can select a dataset from the list above to load it for profiling and cleaning."
            )
            selected_dataset_name = st.selectbox(
                "Select a dataset to load:",
                [d["Name"] for d in recent_datasets],
                index=None,
            )
            if selected_dataset_name:
                selected_dataset_info = next(
                    (d for d in recent_datasets if d["Name"] == selected_dataset_name),
                    None,
                )
                if selected_dataset_info:
                    st.session_state["current_dataset_path"] = selected_dataset_info[
                        "Path"
                    ]
                    st.session_state["current_dataset_name"] = selected_dataset_info[
                        "Name"
                    ]
                    st.session_state["last_uploaded_time"] = selected_dataset_info[
                        "Last Modified"
                    ]
                    # Load the actual dataframe into session state
                    try:
                        df_loaded = pd.read_csv(
                            selected_dataset_info["Path"]
                        )  # Assuming CSV for simplicity, extend for XLSX
                        st.session_state["current_df"] = df_loaded.to_json()
                        st.session_state["original_df"] = (
                            df_loaded.to_json()
                        )  # Also load original when loading saved
                        st.success(
                            f"Dataset '{selected_dataset_name}' loaded successfully. Proceed to 'Profile' page."
                        )
                        st.rerun()
                    except Exception as load_e:
                        st.error(
                            f"Error loading selected dataset '{selected_dataset_name}': {load_e}"
                        )
    else:
        st.info("No datasets have been saved yet.")
except Exception as e:
    st.error(f"Error displaying recently saved datasets: {e}")
