import streamlit as st
import pandas as pd
from app.utils.llm_agent import suggest_cleaning
from app.utils.cleaning import apply_cleaning_actions
import json  # For handling cleaning actions
from datetime import datetime

st.set_page_config(page_title="Clean Dataset", page_icon="🧹", layout="wide")

st.title("🧹 Clean Dataset")
st.markdown("Apply intelligent cleaning suggestions to your data.")

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
    st.subheader(f"Cleaning: {dataset_name}")

    if "cleaning_actions" not in st.session_state:
        st.session_state["cleaning_actions"] = []  # List to store applied actions

    st.subheader("AI-Simulated Cleaning Suggestions")
    selected_column = st.selectbox(
        "Select a column to get cleaning suggestions for:", df.columns
    )

    if selected_column:
        col_profile = column_profiles.get(selected_column, {})
        st.write(f"**Current Profile for '{selected_column}':**")
        st.json(col_profile)

        if st.button(f"Get Suggestions for {selected_column}"):
            with st.spinner("Getting AI-simulated suggestions..."):
                suggestions = suggest_cleaning(
                    col_profile, model="simulated_model"
                )  # Pass simulated model
                st.session_state["current_suggestions"] = suggestions
                st.success("Suggestions loaded!")

        if (
            "current_suggestions" in st.session_state
            and st.session_state["current_suggestions"]
        ):
            st.markdown("---")
            st.write("##### Suggested Cleaning Actions:")
            for i, suggestion in enumerate(st.session_state["current_suggestions"]):
                expander_title = (
                    f"Suggestion {i+1}: {suggestion.get('method', 'Unknown Method')}"
                )
                with st.expander(expander_title):
                    st.write(f"**Method:** `{suggestion.get('method')}`")
                    st.write(f"**Pros:** {suggestion.get('pro')}")
                    st.write(f"**Cons:** {suggestion.get('con')}")

                    # Option to apply this specific suggestion
                    if st.button(
                        f"Apply '{suggestion.get('method')}' to {selected_column}",
                        key=f"apply_{selected_column}_{i}",
                    ):
                        action = {
                            "column": selected_column,
                            "method": suggestion.get("method"),
                            "parameters": suggestion.get(
                                "parameters", {}
                            ),  # Add parameters if any are suggested
                        }
                        st.session_state["cleaning_actions"].append(action)
                        st.success(
                            f"Action '{suggestion.get('method')}' added for column '{selected_column}'."
                        )
                        # Clear current suggestions after adding an action to avoid re-applying easily
                        st.session_state["current_suggestions"] = []
                        st.rerun()  # Rerun to update the actions list

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
                st.write(f"   **Parameters:** {action['parameters']}")
        st.markdown("---")
        if st.button("Apply All Selected Cleaning Actions and Preview"):
            with st.spinner("Applying cleaning actions..."):
                try:
                    cleaned_df = apply_cleaning_actions(
                        df.copy(), st.session_state["cleaning_actions"]
                    )
                    st.session_state["cleaned_df"] = cleaned_df.to_json()
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
                    )
                    st.info(
                        "You can download the cleaned data from here, or go to the 'Export' page for more options."
                    )
                except Exception as e:
                    st.error(f"Error applying cleaning actions: {e}")
    st.markdown("---")
    st.info(
        "After applying actions, you can go to the 'Export' page to download the cleaned dataset or the Python script."
    )
