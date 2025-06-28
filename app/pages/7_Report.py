import streamlit as st
import pandas as pd
import json
from datetime import datetime

st.set_page_config(page_title="Generate Report", page_icon="📄", layout="wide")

st.title("📄 Generate Data Quality Report")
st.markdown(
    "Generate a summarized HTML report of your dataset's profiling and cleaning process."
)


def generate_html_report_content(
    original_df_json,
    current_df_json,
    overall_stats,
    column_profiles,
    cleaning_actions,
    validation_results,
) -> str:
    """
    Generates the HTML content for the data quality report.
    """
    original_df = pd.read_json(original_df_json) if original_df_json else None
    current_df = pd.read_json(current_df_json) if current_df_json else None

    report_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DataDoc Data Quality Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 20px; background-color: #f9f9f9; }}
            .container {{ max-width: 1000px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1, h2, h3, h4 {{ color: #0056b3; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .section {{ margin-bottom: 30px; padding: 15px; border: 1px solid #eee; border-radius: 5px; background-color: #fcfcfc; }}
            .info {{ background-color: #e6f7ff; border-left: 5px solid #2196F3; padding: 10px; margin-bottom: 10px; }}
            .warning {{ background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 10px; margin-bottom: 10px; }}
            .success {{ background-color: #e0ffe0; border-left: 5px solid #4CAF50; padding: 10px; margin-bottom: 10px; }}
            pre {{ background-color: #eee; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>DataDoc Data Quality Report</h1>
            <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Dataset Name: {st.session_state.get('current_dataset_name', 'N/A')}</p>

            <div class="section">
                <h2>Overall Dataset Statistics</h2>
                {"<p>No overall statistics available. Please profile the dataset first.</p>" if not overall_stats else pd.DataFrame([overall_stats]).T.to_html()}
            </div>

            <div class="section">
                <h2>Column-level Profiles</h2>
                {"<p>No column profiles available. Please profile the dataset first.</p>" if not column_profiles else pd.DataFrame.from_dict(column_profiles, orient="index").to_html()}
                <p class="info">Note: Full interactive visualizations are available on the 'Profile' and 'Dashboard' pages within the DataDoc application.</p>
            </div>

            <div class="section">
                <h2>Applied Cleaning Actions</h2>
                {"<p>No cleaning actions were applied.</p>" if not cleaning_actions else pd.DataFrame(cleaning_actions).to_html(index=False)}
            </div>

            <div class="section">
                <h2>Custom Validation Results</h2>
                {"<p>No validation results available. Please define and run validations on the 'Clean' page.</p>" if not validation_results else ""}
                {"".join([
                    f"""
                    <h3>Rule: {val['rule_description']}</h3>
                    <p>Violating Rows Count: {val['violating_rows_count']}</p>
                    {"<h4>Sample of Violating Rows:</h4>" + val['violating_rows_sample'].to_html() if not val['violating_rows_sample'].empty else "<p class='success'>No violations found for this rule.</p>"}
                    """
                    for rule_id, val in validation_results.items()
                ])}
            </div>

            <div class="section">
                <h2>Original Data Snapshot (First 5 rows)</h2>
                {"<p>Original data not available. Please upload a dataset first.</p>" if original_df is None else original_df.head().to_html()}
            </div>
            
            <div class="section">
                <h2>Current Data Snapshot (First 5 rows)</h2>
                {"<p>Current data not available. Please process a dataset.</p>" if current_df is None else current_df.head().to_html()}
            </div>

        </div>
    </body>
    </html>
    """
    return report_html


if "current_df" not in st.session_state or not st.session_state["current_df"]:
    st.warning(
        "No dataset loaded. Please go to the 'Upload' page to upload or select a dataset."
    )
else:
    st.info(
        "Ensure you have profiled your data and applied any desired cleaning actions on the 'Profile' and 'Clean' pages before generating the report."
    )

    if st.button("Generate HTML Report"):
        with st.spinner("Generating report..."):
            # Fetch all necessary data from session state
            original_df_json = st.session_state.get("original_df")
            current_df_json = st.session_state.get("current_df")
            overall_stats = st.session_state.get("overall_stats", {})
            column_profiles = st.session_state.get("column_profiles", {})
            cleaning_actions = st.session_state.get("cleaning_actions", [])
            validation_results = st.session_state.get("validation_results", {})

            report_content = generate_html_report_content(
                original_df_json,
                current_df_json,
                overall_stats,
                column_profiles,
                cleaning_actions,
                validation_results,
            )

            st.session_state["generated_report_html"] = report_content
            st.success("Report generated! You can now download it.")
            st.rerun()  # Rerun to display download button


if "generated_report_html" in st.session_state:
    st.download_button(
        label="Download HTML Report",
        data=st.session_state["generated_report_html"].encode("utf-8"),
        file_name=f"DataDoc_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
    )
    st.markdown("---")
    st.info(
        "The HTML report summarizes your data profiling and cleaning steps. For interactive visualizations, use the 'Dashboard' page."
    )

st.markdown("---")
st.info("This report provides a static summary of your data operations.")
