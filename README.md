# DataDoc: AI-Powered Data Profiler & Cleaner

"DataDoc" is a professional-grade, LLM-enhanced data profiling and cleaning tool designed to help data scientists and analysts rapidly explore and clean raw datasets. It offers one-click profiling, intelligent cleaning suggestions, visual diagnostics, and Python script export — all wrapped in a clean, deployable Streamlit interface.

This tool operates **fully offline**, **without logins**, and uses **open-source LLMs** (simulated in this version due to environment constraints) and **local object storage** to maintain privacy and flexibility.

<img width="2560" height="1319" alt="image" src="https://github.com/user-attachments/assets/403baa0d-9c5d-4d68-ae19-e016e6e10ab5" />

## Features:

- **Home**: Overview of recent datasets and profiling status.
- **Upload**: Upload CSV/XLSX files, preview, and store locally.
- **Profile**: View comprehensive column statistics, data quality metrics, and interactive visualizations.
- **Clean**: Get LLM-simulated cleaning suggestions and apply transformations to your data.
- **Export**: Download cleaned data and a generated Python script of applied cleaning actions.
- **History**: Basic log of previously processed datasets.

## Setup and Run:

1.  **Clone this repository** (or create the files as provided).
2.  **Navigate to the `datadoc` directory**:
    ```bash
    cd datadoc
    ```
3.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\\Scripts\\activate`
    ```
4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the application**:
    ```bash
    python run.py
    ```
    This will open the Streamlit app in your web browser.
