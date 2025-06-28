import pandas as pd
from pathlib import Path
from datetime import datetime

# Define base paths relative to the project root
DATASETS_DIR = Path("datadoc") / "datasets"
RESULTS_DIR = Path("datadoc") / "results"

# Ensure directories exist
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_dataset(uploaded_file, file_name: str) -> Path:
    """
    Saves an uploaded Streamlit file object to the datasets directory.
    """
    file_path = DATASETS_DIR / file_name
    # Save the file content
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Loads a dataset from the given path. Supports CSV and XLSX.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".xlsx":
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type for loading: {path.suffix}")
    return df


def save_result(df: pd.DataFrame, file_name: str) -> Path:
    """
    Saves a cleaned DataFrame to the results directory.
    """
    file_path = RESULTS_DIR / file_name
    df.to_csv(file_path, index=False)
    return file_path


def list_saved_datasets() -> list[dict]:
    """
    Lists all saved datasets in the 'datasets' directory.
    Returns a list of dictionaries with Name, Size, Last Modified, Path.
    """
    datasets = []
    for f in DATASETS_DIR.iterdir():
        if f.is_file() and f.suffix in [".csv", ".xlsx"]:
            stats = f.stat()
            datasets.append(
                {
                    "Name": f.name,
                    "Size (KB)": round(stats.st_size / 1024, 2),
                    "Last Modified": datetime.fromtimestamp(stats.st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "Path": str(f),
                }
            )
    return datasets


def list_saved_results() -> list[dict]:
    """
    Lists all saved results in the 'results' directory.
    Returns a list of dictionaries with Name, Size, Last Modified, Path.
    """
    results = []
    for f in RESULTS_DIR.iterdir():
        if f.is_file() and f.suffix == ".csv":
            stats = f.stat()
            results.append(
                {
                    "Name": f.name,
                    "Size (KB)": round(stats.st_size / 1024, 2),
                    "Last Modified": datetime.fromtimestamp(stats.st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "Path": str(f),
                }
            )
    return results
