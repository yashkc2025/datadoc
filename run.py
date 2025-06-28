import subprocess
import sys


def run_streamlit_app():
    print("Starting DataDoc Streamlit app...")
    try:
        # Use subprocess to run the streamlit command
        # This assumes streamlit is installed and in the PATH
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app/Home.py",
                "--server.port",
                "8501",
                "--browser.gatherUsageStats",
                "false",
            ],
            check=True,
        )
    except FileNotFoundError:
        print(
            "Error: 'streamlit' command not found. Please ensure Streamlit is installed."
        )
        print("You can install it using: pip install streamlit")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    run_streamlit_app()
