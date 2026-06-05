import subprocess
import sys
from pathlib import Path


def main():
    app_file = Path(__file__).with_name("app.py")
    if not app_file.exists():
        raise FileNotFoundError(f"Streamlit app file not found: {app_file}")

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)]
    print("starting streamlit app...")
    print("open the browser at http://localhost:8501")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
