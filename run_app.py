import subprocess
import sys
import webbrowser
import os
import time


def main() -> None:
    app_path = "/Users/vidyadharpothula/dsa_project/multilane_streamlit_original.py"
    url = "http://localhost:8501"

    env = os.environ.copy()
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port=8501"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait briefly for the server to bind to the port
    time.sleep(1.5)
    try:
        webbrowser.open(url)
    except Exception:
        pass

    # Stream output to console for visibility; exit when streamlit exits
    if proc.stdout is not None:
        for line in iter(proc.stdout.readline, b""):
            if not line:
                break
            try:
                sys.stdout.write(line.decode("utf-8", errors="ignore"))
            except Exception:
                pass


if __name__ == "__main__":
    main()


