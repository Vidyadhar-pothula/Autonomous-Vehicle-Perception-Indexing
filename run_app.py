import subprocess
import sys
import webbrowser
import os
import time


def main() -> None:
    app_path = "/Users/vidyadharpothula/Desktop/dsa_project/smooth_av_simulation.py"
    url = "http://localhost:8501"

    env = os.environ.copy()
    # Make it publicly accessible by binding to all interfaces
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path, 
        "--server.port=8501", 
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait briefly for the server to bind to the port
    time.sleep(2.0)
    try:
        webbrowser.open(url)
    except Exception:
        pass

    print(f"Professional AV Simulation is running at: {url}")
    print("The simulation is now publicly accessible on your network.")
    print("Press Ctrl+C to stop the simulation.")

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


