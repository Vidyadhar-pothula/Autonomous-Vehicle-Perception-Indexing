#!/usr/bin/env python3
"""
Public deployment script for the Professional AV Simulation
Makes the simulation accessible from anywhere on the network
"""

import subprocess
import sys
import socket
import os
import time
from pathlib import Path


def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote server to get local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        return "127.0.0.1"


def main():
    """Deploy the simulation publicly"""
    print("=" * 60)
    print("Professional AV Simulation - Public Deployment")
    print("=" * 60)
    
    # Get paths
    app_path = Path(__file__).parent / "smooth_av_simulation.py"
    local_ip = get_local_ip()
    
    print(f"App path: {app_path}")
    print(f"Local IP: {local_ip}")
    print(f"Public URL: http://{local_ip}:8501")
    print("=" * 60)
    
    # Check if app exists
    if not app_path.exists():
        print(f"ERROR: App file not found at {app_path}")
        sys.exit(1)
    
    # Command to run Streamlit publicly
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port=8501",
        "--server.address=0.0.0.0",  # Bind to all interfaces
        "--server.headless=true",    # Don't auto-open browser
        "--browser.gatherUsageStats=false",
        "--server.enableCORS=false",  # Disable CORS for public access
        "--server.enableXsrfProtection=false"  # Disable XSRF for public access
    ]
    
    print("Starting Professional AV Simulation...")
    print("The simulation will be accessible at:")
    print(f"  Local:  http://localhost:8501")
    print(f"  Public: http://{local_ip}:8501")
    print("\nPress Ctrl+C to stop the simulation")
    print("=" * 60)
    
    try:
        # Start the process
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Stream output
        if proc.stdout:
            for line in iter(proc.stdout.readline, b""):
                if not line:
                    break
                try:
                    output = line.decode("utf-8", errors="ignore").strip()
                    if output:
                        print(output)
                except Exception:
                    pass
                    
    except KeyboardInterrupt:
        print("\nShutting down simulation...")
        if 'proc' in locals():
            proc.terminate()
        print("Simulation stopped.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
