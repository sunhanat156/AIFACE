"""AIFACE one-click launcher.

Checks dependencies, picks a free port, starts the server and opens the
dashboard in the default browser once it actually answers.

Run it with:  python launcher.py      (or just double-click START_AIFACE.bat)
"""
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.paths import APP_DIR as PROJECT_ROOT, FROZEN

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
PORT_ATTEMPTS = 20

# import name -> pip name
REQUIRED = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "sqlalchemy": "sqlalchemy",
    "cv2": "opencv-python",
    "insightface": "insightface",
    "onnxruntime": "onnxruntime-directml",
    "numpy": "numpy",
    "requests": "requests",
    "multipart": "python-multipart",
}


def banner():
    print("=" * 58)
    print("   AIFACE  -  AI Face Recognition & Identification")
    print("=" * 58)


def missing_packages():
    import importlib.util
    missing = []
    for module, pip_name in REQUIRED.items():
        if importlib.util.find_spec(module) is None:
            missing.append(pip_name)
    return missing


def ensure_dependencies():
    if FROZEN:  # AIFACE.exe already bundles everything
        print("[1/4] Dependencies OK.")
        return True
    missing = missing_packages()
    if not missing:
        print("[1/4] Dependencies OK.")
        return True

    print(f"[1/4] Installing missing packages: {', '.join(missing)}")
    print("      (first run only, this can take a few minutes)")
    cmd = [sys.executable, "-m", "pip", "install", *missing]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print("\n[ERROR] Automatic install failed. Please run manually:")
        print(f"        {sys.executable} -m pip install -r requirements.txt")
        return False

    still_missing = missing_packages()
    if still_missing:
        print(f"\n[ERROR] Still missing after install: {', '.join(still_missing)}")
        return False
    print("      Install complete.")
    return True


def find_free_port(start=DEFAULT_PORT, attempts=PORT_ATTEMPTS):
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((HOST, port))
                return port
            except OSError:
                continue
    return None


def open_browser_when_ready(url, health_url, timeout=180):
    """Poll /api/health, then open the dashboard. Loading the AI models takes
    a while on a cold start, so we wait rather than opening a dead page."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as r:
                if r.status == 200:
                    print(f"\n[4/4] Ready! Opening {url}\n")
                    print("      Keep this window open while you use AIFACE.")
                    print("      Press Ctrl+C here to stop the server.\n")
                    webbrowser.open(url)
                    return
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    print(f"\n[WARN] Server did not answer in {timeout}s. Try opening {url} manually.")


def main():
    banner()
    os.chdir(PROJECT_ROOT)

    if not ensure_dependencies():
        return 1

    port = find_free_port()
    if port is None:
        print(f"[ERROR] No free port between {DEFAULT_PORT} and {DEFAULT_PORT + PORT_ATTEMPTS}.")
        return 1
    if port != DEFAULT_PORT:
        print(f"[2/4] Port {DEFAULT_PORT} is busy, using {port} instead.")
    else:
        print(f"[2/4] Using port {port}.")

    url = f"http://{HOST}:{port}/"
    print("[3/4] Loading AI models and starting the server...")

    threading.Thread(
        target=open_browser_when_ready,
        args=(url, f"{url}api/health"),
        daemon=True,
    ).start()

    import uvicorn
    from backend.main import app

    try:
        uvicorn.run(app, host=HOST, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    print("\nAIFACE stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
