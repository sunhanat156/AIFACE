"""Where AIFACE reads its bundled files and writes its data.

Run from source, both are the project folder. Packaged as AIFACE.exe
(PyInstaller), bundled read-only files (frontend, AI models) live in the
unpacked bundle, while the database, config and captured faces go next to
the .exe so they survive restarts and are easy to find.
"""
import os
import sys

FROZEN = getattr(sys, "frozen", False)

if FROZEN:
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable))
    RESOURCE_DIR = getattr(sys, "_MEIPASS", APP_DIR)
else:
    APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESOURCE_DIR = APP_DIR

FRONTEND_DIR = os.path.join(RESOURCE_DIR, "frontend")

# insightface looks for <root>/models/buffalo_l. The .exe ships the models
# inside the bundle; from source the usual ~/.insightface cache is used.
BUNDLED_MODELS_ROOT = os.path.join(RESOURCE_DIR, "insightface")
INSIGHTFACE_ROOT = (
    BUNDLED_MODELS_ROOT
    if os.path.isdir(os.path.join(BUNDLED_MODELS_ROOT, "models", "buffalo_l"))
    else os.path.join(os.path.expanduser("~"), ".insightface")
)
