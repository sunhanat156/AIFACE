# PyInstaller build for AIFACE.exe  —  run:  python -m PyInstaller AIFACE.spec --noconfirm
# One-folder build (dist/AIFACE/): a one-file .exe would unpack ~500 MB on
# every launch. The face models are bundled so it works offline.
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

MODELS = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l")

datas = [
    ("frontend/index.html", "frontend"),
    ("frontend/css/style.css", "frontend/css"),
    ("frontend/js/main.js", "frontend/js"),
    (os.path.join(MODELS, "det_10g.onnx"), "insightface/models/buffalo_l"),
    (os.path.join(MODELS, "w600k_r50.onnx"), "insightface/models/buffalo_l"),
]
binaries = []
hiddenimports = collect_submodules("uvicorn") + collect_submodules("backend") + ["multipart", "python_multipart"]

for pkg in ("onnxruntime", "insightface"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "tensorflow", "matplotlib", "pandas", "tkinter", "IPython",
              "mediapipe", "face_recognition", "dlib", "jupyter", "notebook",
              "PySide6", "shiboken6", "PyQt5", "PyQt6", "MetaTrader5", "pytest",
              "sphinx", "docutils"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AIFACE",
          console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="AIFACE", upx=False)
