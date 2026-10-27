# PyInstaller build for AIFACE.exe  —  run:  python -m PyInstaller AIFACE.spec --noconfirm
# One-folder build (dist/AIFACE/): a one-file .exe would unpack ~500 MB on
# every launch. The face models are bundled so it works offline.
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

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

d, b, h = collect_all("onnxruntime")
datas += d
binaries += b
hiddenimports += h

# insightface 2.x also ships a GUI/"privateframe" app that drags in jax,
# transformers, pyarrow, av... (~600 MB). AIFACE only uses the core API.
_UNUSED = ("insightface.gui", "insightface.app.privateframe", "insightface.commands")
hiddenimports += collect_submodules("insightface", filter=lambda n: not n.startswith(_UNUSED))
datas += collect_data_files("insightface", excludes=["gui/**", "app/privateframe/**", "models/**"])

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "tensorflow", "matplotlib", "pandas", "tkinter", "IPython",
              "mediapipe", "face_recognition", "dlib", "jupyter", "notebook",
              "PySide6", "shiboken6", "PyQt5", "PyQt6", "MetaTrader5", "pytest",
              "sphinx", "docutils", "jax", "jaxlib", "transformers", "pyarrow", "av",
              "pythainlp", "numba", "llvmlite", "insightface.gui",
              "insightface.app.privateframe", "insightface.commands"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AIFACE",
          console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="AIFACE", upx=False)
