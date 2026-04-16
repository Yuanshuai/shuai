import os

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


project_root = os.path.abspath(os.path.join(SPECPATH, ".."))

# 收集 rapidocr 数据文件，但排除 v4 模型
def collect_rapidocr_data():
    all_data = collect_data_files("rapidocr")
    filtered = []
    for src, dst in all_data:
        # 排除 v4 模型
        if "ch_PP-OCRv4" in src:
            continue
        filtered.append((src, dst))
    return filtered

# 只收集需要的模型文件
datas = []
# rapidocr - 只收集v5模型和配置
datas += collect_rapidocr_data()
# rapid_layout - 只收集v3模型和配置
datas += [(os.path.join(project_root, "venv", "Lib", "site-packages", "rapid_layout", "configs"), "rapid_layout/configs")]
datas += [(os.path.join(project_root, "venv", "Lib", "site-packages", "rapid_layout", "models", "pp_doc_layoutv3.onnx"), "rapid_layout/models")]
# pymupdf
datas += collect_data_files("pymupdf")
datas += [(os.path.join(project_root, ".env"), ".")]

hiddenimports = []
hiddenimports += collect_submodules("rapidocr")
hiddenimports += collect_submodules("rapid_layout")
hiddenimports += collect_submodules("fitz")
hiddenimports += collect_submodules("pymupdf")
hiddenimports += collect_submodules("openpyxl")
hiddenimports += collect_submodules("cv2")
hiddenimports += collect_submodules("dotenv")

excludes = [
    "PySide6",
    "PySide2",
    "PyQt5",
    "PyQt6",
    "torch",
    "tensorflow",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "unittest",
    "pdb",
    "pydoc",
]

a = Analysis(
    [os.path.join(project_root, "lin.py")],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="lin_server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="lin_server",
)
