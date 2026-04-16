import os

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


project_root = os.path.abspath(os.path.join(SPECPATH, ".."))

# 收集 rapidocr 数据文件，只保留 v5 识别模型和必要文件
def collect_rapidocr_data():
    all_data = collect_data_files("rapidocr")
    filtered = []
    for src, dst in all_data:
        # 排除所有不需要的模型文件
        excluded_models = [
            "ch_PP-OCRv4",           # v4 版本模型
            "ch_PP-OCRv5_mobile_det", # 检测模型 (det)
            "ch_ppocr_mobile_v2.0_cls", # 分类模型 (cls)
        ]
        # 如果路径包含任何排除项，跳过
        if any(excluded in src for excluded in excluded_models):
            print(f"[EXCLUDE] {src}")
            continue
        print(f"[INCLUDE] {src}")
        filtered.append((src, dst))
    return filtered

# 只收集需要的模型文件
datas = []
# rapidocr - 只收集 v5 识别模型和配置
datas += collect_rapidocr_data()
# rapid_layout - 只收集 v3 模型和配置
datas += [(os.path.join(project_root, "venv", "Lib", "site-packages", "rapid_layout", "configs"), "rapid_layout/configs")]
datas += [(os.path.join(project_root, "venv", "Lib", "site-packages", "rapid_layout", "models", "pp_doc_layoutv3.onnx"), "rapid_layout/models")]
# rapid_orientation - 收集配置文件和模型
datas += [(os.path.join(project_root, "venv", "Lib", "site-packages", "rapid_orientation", "config.yaml"), "rapid_orientation")]
datas += [(os.path.join(project_root, "venv", "Lib", "site-packages", "rapid_orientation", "models", "rapid_orientation.onnx"), "rapid_orientation/models")]
# pymupdf
datas += collect_data_files("pymupdf")
datas += [(os.path.join(project_root, ".env"), ".")]
# 图标文件
datas += [(os.path.join(project_root, "favicon.ico"), ".")]

hiddenimports = []
hiddenimports += collect_submodules("rapidocr")
hiddenimports += collect_submodules("rapid_layout")
hiddenimports += collect_submodules("rapid_orientation")
hiddenimports += collect_submodules("fitz")
hiddenimports += collect_submodules("pymupdf")
hiddenimports += collect_submodules("openpyxl")
hiddenimports += collect_submodules("cv2")
hiddenimports += collect_submodules("dotenv")

excludes = [
    "PySide2",
    "PyQt6",
    "PyQt5",
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
    [os.path.join(project_root, "qt.py")],
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
    name="Conversion2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=r"C:\Users\86195\PycharmProjects\pythonProject2\favicon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Convert",
)
