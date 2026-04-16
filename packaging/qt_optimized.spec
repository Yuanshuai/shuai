# -*- mode: python ; coding: utf-8 -*-
"""
优化版打包配置 - 排除不需要的 Qt 组件和其他文件以减小体积
"""

import os
from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

project_root = os.path.abspath(os.path.join(SPECPATH, ".."))

# 收集 rapidocr 数据文件，只保留 v5 识别模型
def collect_rapidocr_data():
    all_data = collect_data_files("rapidocr")
    filtered = []
    for src, dst in all_data:
        excluded_models = [
            "ch_PP-OCRv4",
            "ch_PP-OCRv5_mobile_det",
            "ch_ppocr_mobile_v2.0_cls",
        ]
        if any(excluded in src for excluded in excluded_models):
            print(f"[EXCLUDE] {src}")
            continue
        print(f"[INCLUDE] {src}")
        filtered.append((src, dst))
    return filtered

# 收集 PySide6 数据，但排除不需要的组件
def collect_pyside6_data():
    all_data = collect_data_files("PySide6")
    filtered = []
    for src, dst in all_data:
        # 排除不需要的 Qt 组件
        excluded = [
            "Qt6Quick.dll",      # QML/Quick
            "Qt6Qml",            # QML
            "Qt6VirtualKeyboard", # 虚拟键盘
            "qml",               # QML 文件
            "qmltooling",        # QML 工具
            "qtmultimedia",      # 多媒体
            "qtwebengine",       # Web引擎
            "qtwebsockets",      # WebSocket
            "qt3d",              # 3D
            "qtgamepad",         # 游戏手柄
            "qtsensors",         # 传感器
            "qtserialport",      # 串口
            "qtlocation",        # 定位
            "qtnetworkauth",     # 网络认证
            "qtpurchasing",      # 购买
            "qtremoteobjects",   # 远程对象
            "qtscxml",           # SCXML
            "qtspeech",          # 语音
            "qtcharts",          # 图表
            "qtdatavis3d",       # 3D数据可视化
            "qtlottie",          # Lottie动画
            "qtnfc",             # NFC
            "qtpositioning",     # 定位
            "qtquick3d",         # 3D Quick
            "qtquicktimeline",   # Quick时间线
            "qtshadertools",     # 着色器工具
            "qtsvg",             # SVG（如果不需要）
        ]
        if any(excluded_item.lower() in src.lower() for excluded_item in excluded):
            print(f"[EXCLUDE QT] {src}")
            continue
        filtered.append((src, dst))
    return filtered

datas = []
# rapidocr - 只收集 v5 识别模型和配置
datas += collect_rapidocr_data()
# rapid_layout - 只收集 v3 模型和配置
datas += [(os.path.join(project_root, "venv", "Lib", "site-packages", "rapid_layout", "configs"), "rapid_layout/configs")]
datas += [(os.path.join(project_root, "venv", "Lib", "site-packages", "rapid_layout", "models", "pp_doc_layoutv3.onnx"), "rapid_layout/models")]
# pymupdf - 只收集必要的文件
datas += [(os.path.join(project_root, "venv", "Lib", "site-packages", "pymupdf"), "pymupdf")]
# 其他必要文件
datas += [(os.path.join(project_root, ".env"), ".")]
datas += [(os.path.join(project_root, "favicon.ico"), ".")]

hiddenimports = []
hiddenimports += collect_submodules("rapidocr")
hiddenimports += collect_submodules("rapid_layout")
hiddenimports += collect_submodules("fitz")
hiddenimports += collect_submodules("pymupdf")
hiddenimports += collect_submodules("openpyxl")
hiddenimports += collect_submodules("cv2")
hiddenimports += collect_submodules("dotenv")

excludes = [
    # Qt 相关
    "PySide2",
    "PyQt6",
    "PyQt5",
    "PyQt4",
    "Qt5",
    "Qt4",
    # 深度学习框架
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "keras",
    "jax",
    "mxnet",
    "paddle",
    # 数据可视化
    "matplotlib",
    "seaborn",
    "plotly",
    "bokeh",
    # 开发工具
    "IPython",
    "jupyter",
    "notebook",
    "qtconsole",
    "spyder",
    # 测试框架
    "pytest",
    "nose",
    "doctest",
    # 调试工具
    "pudb",
    "ipdb",
    "cProfile",
    "profile",
    "trace",
    "tracemalloc",
    # 网络/爬虫
    "scrapy",
    "selenium",
    "requests_html",
    # 数据库（不需要的）
    "sqlalchemy",
    "pymysql",
    "psycopg2",
    "pymongo",
    "redis",
    # 其他
    "django",
    "flask",
    "fastapi",
    "tornado",
    "aiohttp",
    "grpc",
    "thrift",
    "zmq",
    "twisted",
    "gevent",
    "eventlet",
    "asyncio.test_utils",
    "concurrent.futures.process",
    "multiprocessing.popen_spawn_win32",
    "multiprocessing.popen_fork",
    "multiprocessing.popen_forkserver",
    "ctypes.test",
    "distutils.tests",
    "email.mime.audio",
    "email.mime.image",
    "email.mime.message",
    "http.server",
    "idlelib",
    "lib2to3",
    "pip",
    "pkg_resources.tests",
    "setuptools.tests",
    "sphinx",
    "tkinter",
    "turtle",
    "turtledemo",
    "venv",
    "wsgiref",
    "xmlrpc",
    "test",
    "tests",
    "_test",
    "__pycache__",
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
    optimize=2,  # 启用字节码优化
)

# 手动排除不需要的二进制文件
excluded_binaries = [
    # Qt 相关
    'Qt6Quick.dll',
    'Qt6Qml.dll',
    'Qt6QmlMeta.dll',
    'Qt6QmlModels.dll',
    'Qt6QmlWorkerScript.dll',
    'Qt6VirtualKeyboard.dll',
    'Qt6Svg.dll',
    'Qt6OpenGL.dll',
    'Qt6Network.dll',  # 如果不需要网络功能
    'Qt6Pdf.dll',      # 如果不需要 PDF 显示
    # OpenCV 相关（可选）
    # 'opencv_videoio_ffmpeg',
    # 其他
    'opengl32sw.dll',  # 软件 OpenGL
]

# 过滤二进制文件
a.binaries = [b for b in a.binaries if not any(excluded in b[0] for excluded in excluded_binaries)]

pyz = PYZ(a.pure, optimize=2)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Conversion2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # 启用 UPX 压缩
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    icon=os.path.join(project_root, "favicon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,  # 启用 UPX 压缩
    upx_exclude=[],
    name="Convert",
)
