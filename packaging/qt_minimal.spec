# -*- mode: python ; coding: utf-8 -*-
"""
极简版打包配置 - 最大化减少体积
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
hiddenimports += ["cv2", "cv2.cv2"]  # 最小化 cv2 导入
hiddenimports += collect_submodules("dotenv")

excludes = [
    # Qt 相关 - 只保留核心组件
    "PySide2", "PyQt6", "PyQt5", "PyQt4", "Qt5", "Qt4",
    # 深度学习框架
    "torch", "torchvision", "torchaudio", "tensorflow", "keras", "jax", "mxnet", "paddle",
    # 数据可视化
    "matplotlib", "seaborn", "plotly", "bokeh",
    # 开发工具
    "IPython", "jupyter", "notebook", "qtconsole", "spyder",
    # 测试框架
    "pytest", "nose", "doctest",
    # 调试工具
    "pudb", "ipdb", "cProfile", "profile", "trace", "tracemalloc",
    # 网络/爬虫
    "scrapy", "selenium", "requests_html",
    # 数据库
    "sqlalchemy", "pymysql", "psycopg2", "pymongo", "redis",
    # Web 框架
    "django", "flask", "fastapi", "tornado", "aiohttp", "grpc", "thrift", "zmq", "twisted",
    # 并发
    "gevent", "eventlet", "asyncio.test_utils",
    # 其他
    "concurrent.futures.process",
    "multiprocessing.popen_spawn_win32", "multiprocessing.popen_fork", "multiprocessing.popen_forkserver",
    "ctypes.test", "distutils.tests",
    "email.mime.audio", "email.mime.image", "email.mime.message",
    "http.server", "idlelib", "lib2to3", "pip",
    "pkg_resources.tests", "setuptools.tests", "sphinx",
    "tkinter", "turtle", "turtledemo", "venv", "wsgiref", "xmlrpc",
    "test", "tests", "_test", "__pycache__",
    # OpenCV 不需要的模块
    "cv2.gapi",  # G-API 图形处理
    "cv2.mat_wrapper",
    "cv2.misc",
    "cv2.typing",
    "cv2.utils",
    "cv2.data",  # 级联分类器数据
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
    optimize=2,
)

# 排除不需要的二进制文件
excluded_binaries = [
    # Qt 组件 - 只保留核心
    'Qt6Quick.dll', 'Qt6Qml.dll', 'Qt6QmlMeta.dll', 'Qt6QmlModels.dll', 'Qt6QmlWorkerScript.dll',
    'Qt6VirtualKeyboard.dll', 'Qt6Svg.dll', 'Qt6OpenGL.dll', 
    'Qt6Network.dll',  # 如果不需要网络
    'Qt6Pdf.dll',      # PDF 显示（如果不需要预览 PDF）
    # Qt 插件
    'qtvirtualkeyboardplugin.dll',
    'qtuiotouchplugin.dll',
    'qschannelbackend.dll',  # TLS 后端
    'qopensslbackend.dll',
    'qcertonlybackend.dll',
    'qdirect2d.dll',  # 平台插件
    'qminimal.dll',
    'qoffscreen.dll',
    # OpenCV
    'opencv_videoio_ffmpeg',
    # 其他
    'opengl32sw.dll',
]

a.binaries = [b for b in a.binaries if not any(excluded in b[0] for excluded in excluded_binaries)]

# 排除不需要的数据文件
excluded_datas = [
    # Qt 翻译文件 - 只保留中文和英文
    'translations/qt_ar.qm', 'translations/qt_bg.qm', 'translations/qt_ca.qm',
    'translations/qt_cs.qm', 'translations/qt_da.qm', 'translations/qt_de.qm',
    'translations/qt_es.qm', 'translations/qt_fa.qm', 'translations/qt_fi.qm',
    'translations/qt_fr.qm', 'translations/qt_gd.qm', 'translations/qt_gl.qm',
    'translations/qt_he.qm', 'translations/qt_help',  # 帮助翻译
    'translations/qt_hr.qm', 'translations/qt_hu.qm',
    'translations/qt_it.qm', 'translations/qt_ja.qm', 'translations/qt_ka.qm',
    'translations/qt_ko.qm', 'translations/qt_lg.qm', 'translations/qt_lt.qm',
    'translations/qt_lv.qm', 'translations/qt_nl.qm', 'translations/qt_nn.qm',
    'translations/qt_pl.qm', 'translations/qt_pt_BR.qm', 'translations/qt_pt_PT.qm',
    'translations/qt_ru.qm', 'translations/qt_sk.qm', 'translations/qt_sl.qm',
    'translations/qt_sv.qm', 'translations/qt_tr.qm', 'translations/qt_uk.qm',
    'translations/qtbase_ar.qm', 'translations/qtbase_bg.qm', 'translations/qtbase_ca.qm',
    'translations/qtbase_cs.qm', 'translations/qtbase_da.qm', 'translations/qtbase_de.qm',
    'translations/qtbase_es.qm', 'translations/qtbase_fa.qm', 'translations/qtbase_fi.qm',
    'translations/qtbase_fr.qm', 'translations/qtbase_gd.qm', 'translations/qtbase_he.qm',
    'translations/qtbase_hr.qm', 'translations/qtbase_hu.qm', 'translations/qtbase_it.qm',
    'translations/qtbase_ja.qm', 'translations/qtbase_ka.qm', 'translations/qtbase_ko.qm',
    'translations/qtbase_lg.qm', 'translations/qtbase_lv.qm', 'translations/qtbase_nl.qm',
    'translations/qtbase_nn.qm', 'translations/qtbase_pl.qm', 'translations/qtbase_pt_BR.qm',
    'translations/qtbase_ru.qm', 'translations/qtbase_sk.qm', 'translations/qtbase_sv.qm',
    'translations/qtbase_tr.qm', 'translations/qtbase_uk.qm',
    # Qt 图像格式 - 只保留常用格式
    'imageformats/qgif.dll', 'imageformats/qicns.dll', 'imageformats/qtga.dll',
    'imageformats/qtiff.dll', 'imageformats/qwbmp.dll', 'imageformats/qwebp.dll',
    # OpenCV 数据
    'cv2/data',
]

a.datas = [d for d in a.datas if not any(excluded in d[0] for excluded in excluded_datas)]

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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name="Convert",
)
