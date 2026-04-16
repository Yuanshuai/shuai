# 图片转 Excel 表格工具 (Image/PDF to Excel Converter)

这是一个强大的文档转换工具，支持将包含表格的图片或 PDF 文件一键转换为可编辑的 Excel 表格。项目集成了先进的文档版面分析、表格结构识别以及文字识别（OCR）技术，支持自动检测与修正文档方向，能够处理跨行跨列等复杂表格结构。

项目包含**桌面端 GUI 应用**和**轻量 HTTP 服务**两种使用方式，方便不同场景下的调用。

## ✨ 核心特性

- **多格式支持**：支持 `jpg`, `png`, `bmp`, `tiff` 等主流图片格式，以及 `pdf` 文档的解析。
- **混合页面处理**：对 PDF 支持扫描件（图像识别）与电子档（矢量提取）混合智能处理。
- **高精度识别**：集成 RapidOCR 与版面分析模型，对复杂表格的框线提取与合并单元格识别有良好效果。
- **自动纠偏**：内置方向检测模型，自动将倒置、旋转的文档摆正后进行识别。
- **双模访问**：
  - 🖥️ **桌面 GUI**：友好的可视化界面，支持拖拽文件、预览处理页面、一键导出合并后的 Excel。
  - 🌐 **HTTP API**：支持通过 Web 接口上传文件，返回结构化 JSON 或直接保存为 Excel。

## 📂 目录结构

项目经过清晰的职责划分，结构如下：

```text
├── app/                  # 应用程序入口层
│   ├── desktop_app.py    # 桌面 GUI 应用程序核心实现
│   └── http_server.py    # 轻量级 HTTP API 服务核心实现
├── algorithms/           # 核心算法层
│   ├── table_detect.py   # 表格区域检测与行列网格提取
│   └── table_recognition.py # 表格单元格内容识别与结构化
├── core/                 # 业务逻辑与处理引擎
│   ├── processor.py      # 文档处理主引擎（串联排版、OCR、表格还原）
│   ├── pdf_loader_v2.py  # PDF 智能加载与解析模块
│   ├── orientation_detector.py # 图像方向检测与纠偏
│   └── qt_compat.py      # Qt GUI 跨版本兼容层 (PySide6 / PyQt5)
├── scripts/              # 工程化脚本
│   └── build_app.py      # 自动化打包脚本
├── packaging/            # PyInstaller 打包配置目录
├── models/               # 深度学习模型存放目录（被 Git 忽略）
├── qt.py                 # 桌面应用启动入口
├── lin.py                # HTTP 服务启动入口
└── build.py              # 打包执行入口
```

## 🚀 快速开始

### 1. 环境准备

建议使用 Python 3.10+。在根目录下创建并激活虚拟环境：

```bash
python -m venv venv
# Windows 激活
.\venv\Scripts\activate
# Linux/macOS 激活
source venv/bin/activate
```

安装依赖（项目可能需要安装 PySide6/PyQt5、opencv-python、rapidocr-onnxruntime 等，请根据报错补齐依赖）：
```bash
pip install -r requirements.txt # 如果有
```

### 2. 运行应用

**启动桌面 GUI 版本：**
```bash
python qt.py
```

**启动 HTTP 服务版本：**
```bash
python lin.py
```
默认会在 `8000` 端口启动，可通过 `/health` 接口检查状态，通过 `/upload` 接口 POST 文件。

### 3. 构建打包

本项目配置了极简的打包脚本，用于生成独立的可执行文件（减小体积）。
```bash
python build.py
```
打包产物将输出在 `dist/` 目录下。

## 🧠 技术栈

- **界面与服务**：PySide6 (Qt) / HTTP Server
- **视觉处理**：OpenCV / Numpy
- **模型推理**：ONNX Runtime
- **文档解析**：PyMuPDF (fitz)
- **OCR与版面分析**：RapidOCR / RapidLayout

## 📝 备注

由于模型文件（如 `*.onnx`）体积较大，已在 `.gitignore` 中被忽略，未包含在代码仓库中。如果你在本地克隆了本项目，需要将必要的 ONNX 模型放置于 `models/` 目录下才能正常运行推理流程。
