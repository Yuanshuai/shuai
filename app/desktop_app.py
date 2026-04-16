"""桌面 GUI 入口，负责文件选择、预览和转换流程编排。"""

import sys
import os
import cv2
import fitz
import numpy as np
from typing import Optional, Dict, List, Set, Tuple, Any
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass
from core.qt_compat import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QGroupBox,
    QMessageBox,
    QCheckBox,
    QComboBox,
    QDialog,
    Qt,
    QThread,
    pyqtSignal,
    QPixmap,
    QImage,
    QFont,
    QDragEnterEvent,
    QDropEvent,
    QPainter,
    QColor,
    QPolygon,
    QPoint,
    QIcon,
    QMenu,
)

from core.processor import DocumentProcessor
from core.pdf_loader_on_demand import PDFOnDemandLoader, LoadPriority
from core.pdf_loader_v2 import PDFLoaderV2
from multiprocessing import freeze_support
from collections import OrderedDict
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 支持的文件格式定义
SUPPORTED_IMG_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
SUPPORTED_PDF_FORMATS = ('.pdf',)
ALL_SUPPORTED_FORMATS = [fmt.lower() for fmt in SUPPORTED_IMG_FORMATS + SUPPORTED_PDF_FORMATS]


class PageType(Enum):
    """页面类型"""
    SCANNED = auto()      # 扫描件（有内嵌图像）
    ELECTRONIC = auto()   # 电子档（有文本层）
    BLANK = auto()        # 空白页


class ProcessMethod(Enum):
    """处理方法"""
    AUTO = auto()         # 自动选择
    IMAGE = auto()        # 图像识别
    VECTOR = auto()       # 矢量提取


@dataclass
class PageProcessConfig:
    """
    页面处理配置
    
    统一存储页面的处理配置，替代原来分散的 page_methods 和状态判断
    """
    page_idx: int
    page_type: PageType
    method: ProcessMethod = ProcessMethod.AUTO
    
    def get_effective_method(self) -> ProcessMethod:
        """获取实际使用的处理方法"""
        if self.page_type == PageType.SCANNED:
            # 扫描件强制使用图像识别
            return ProcessMethod.IMAGE
        elif self.method == ProcessMethod.AUTO:
            # 电子档默认使用矢量提取
            return ProcessMethod.VECTOR
        else:
            # 使用用户指定的方法
            return self.method
    
    def is_image_processing(self) -> bool:
        """是否需要图像处理"""
        return self.get_effective_method() == ProcessMethod.IMAGE
    
    def is_vector_processing(self) -> bool:
        """是否需要矢量提取"""
        return self.get_effective_method() == ProcessMethod.VECTOR


class PageProcessManager:
    """
    页面处理管理器
    
    统一管理所有页面的处理配置
    """
    def __init__(self):
        self._configs: Dict[int, PageProcessConfig] = {}
    
    def register_page(self, page_idx: int, page_type: PageType) -> PageProcessConfig:
        """注册页面"""
        config = PageProcessConfig(page_idx=page_idx, page_type=page_type)
        self._configs[page_idx] = config
        return config
    
    def get_config(self, page_idx: int) -> Optional[PageProcessConfig]:
        """获取页面配置"""
        return self._configs.get(page_idx)
    
    def set_method(self, page_idx: int, method: ProcessMethod) -> bool:
        """设置页面处理方法"""
        if page_idx in self._configs:
            self._configs[page_idx].method = method
            return True
        return False
    
    def get_all_configs(self) -> Dict[int, PageProcessConfig]:
        """获取所有配置"""
        return self._configs.copy()
    
    def get_pages_by_method(self, method: ProcessMethod) -> List[int]:
        """获取使用指定处理方法的所有页面"""
        return [idx for idx, config in self._configs.items() 
                if config.get_effective_method() == method]
    
    def get_image_processing_pages(self) -> List[int]:
        """获取需要图像处理的所有页面"""
        return [idx for idx, config in self._configs.items() 
                if config.is_image_processing()]
    
    def get_vector_processing_pages(self) -> List[int]:
        """获取需要矢量提取的所有页面"""
        return [idx for idx, config in self._configs.items() 
                if config.is_vector_processing()]
    
    def clear(self):
        """清空所有配置"""
        self._configs.clear()


class Logger:
    """日志管理器，支持文件日志和开关控制"""
    
    def __init__(self, log_dir="logs"):
        self.enabled = False
        self.log_dir = log_dir
        self.log_file = None
        self.log_path = None
        
    def enable(self, session_name=None):
        """启用日志记录"""
        self.enabled = True
        os.makedirs(self.log_dir, exist_ok=True)
        
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"pdf_table_extract_{session_name}.log")
        self.log_file = open(self.log_path, 'a', encoding='utf-8')
        self._write_log("="*60)
        self._write_log(f"日志开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log("="*60)
        
    def disable(self):
        """禁用日志记录"""
        if self.log_file:
            self._write_log("="*60)
            self._write_log(f"日志结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._write_log("="*60)
            self.log_file.close()
            self.log_file = None
        self.enabled = False
        
    def _write_log(self, message):
        """写入日志文件"""
        if self.enabled and self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.log_file.write(f"[{timestamp}] {message}\n")
            self.log_file.flush()
            
    def debug(self, message):
        """记录调试信息"""
        self._write_log(f"[DEBUG] {message}")
        
    def info(self, message):
        """记录普通信息"""
        self._write_log(f"[INFO] {message}")
        
    def warning(self, message):
        """记录警告信息"""
        self._write_log(f"[WARNING] {message}")
        
    def error(self, message):
        """记录错误信息"""
        self._write_log(f"[ERROR] {message}")
        
    def get_log_path(self):
        """获取当前日志文件路径"""
        return self.log_path


class PDFPageCache:
    """PDF 页面缓存管理器，支持按需加载和 LRU 缓存"""
    
    def __init__(self, pdf_path, cache_size=10):
        """
        Args:
            pdf_path: PDF 文件路径
            cache_size: 缓存的页面数量（LRU 缓存）
        """
        self.pdf_path = pdf_path
        self.cache_size = cache_size
        self.doc = fitz.open(pdf_path)
        self.total_pages = len(self.doc)
        
        # LRU 缓存：OrderedDict 自动维护访问顺序
        self.cache = OrderedDict()  # page_idx -> image_path
        self.temp_files = []
        
        # 创建 data 文件夹
        self.data_dir = os.path.join(os.path.dirname(sys.executable), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 加载状态
        self.loading_pages = set()  # 正在加载的页面
        self.loaded_pages = set()   # 已加载的页面
        
        # 线程锁
        self.lock = threading.Lock()
    
    def get_page_path(self, page_idx):
        """获取页面图片路径"""
        return os.path.join(self.data_dir, f"pdf_page_{page_idx + 1}.png")
    
    def load_page(self, page_idx):
        """
        加载单页 PDF 到缓存
        
        Args:
            page_idx: 页面索引（0-based）
            
        Returns:
            str: 加载后的图片路径，加载失败返回 None
        """
        if page_idx < 0 or page_idx >= self.total_pages:
            return None
        
        with self.lock:
            # 如果已经在缓存中，直接返回
            if page_idx in self.cache:
                # 更新 LRU 顺序
                self.cache.move_to_end(page_idx)
                return self.cache[page_idx]
            
            # 如果正在加载，等待
            if page_idx in self.loading_pages:
                return None
            
            # 标记为正在加载
            self.loading_pages.add(page_idx)
        
        try:
            # 加载页面
            page = self.doc[page_idx]
            pix = page.get_pixmap(dpi=200, alpha=False)
            out_path = self.get_page_path(page_idx)
            pix.save(out_path)
            
            # 进行小角度微调（投影法 ±5°扫描 30 次纠偏）
            try:
                img = cv2.imread(out_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # 二值化
                    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    
                    # 投影法计算得分
                    def determine_score(arr):
                        histogram = np.sum(arr, axis=2, dtype=float)
                        score = np.sum((histogram[..., 1:] - histogram[..., :-1]) ** 2, axis=1, dtype=float)
                        return score
                    
                    # 旋转图像（彩色图像，白色填充）
                    def rotate_image(image, angle):
                        h, w = image.shape[:2]
                        center = (w//2, h//2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        if len(image.shape) == 3:
                            rotated = cv2.warpAffine(
                                image, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255)
                            )
                        else:
                            rotated = cv2.warpAffine(
                                image, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=255
                            )
                        return rotated
                    
                    # 扫描角度范围 ±5°，30 次
                    angles = np.linspace(-5, 5, 30)
                    img_stack = np.stack([rotate_image(thresh, angle) for angle in angles], axis=0)
                    scores = determine_score(img_stack)
                    best_angle = angles[np.argmax(scores)]

                    corrected = rotate_image(img, best_angle)
                    cv2.imwrite(out_path, corrected)
            except Exception as e:
                pass
            
            # 添加到缓存
            with self.lock:
                self.cache[page_idx] = out_path
                self.temp_files.append(out_path)
                self.loaded_pages.add(page_idx)
                self.loading_pages.discard(page_idx)
                
                # LRU 淘汰：如果缓存超出大小，移除最久未使用的
                while len(self.cache) > self.cache_size:
                    oldest_idx, oldest_path = self.cache.popitem(last=False)
                    # 注意：不删除文件，因为可能还需要
            
            return out_path
            
        except Exception as e:
            with self.lock:
                self.loading_pages.discard(page_idx)
            return None
    
    def load_pages_batch(self, page_indices, progress_callback=None):
        """
        批量加载页面
        
        Args:
            page_indices: 页面索引列表
            progress_callback: 进度回调函数 (current, total)
            
        Returns:
            list: 加载成功的图片路径列表
        """
        results = []
        total = len(page_indices)
        
        for i, page_idx in enumerate(page_indices):
            path = self.load_page(page_idx)
            if path:
                results.append(path)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def preload_pages(self, start_idx, count):
        """
        预加载页面（后台加载，不阻塞）
        
        Args:
            start_idx: 起始页面索引
            count: 加载数量
        """
        def preload_worker():
            for i in range(count):
                page_idx = start_idx + i
                if 0 <= page_idx < self.total_pages:
                    self.load_page(page_idx)
        
        # 启动后台线程预加载
        thread = threading.Thread(target=preload_worker, daemon=True)
        thread.start()
    
    def get_loaded_count(self):
        """获取已加载的页面数量"""
        return len(self.loaded_pages)
    
    def is_page_loaded(self, page_idx):
        """检查页面是否已加载"""
        return page_idx in self.loaded_pages
    
    def get_loaded_pages(self):
        """获取所有已加载的页面索引"""
        return sorted(list(self.loaded_pages))
    
    def cleanup(self):
        """清理资源"""
        if self.doc:
            self.doc.close()


# ---------------------- 转换线程（独立线程，不阻塞 UI） ----------------------
class PDFLoaderThread(QThread):
    """PDF 文件加载线程"""
    progress = pyqtSignal(int, int)
    page_loaded = pyqtSignal(str)
    total_pages_known = pyqtSignal(int)  # 总页数已知信号
    finished = pyqtSignal(list, bool, str, list)
    
    def __init__(self, file_path, load_all=True, parent=None):
        """
        Args:
            file_path: PDF 文件路径
            load_all: 是否加载所有页面（False=只加载前 10 页用于预览）
            parent: 父对象
        """
        super().__init__(parent)
        self.file_path = file_path
        self.load_all = load_all
        self._is_running = True  # 运行标志
    
    def stop(self):
        """停止加载"""
        self._is_running = False
    
    def run(self):
        """执行 PDF 加载"""
        try:
            image_list = []
            temp_files = []
            
            if not self._is_running:
                self.finished.emit([], False, "加载已取消", [])
                return
            
            if self.file_path.lower().endswith(SUPPORTED_PDF_FORMATS):
                # 创建 data 文件夹
                data_dir = os.path.join(os.path.dirname(sys.executable), "data")
                os.makedirs(data_dir, exist_ok=True)
                
                doc = fitz.open(self.file_path)
                total_pages = len(doc)
                
                # 发送总页数已知信号
                self.total_pages_known.emit(total_pages)
                
                # 确定要加载的页面范围
                if self.load_all:
                    pages_to_load = range(total_pages)
                else:
                    # 只加载前 10 页用于预览
                    pages_to_load = range(min(10, total_pages))
                
                for page_num in pages_to_load:
                    # 检查是否被取消
                    if not self._is_running:
                        doc.close()
                        self.finished.emit([], False, "加载已取消", temp_files)
                        return
                    
                    page = doc[page_num]
                    
                    # 发送进度信号
                    self.progress.emit(page_num + 1, total_pages)
                    
                    # 转换页面为图像
                    pix = page.get_pixmap(dpi=200, alpha=False)
                    out_path = os.path.join(data_dir, f"pdf_page_{page_num + 1}.png")
                    pix.save(out_path)
                    
                    # 进行小角度微调（投影法 ±5°扫描 30 次纠偏）
                    try:
                        img = cv2.imread(out_path)
                        if img is not None:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            
                            # 二值化
                            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                            
                            # 投影法计算得分
                            def determine_score(arr):
                                histogram = np.sum(arr, axis=2, dtype=float)
                                score = np.sum((histogram[..., 1:] - histogram[..., :-1]) ** 2, axis=1, dtype=float)
                                return score
                            
                            # 旋转图像（彩色图像，白色填充）
                            def rotate_image(image, angle):
                                h, w = image.shape[:2]
                                center = (w//2, h//2)
                                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                                if len(image.shape) == 3:
                                    rotated = cv2.warpAffine(
                                        image, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(255, 255, 255)
                                    )
                                else:
                                    rotated = cv2.warpAffine(
                                        image, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=255
                                    )
                                return rotated
                            
                            # 扫描角度范围 ±5°，30 次
                            angles = np.linspace(-5, 5, 30)
                            img_stack = np.stack([rotate_image(thresh, angle) for angle in angles], axis=0)
                            scores = determine_score(img_stack)
                            best_angle = angles[np.argmax(scores)]

                            corrected = rotate_image(img, best_angle)
                            cv2.imwrite(out_path, corrected)
                    except Exception as e:
                        pass
                    
                    image_list.append(out_path)
                    temp_files.append(out_path)
                    
                    # 发送单页加载完成信号
                    self.page_loaded.emit(out_path)
                
                doc.close()
                
                if self.load_all:
                    # 全部加载完成
                    self.finished.emit(image_list, True, f"成功加载 PDF 文件，共{total_pages}页", temp_files)
                else:
                    # 只加载了前几页用于预览
                    self.finished.emit(image_list, True, f"已加载前{len(image_list)}页用于预览（共{total_pages}页）", temp_files)
            
            else:
                # 处理单个图像文件
                self.total_pages_known.emit(1)
                self.progress.emit(1, 1)
                image_list.append(self.file_path)
                self.page_loaded.emit(self.file_path)
                self.finished.emit(image_list, True, "成功加载图像文件", [])
                
        except Exception as e:
            self.finished.emit([], False, f"加载失败：{str(e)}", [])


class SinglePageLoaderThread(QThread):
    """单页 PDF 加载线程（用于按需加载）"""
    page_loaded = pyqtSignal(int, str)  # 页码，路径
    
    def __init__(self, pdf_path, page_idx, data_dir, parent=None):
        """
        Args:
            pdf_path: PDF 文件路径
            page_idx: 页面索引（0-based）
            data_dir: 数据目录
            parent: 父对象
        """
        super().__init__(parent)
        self.pdf_path = pdf_path
        self.page_idx = page_idx
        self.data_dir = data_dir
        self._is_running = True
    
    def stop(self):
        self._is_running = False
    
    def run(self):
        """加载单页 PDF"""
        try:
            if not self._is_running:
                return
            
            doc = fitz.open(self.pdf_path)
            if self.page_idx >= len(doc):
                doc.close()
                return
            
            page = doc[self.page_idx]
            pix = page.get_pixmap(dpi=200, alpha=False)
            out_path = os.path.join(self.data_dir, f"pdf_page_{self.page_idx + 1}.png")
            pix.save(out_path)
            
            # 进行小角度微调（投影法 ±5°扫描 30 次纠偏）
            try:
                img = cv2.imread(out_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    
                    def determine_score(arr):
                        histogram = np.sum(arr, axis=2, dtype=float)
                        score = np.sum((histogram[..., 1:] - histogram[..., :-1]) ** 2, axis=1, dtype=float)
                        return score
                    
                    def rotate_image(image, angle):
                        h, w = image.shape[:2]
                        center = (w//2, h//2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        if len(image.shape) == 3:
                            rotated = cv2.warpAffine(
                                image, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255)
                            )
                        else:
                            rotated = cv2.warpAffine(
                                image, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=255
                            )
                        return rotated
                    
                    angles = np.linspace(-5, 5, 30)
                    img_stack = np.stack([rotate_image(thresh, angle) for angle in angles], axis=0)
                    scores = determine_score(img_stack)
                    best_angle = angles[np.argmax(scores)]
                    corrected = rotate_image(img, best_angle)
                    cv2.imwrite(out_path, corrected)
            except Exception as e:
                pass
            
            doc.close()
            self.page_loaded.emit(self.page_idx, out_path)
            
        except Exception as e:
            pass  # 静默失败，因为这是后台加载


class VectorExtractThread(QThread):
    """纯矢量提取线程（不需要模型）"""
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(dict)

    def __init__(self, pdf_path: str, page_indices: List[int], excel_output_path: str, logger=None):
        super().__init__()
        self.pdf_path = pdf_path
        self.page_indices = page_indices
        self.excel_output_path = excel_output_path
        self.logger = logger

    def run(self):
        try:
            from core.pdf_vector_table import PDFVectorTableExtractor, extract_tables_from_pdf
            
            self.log_signal.emit("🚀 开始矢量提取...")
            if self.logger:
                self.logger.info("VectorExtractThread 开始执行")
                self.logger.info(f"PDF路径: {self.pdf_path}")
                self.logger.info(f"页面: {[p+1 for p in self.page_indices]}")
            
            all_tables = []
            
            with fitz.open(self.pdf_path) as doc:
                for page_idx in self.page_indices:
                    if page_idx >= len(doc):
                        continue
                    
                    page = doc[page_idx]
                    self.log_signal.emit(f"   正在提取第 {page_idx + 1} 页...")
                    if self.logger:
                        self.logger.info(f"提取第 {page_idx + 1} 页矢量表格")
                    
                    try:
                        extractor = PDFVectorTableExtractor()
                        table = extractor.extract_table_from_page(page, debug=False)
                        
                        if table and table.cells:
                            # 检查是否有文本内容
                            total_text = sum(len(c.text.strip()) for c in table.cells)
                            if total_text > 0:
                                all_tables.append({
                                    'page_index': page_idx,
                                    'table': table
                                })
                                self.log_signal.emit(f"   ✅ 第 {page_idx + 1} 页提取到 {table.rows}x{table.cols} 表格")
                            else:
                                self.log_signal.emit(f"   ⚠️ 第 {page_idx + 1} 页表格无文本内容")
                        else:
                            self.log_signal.emit(f"   ⚠️ 第 {page_idx + 1} 页未检测到表格")
                    
                    except Exception as e:
                        self.log_signal.emit(f"   ❌ 第 {page_idx + 1} 页提取失败: {str(e)}")
                        if self.logger:
                            self.logger.error(f"第 {page_idx + 1} 页提取失败: {e}")
            
            # 转换为 Region 对象并保存
            regions = []
            for item in all_tables:
                page_idx = item['page_index']
                table = item['table']
                
                region = {
                    'page_index': page_idx,
                    'label': 'table',
                    'score': 1.0,
                    'bbox': [
                        min(c.x1 for c in table.cells),
                        min(c.y1 for c in table.cells),
                        max(c.x2 for c in table.cells),
                        max(c.y2 for c in table.cells)
                    ],
                    'text': '',
                    'meta': {
                        'rows': table.rows,
                        'cols': table.cols,
                        'grid': table.to_grid(),
                        'cells': [
                            {
                                'row': c.row,
                                'col': c.col,
                                'rowspan': c.rowspan,
                                'colspan': c.colspan,
                                'text': c.text,
                                'bbox': [c.x1, c.y1, c.x2, c.y2]
                            }
                            for c in table.cells
                        ]
                    }
                }
                regions.append(region)
            
            # 保存到 Excel
            if regions and self.excel_output_path:
                try:
                    from algorithms.table_recognition import save_tables_to_single_workbook, TableResult, TableCell
                    tables_data = []
                    for r in regions:
                        if 'meta' in r and 'grid' in r['meta']:
                            grid = r['meta']['grid']
                            if grid and len(grid) > 0:
                                # 创建 TableResult 对象
                                header = grid[0] if grid else []
                                first_row = grid[1] if len(grid) > 1 else []
                                
                                # 从 meta 中提取 cells 信息（包含 rowspan 和 colspan）
                                cells = []
                                if 'cells' in r['meta']:
                                    for cell_meta in r['meta']['cells']:
                                        cells.append(TableCell(
                                            row=cell_meta.get('row', 0),
                                            col=cell_meta.get('col', 0),
                                            bbox=cell_meta.get('bbox', [0, 0, 0, 0]),
                                            text=cell_meta.get('text', ''),
                                            rowspan=cell_meta.get('rowspan', 1),
                                            colspan=cell_meta.get('colspan', 1)
                                        ))
                                
                                table_result = TableResult(
                                    rows=len(grid),
                                    cols=len(grid[0]) if grid else 0,
                                    cells=cells,
                                    grid=grid,
                                    header=header,
                                    first_row=first_row
                                )
                                page_num = r['page_index'] + 1
                                tables_data.append((f"Page_{page_num}", table_result))
                    
                    if tables_data:
                        save_tables_to_single_workbook(tables_data, self.excel_output_path)
                        self.log_signal.emit(f"✅ Excel已保存: {self.excel_output_path}")
                except Exception as e:
                    self.log_signal.emit(f"⚠️ 保存Excel失败: {str(e)}")
            
            result = {
                'regions': regions,
                'tables': all_tables,
                'excel_path': self.excel_output_path if regions else None
            }
            
            self.log_signal.emit(f"✅ 矢量提取完成，共 {len(all_tables)} 个表格")
            self.finish_signal.emit(result)
            
        except Exception as e:
            error_msg = f"矢量提取失败: {str(e)}"
            self.log_signal.emit(f"❌ {error_msg}")
            if self.logger:
                self.logger.error(error_msg)
            self.finish_signal.emit({})


class ConversionThread(QThread):
    """转换处理线程"""
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(dict)

    def __init__(self, processor, image_list, excel_output_path, pdf_path=None, page_indices=None, 
                 scan_pages=None, text_vector_pages=None, text_image_pages=None, logger=None,
                 page_idx_to_image=None, pdf_loader_v2=None, dpi=200):
        super().__init__()
        self.processor = processor
        self.image_list = image_list
        self.excel_output_path = excel_output_path
        self.pdf_path = pdf_path
        self.page_indices = page_indices
        self.scan_pages = scan_pages or []  # 扫描件页面索引（强制图像处理）
        self.text_vector_pages = text_vector_pages or []  # 电子档页面索引（矢量提取）
        self.text_image_pages = text_image_pages or []  # 电子档页面索引（图像识别）
        self.logger = logger  # 日志记录器
        self.page_idx_to_image = page_idx_to_image or {}  # 页面索引到图像路径的映射
        self.pdf_loader_v2 = pdf_loader_v2  # PDF加载器，用于后台生成DPI图像
        self.dpi = dpi  # 图像渲染DPI

    def _merge_tables(self, table_results: List[Tuple[int, Any]]) -> List[Tuple[str, Any]]:
        """
        合并表格 - 如果表头相同或是连续数据，则合并
        
        Args:
            table_results: [(page_num, TableResult), ...]
            
        Returns:
            [(sheet_name, TableResult), ...]
        """
        if not table_results:
            return []
        
        from algorithms.table_recognition import TableResult
        from typing import Optional, Tuple, List, Any
        
        def _header_key(header):
            """生成表头键用于比较"""
            if not isinstance(header, list):
                return None
            norm = []
            for v in header:
                s = str(v or "").strip()
                s = " ".join(s.split())
                norm.append(s)
            if not any(norm):
                return None
            return tuple(norm)
        
        def _check_row_continuity(prev_table, curr_table):
            """检查两行数据是否连续"""
            result = {"is_continuous": False}
            
            if not prev_table.grid or not curr_table.grid:
                return result
            
            prev_last_row = prev_table.grid[-1] if prev_table.grid else None
            curr_first_row = curr_table.grid[0] if curr_table.grid else None
            
            if not prev_last_row or not curr_first_row:
                return result
            
            # 检查是否有连续数字关系
            continuous_count = 0
            sequence_indicators = 0  # 序号列指示器
            
            for prev_val, curr_val in zip(prev_last_row, curr_first_row):
                prev_str = str(prev_val or "").strip()
                curr_str = str(curr_val or "").strip()
                
                # 尝试解析为数字
                try:
                    prev_num = float(prev_str)
                    curr_num = float(curr_str)
                    # 严格连续：+1
                    if curr_num == prev_num + 1:
                        continuous_count += 1
                        sequence_indicators += 1
                    # 宽松连续：递增即可（适用于跨页表格）
                    elif curr_num > prev_num:
                        continuous_count += 0.5
                except:
                    # 检查是否完全相同
                    if prev_str == curr_str and prev_str:
                        continuous_count += 1
            
            # 判断条件：
            # 1. 有明确的序号列（+1递增）
            # 2. 或者超过一半的列满足连续条件
            total_cols = len(prev_last_row)
            if sequence_indicators >= 1 or continuous_count >= total_cols // 2:
                result["is_continuous"] = True
            
            return result
        
        # 按页面顺序排序
        sorted_results = sorted(table_results, key=lambda x: x[0])
        
        merged_tables = []
        last_header_key = None
        last_table_structure = None
        
        for page_num, table in sorted_results:
            header_key = _header_key(table.header)
            current_structure = (table.rows, table.cols)
            
            should_merge = False
            merge_mode = "none"
            
            if merged_tables:
                # 情况1：表头相同
                if header_key is not None and header_key == last_header_key:
                    should_merge = True
                    merge_mode = "header"
                # 情况2：检查是否连续
                elif last_table_structure is not None:
                    prev_rows, prev_cols = last_table_structure
                    curr_rows, curr_cols = current_structure
                    if prev_cols == curr_cols:
                        prev_name, prev_table = merged_tables[-1]
                        continuity = _check_row_continuity(prev_table, table)
                        if continuity["is_continuous"]:
                            should_merge = True
                            merge_mode = "continuous"
            
            if should_merge and merged_tables:
                prev_name, prev_table = merged_tables[-1]
                merged_grid = list(prev_table.grid)
                
                if merge_mode == "header":
                    # 表头相同，跳过当前表格的表头行
                    if len(table.grid) >= 2:
                        merged_grid.extend(table.grid[1:])
                else:
                    # 连续数据，保留所有行
                    merged_grid.extend(table.grid)
                
                merged_table = TableResult(
                    rows=len(merged_grid),
                    cols=prev_table.cols,
                    cells=[],
                    grid=merged_grid,
                    header=prev_table.header,
                    first_row=prev_table.first_row,
                )
                merged_tables[-1] = (prev_name, merged_table)
            else:
                # 新建工作表
                name = f"Page_{page_num}_table_{len(merged_tables) + 1}"
                merged_tables.append((name, table))
                last_header_key = header_key
                last_table_structure = current_structure
        
        return merged_tables

    def run(self):
        try:
            self.log_signal.emit("🚀 开始处理文件...")
            
            if self.logger:
                self.logger.info("="*60)
                self.logger.info("ConversionThread 开始执行")
                self.logger.info(f"PDF路径: {self.pdf_path}")
                self.logger.info(f"Excel输出: {self.excel_output_path}")
                self.logger.info(f"扫描件页面: {[p+1 for p in self.scan_pages]}")
                self.logger.info(f"矢量提取页面: {[p+1 for p in self.text_vector_pages]}")
                self.logger.info(f"图像识别页面: {[p+1 for p in self.text_image_pages]}")
            
            all_regions = []
            
            # 使用传入的页面索引到图像路径的映射
            page_idx_to_image = self.page_idx_to_image
            
            if self.logger:
                self.logger.debug(f"页面到图像映射: {page_idx_to_image}")
            
            # 处理扫描件页面（强制图像处理）
            if self.scan_pages:
                # 检查是否有 processor（需要模型）
                if not self.processor:
                    self.log_signal.emit("❌ 扫描件处理需要模型，但模型未加载")
                    self.finish_signal.emit({})
                    return
                
                self.log_signal.emit(f"🖼️ 处理 {len(self.scan_pages)} 页扫描件（强制图像识别）...")
                
                scan_images = []
                for idx in self.scan_pages:
                    img_path = None
                    # 优先使用已提供的图像路径
                    if idx in page_idx_to_image:
                        img_path = page_idx_to_image[idx]
                    # 如果没有且提供了pdf_loader_v2，在后台生成图像
                    elif self.pdf_loader_v2:
                        img_path = self.pdf_loader_v2.get_page_for_processing(idx, dpi=self.dpi)
                    
                    if img_path:
                        scan_images.append(img_path)
                
                if scan_images:
                    result = self.processor.process_images(scan_images, excel_output_path=None, page_indices=self.scan_pages)
                    
                    if result and 'regions' in result:
                        all_regions.extend(result['regions'])
            
            # 处理电子档页面 - 矢量提取（不需要模型）
            if self.text_vector_pages and self.pdf_path and os.path.exists(self.pdf_path):
                self.log_signal.emit(f"📄 处理 {len(self.text_vector_pages)} 页电子档（矢量提取）...")
                if self.logger:
                    self.logger.info(f"开始处理 {len(self.text_vector_pages)} 页电子档（矢量提取）")
                    self.logger.debug(f"矢量提取页面索引: {self.text_vector_pages}")
                
                try:
                    import fitz
                    from core.pdf_vector_table import PDFVectorTableExtractor
                    
                    vector_tables = []
                    with fitz.open(self.pdf_path) as doc:
                        for page_idx in self.text_vector_pages:
                            if page_idx < len(doc):
                                page = doc[page_idx]
                                self.log_signal.emit(f"   正在提取第 {page_idx + 1} 页...")
                                if self.logger:
                                    self.logger.info(f"提取第 {page_idx + 1} 页矢量表格")
                                
                                # 直接使用 PDFVectorTableExtractor，不需要 processor
                                extractor = PDFVectorTableExtractor()
                                table = extractor.extract_table_from_page(page, debug=False)
                                
                                if self.logger:
                                    self.logger.debug(f"第 {page_idx + 1} 页提取结果: {table is not None}")
                                
                                if table and table.cells:
                                    # 检查是否有文本内容
                                    total_text = sum(len(c.text.strip()) for c in table.cells)
                                    if total_text > 0:
                                        # 创建 Region 对象
                                        region = {
                                            'page_index': page_idx,
                                            'label': 'table',
                                            'score': 1.0,
                                            'bbox': [
                                                min(c.x1 for c in table.cells),
                                                min(c.y1 for c in table.cells),
                                                max(c.x2 for c in table.cells),
                                                max(c.y2 for c in table.cells)
                                            ],
                                            'text': '',
                                            'meta': {
                                                'rows': table.rows,
                                                'cols': table.cols,
                                                'grid': table.to_grid(),
                                                'cells': [
                                                    {
                                                        'row': c.row,
                                                        'col': c.col,
                                                        'rowspan': c.rowspan,
                                                        'colspan': c.colspan,
                                                        'text': c.text,
                                                        'bbox': [c.x1, c.y1, c.x2, c.y2]
                                                    }
                                                    for c in table.cells
                                                ]
                                            }
                                        }
                                        vector_tables.append(region)
                                        self.log_signal.emit(f"   ✅ 第 {page_idx + 1} 页提取到 {table.rows}x{table.cols} 表格")
                                    else:
                                        self.log_signal.emit(f"   ⚠️ 第 {page_idx + 1} 页表格无文本内容")
                                else:
                                    self.log_signal.emit(f"   ⚠️ 第 {page_idx + 1} 页未检测到表格")
                                    # 矢量提取失败，如果提供了processor则回退到图像处理
                                    if self.processor:
                                        if self.logger:
                                            self.logger.warning(f"第 {page_idx + 1} 页矢量提取失败，回退到图像处理")
                                        self.log_signal.emit(f"   第 {page_idx + 1} 页矢量提取失败，动态渲染并回退到图像识别...")
                                        
                                        # 动态渲染页面
                                        temp_path = None
                                        try:
                                            import tempfile
                                            temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                                            temp_img.close()
                                            temp_path = temp_img.name
                                            
                                            # 渲染页面为图像 (200 DPI)
                                            mat = fitz.Matrix(200/72, 200/72)
                                            pix = page.get_pixmap(matrix=mat)
                                            pix.save(temp_path)
                                            
                                            # 使用图像识别处理
                                            result = self.processor.process_images([temp_path], excel_output_path=None)
                                            if result and 'regions' in result:
                                                all_regions.extend(result['regions'])
                                                self.log_signal.emit(f"   ✅ 第 {page_idx + 1} 页图像识别回退成功")
                                        except Exception as render_err:
                                            self.log_signal.emit(f"   ❌ 第 {page_idx + 1} 页回退失败: {str(render_err)}")
                                        finally:
                                            # 确保清理临时文件
                                            if temp_path:
                                                try:
                                                    os.unlink(temp_path)
                                                except:
                                                    pass
                    
                    # 添加矢量提取的表格到结果
                    if vector_tables:
                        self.log_signal.emit(f"   矢量提取识别到 {len(vector_tables)} 个表格")
                        if self.logger:
                            self.logger.info(f"矢量提取共识别到 {len(vector_tables)} 个表格")
                        all_regions.extend(vector_tables)
                    
                except Exception as e:
                    self.log_signal.emit(f"   ⚠️ 矢量提取异常：{str(e)}")
                    if self.logger:
                        self.logger.error(f"矢量提取异常: {str(e)}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                    
                    # 异常时如果提供了processor则回退到图像处理
                    if self.processor and self.pdf_path:
                        self.log_signal.emit(f"   整体回退到图像处理...")
                        try:
                            import tempfile
                            with fitz.open(self.pdf_path) as fallback_doc:
                                for idx in self.text_vector_pages:
                                    if idx < len(fallback_doc):
                                        temp_path = None
                                        try:
                                            fallback_page = fallback_doc[idx]
                                            temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                                            temp_img.close()
                                            temp_path = temp_img.name
                                            
                                            # 渲染页面
                                            mat = fitz.Matrix(200/72, 200/72)
                                            pix = fallback_page.get_pixmap(matrix=mat)
                                            pix.save(temp_path)
                                            
                                            # 图像识别
                                            result = self.processor.process_images([temp_path], excel_output_path=None)
                                            if result and 'regions' in result:
                                                all_regions.extend(result['regions'])
                                        except Exception as page_err:
                                            self.log_signal.emit(f"   ❌ 第 {idx+1} 页回退失败: {str(page_err)}")
                                        finally:
                                            # 确保清理临时文件
                                            if temp_path:
                                                try:
                                                    os.unlink(temp_path)
                                                except:
                                                    pass
                        except Exception as doc_err:
                            self.log_signal.emit(f"   ❌ 回退处理失败: {str(doc_err)}")
            
            # 处理电子档页面 - 图像识别（用户手动选择）
            if self.text_image_pages:
                # 检查是否有 processor（需要模型）
                if not self.processor:
                    self.log_signal.emit("❌ 图像识别需要模型，但模型未加载")
                    self.finish_signal.emit({})
                    return
                
                self.log_signal.emit(f"🖼️ 处理 {len(self.text_image_pages)} 页电子档（用户选择图像识别）...")
                
                text_images = []
                for idx in self.text_image_pages:
                    img_path = None
                    # 优先使用已提供的图像路径
                    if idx in page_idx_to_image:
                        img_path = page_idx_to_image[idx]
                    # 如果没有且提供了pdf_loader_v2，在后台生成图像
                    elif self.pdf_loader_v2:
                        img_path = self.pdf_loader_v2.get_page_for_processing(idx, dpi=self.dpi)
                    
                    if img_path:
                        text_images.append(img_path)
                
                if text_images:
                    result = self.processor.process_images(text_images, excel_output_path=None, page_indices=self.text_image_pages)
                    
                    if result and 'regions' in result:
                        all_regions.extend(result['regions'])
            
            # 如果没有分类，使用默认处理
            if not self.scan_pages and not self.text_vector_pages and not self.text_image_pages:
                if self.logger:
                    self.logger.info("没有页面分类，使用默认处理")
                
                if self.pdf_path and os.path.exists(self.pdf_path):
                    self.log_signal.emit(f"📄 检测到 PDF 文件，尝试矢量表格提取...")
                    if self.logger:
                        self.logger.info(f"调用 process_pdf_path: {self.pdf_path}")
                    
                    result = self.processor.process_pdf_path(
                        self.pdf_path, 
                        excel_output_path=self.excel_output_path,
                        max_pages=None
                    )
                    
                    if self.logger:
                        self.logger.info(f"矢量提取完成: {result}")
                    
                    self.finish_signal.emit(result)
                    return
                else:
                    if self.logger:
                        self.logger.info(f"调用 process_images: {len(self.image_list)} 张图像")
                    
                    result = self.processor.process_images(self.image_list, excel_output_path=self.excel_output_path)
                    
                    if self.logger:
                        self.logger.info(f"图像处理完成: {result}")
                    
                    self.finish_signal.emit(result)
                    return
            
            # 合并结果并保存
            if self.logger:
                self.logger.info(f"开始保存结果，共 {len(all_regions)} 个区域")
            
            # 提取表格数据并保存到 Excel（不依赖 processor）
            excel_path = None
            if all_regions and self.excel_output_path:
                try:
                    from algorithms.table_recognition import save_tables_to_single_workbook, TableResult
                    
                    # 先创建 TableResult 对象列表
                    table_results = []
                    for r in all_regions:
                        if isinstance(r, dict) and 'meta' in r and 'grid' in r['meta']:
                            grid = r['meta']['grid']
                            if grid and len(grid) > 0:
                                header = grid[0] if grid else []
                                first_row = grid[1] if len(grid) > 1 else []
                                
                                # 从 meta 中提取 cells 信息（包含 rowspan 和 colspan）
                                cells = []
                                if 'cells' in r['meta']:
                                    from algorithms.table_recognition import TableCell
                                    for cell_meta in r['meta']['cells']:
                                        cells.append(TableCell(
                                            row=cell_meta.get('row', 0),
                                            col=cell_meta.get('col', 0),
                                            bbox=cell_meta.get('bbox', [0, 0, 0, 0]),
                                            text=cell_meta.get('text', ''),
                                            rowspan=cell_meta.get('rowspan', 1),
                                            colspan=cell_meta.get('colspan', 1)
                                        ))
                                
                                table_result = TableResult(
                                    rows=len(grid),
                                    cols=len(grid[0]) if grid else 0,
                                    cells=cells,
                                    grid=grid,
                                    header=header,
                                    first_row=first_row
                                )
                                page_num = r.get('page_index', 0) + 1
                                table_results.append((page_num, table_result))
                    
                    # 应用表格合并逻辑
                    tables_data = self._merge_tables(table_results)
                    
                    if tables_data:
                        save_tables_to_single_workbook(tables_data, self.excel_output_path)
                        excel_path = self.excel_output_path
                        self.log_signal.emit(f"✅ Excel已保存: {excel_path}")
                        if len(tables_data) < len(table_results):
                            self.log_signal.emit(f"   （已合并 {len(table_results)} 个表格为 {len(tables_data)} 个工作表）")
                    else:
                        self.log_signal.emit("⚠️ 没有表格数据需要保存")
                        
                except Exception as e:
                    self.log_signal.emit(f"⚠️ 保存Excel失败: {str(e)}")
                    if self.logger:
                        self.logger.error(f"保存Excel失败: {e}")
            
            if self.logger:
                self.logger.info(f"Excel导出完成: {excel_path}")
            
            result = {
                'type': 'mixed',
                'regions': all_regions,
                'tables': all_regions,
                'excel_path': excel_path or self.excel_output_path,
                'total_regions': len(all_regions),
                'non_table_text': '',
                'rotate_preds': []
            }
            
            if self.logger:
                self.logger.info(f"ConversionThread 执行完成，共 {len(all_regions)} 个区域")
                self.logger.info("="*60)
            
            self.finish_signal.emit(result)
            
        except Exception as e:
            self.log_signal.emit(f"❌ 转换异常：{str(e)}")
            if self.logger:
                self.logger.error(f"转换异常: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
            import traceback
            self.log_signal.emit(traceback.format_exc())
            self.finish_signal.emit({})


class RotateThread(QThread):
    """旋转线程，避免阻塞UI"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  # 当前进度, 总数
    finish_signal = pyqtSignal(int, int, int)  # 成功数, 总数, 角度

    def __init__(self, image_list, pages_to_rotate, angle):
        super().__init__()
        self.image_list = image_list
        self.pages_to_rotate = pages_to_rotate
        self.angle = angle % 360  # 标准化角度

    def run(self):
        try:
            total_pages = len(self.pages_to_rotate)
            rotated_count = 0
            skipped_count = 0

            for i, page_idx in enumerate(sorted(self.pages_to_rotate)):
                self.progress_signal.emit(i + 1, total_pages)

                if page_idx >= len(self.image_list):
                    self.log_signal.emit(f"跳过第{page_idx + 1}页（超出范围）")
                    skipped_count += 1
                    continue

                img_path = self.image_list[page_idx]
                # 检查路径是否有效
                if not img_path:
                    self.log_signal.emit(f"跳过第{page_idx + 1}页（尚未加载）")
                    skipped_count += 1
                    continue
                # 读取图像
                img = cv2.imread(img_path)
                if img is None:
                    self.log_signal.emit(f"跳过第{page_idx + 1}页（无法读取）")
                    skipped_count += 1
                    continue

                # 根据角度选择最优旋转方法
                if self.angle == 90:
                    # 顺时针90度：转置后水平翻转
                    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif self.angle == 180:
                    # 180度：直接翻转
                    rotated = cv2.rotate(img, cv2.ROTATE_180)
                elif self.angle == 270:
                    # 顺时针270度（逆时针90度）：转置后垂直翻转
                    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    # 其他角度使用仿射变换
                    (h, w) = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, self.angle, 1.0)

                    # 计算旋转后的图像大小
                    cos = np.abs(M[0, 0])
                    sin = np.abs(M[0, 1])
                    new_w = int((h * sin) + (w * cos))
                    new_h = int((h * cos) + (w * sin))

                    # 调整旋转矩阵的平移部分
                    M[0, 2] += (new_w / 2) - center[0]
                    M[1, 2] += (new_h / 2) - center[1]

                    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                                            flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_REPLICATE)

                # 保存旋转后的图像（覆盖原文件）
                cv2.imwrite(img_path, rotated)
                rotated_count += 1
                self.log_signal.emit(f"✅ 已旋转第{page_idx + 1}页 {self.angle}度")

            self.finish_signal.emit(rotated_count, total_pages, self.angle)
        except Exception as e:
            self.log_signal.emit(f"❌ 旋转异常：{str(e)}")
            self.finish_signal.emit(0, len(self.pages_to_rotate), self.angle)


# ---------------------- 模型加载线程 ----------------------
class ModelLoadThread(QThread):
    """后台加载模型线程"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object)  # 返回加载好的 processor
    
    def run(self):
        try:
            self.log_signal.emit("⏳ 正在加载模型...")
            start_time = time.time()
            
            processor = DocumentProcessor()
            
            elapsed = time.time() - start_time
            self.log_signal.emit(f"✅ 模型加载完成（{elapsed:.1f}秒）")
            self.finished_signal.emit(processor)
            
        except Exception as e:
            self.log_signal.emit(f"❌ 模型加载失败: {str(e)}")
            self.finished_signal.emit(None)


# ---------------------- 主窗口（现代化UI，支持拖拽、预览、参数配置） ----------------------
class TableConverterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 窗口基础配置
        self.setWindowTitle("PDF/图片转表格工具")
        self.setAcceptDrops(True)  # 启用拖拽功能
        
        # 获取屏幕尺寸并设置窗口为全屏
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        self.showMaximized()  # 默认最大化显示

        # 初始化实例变量
        self.selected_file = ""
        self.image_list = []  # 存储预览用的图像路径列表
        self.temp_files = []  # 存储 PDF 转换后的临时图像路径
        self.current_preview_idx = 0
        self.output_path = "output.xlsx"
        self.last_result = {}
        self.processor = None  # 将在后台线程中初始化
        self.font = QFont("微软雅黑", 10)
        self.setFont(self.font)
        
        # 模型加载状态
        self.model_loaded = False
        self.model_loading = False
        
        # PDF 按需加载相关
        self.pdf_loader: Optional[PDFOnDemandLoader] = None  # PDF 按需加载器
        self.pdf_doc = None  # PDF 文档对象
        self.total_pdf_pages = 0  # PDF 总页数
        self.loaded_pages = set()  # 已加载的页面索引集合
        self.background_loader = None  # 后台加载线程
        
        # 选择模式状态变量
        self.selection_mode = False  # 是否在选择模式
        self.selected_pages = set()  # 选中的页面索引集合
        self.selection_start_page = -1  # 选择开始的页面
        self.ctrl_pressed = False  # Ctrl 键是否按下
        self.hover_region = None  # 当前悬停的区域
        
        # 页面处理管理器（替代原来的 page_methods）
        self.process_manager = PageProcessManager()
        
        # 记录当前使用的DPI，用于检测DPI变化
        self.current_dpi = 200

        # 构建 UI 界面
        self._init_ui()
        
        # 安装事件过滤器以支持全局快捷键
        self.installEventFilter(self)

    def _start_model_loading(self):
        """启动后台模型加载"""
        if self.model_loading or self.model_loaded:
            return
        
        self.model_loading = True
        self.log_text.append("⏳ 正在后台加载模型...")
        
        self.model_load_thread = ModelLoadThread()
        self.model_load_thread.log_signal.connect(self.log_text.append)
        self.model_load_thread.finished_signal.connect(self._on_model_loaded)
        self.model_load_thread.start()

    def _on_model_loaded(self, processor):
        """模型加载完成回调"""
        self.model_loading = False
        
        if processor:
            self.processor = processor
            self.model_loaded = True
            self.log_text.append("✅ 模型已就绪")
        else:
            self.processor = None
            self.model_loaded = False
            self.log_text.append("❌ 模型加载失败，请检查环境或重启程序")
            QMessageBox.critical(self, "错误", "模型加载失败，请检查环境或重启程序")

    def _ensure_model_loaded(self):
        """确保模型已加载，如果未加载则自动启动加载，如果正在加载则等待"""
        if self.model_loaded:
            return True
        
        if self.model_loading:
            self.log_text.append("⏳ 等待模型加载完成...")
            # 等待模型加载完成
            while self.model_loading:
                QApplication.processEvents()
                time.sleep(0.1)
            return self.model_loaded
        
        # 未加载且未在加载中，自动启动加载
        self._start_model_loading()
        
        # 等待加载完成
        self.log_text.append("⏳ 等待模型加载完成...")
        while self.model_loading:
            QApplication.processEvents()
            time.sleep(0.1)
        return self.model_loaded

    def _init_ui(self):
        """初始化用户界面布局"""
        # 中心部件与主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧面板（文件选择+参数配置+操作按钮）
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel)

        # 1. 文件选择分组
        self._build_file_group(left_layout)

        # 2. 转换参数分组
        self._build_param_group(left_layout)

        # 3. 操作按钮分组
        self._build_button_group(left_layout)

        # 4. 日志区域（移到左侧底部）
        self._build_log_group(left_layout)

        # 右侧面板（仅预览）
        self._build_right_panel(main_layout)

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪 - 支持拖拽PDF/图片文件到窗口")

    def _build_file_group(self, parent_layout):
        """构建文件选择分组"""
        file_group = QGroupBox("文件选择（支持拖拽）")
        file_layout = QVBoxLayout(file_group)

        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setStyleSheet("background-color: #f5f5f5;")
        file_layout.addWidget(self.file_edit)

        file_btn = QPushButton("选择文件")
        file_btn.clicked.connect(self.select_file)
        file_btn.setStyleSheet("""
            QPushButton {
                padding: 6px;
                background-color: #e3f2fd;
                border: 1px solid #90caf9;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #bbdefb;
                border: 1px solid #64b5f6;
            }
            QPushButton:pressed {
                background-color: #90caf9;
            }
        """)
        file_layout.addWidget(file_btn)

        parent_layout.addWidget(file_group)

    def _build_param_group(self, parent_layout):
        """构建转换参数分组"""
        param_group = QGroupBox("输出配置")
        param_layout = QVBoxLayout(param_group)

        param_layout.addWidget(QLabel("Excel保存路径（文件或目录）："))
        self.outfile_edit = QLineEdit(self.output_path)
        self.outfile_edit.setEnabled(True)  # 始终启用
        param_layout.addWidget(self.outfile_edit)

        outfile_btn = QPushButton("选择 Excel 保存路径")
        outfile_btn.clicked.connect(self.select_outfile)
        outfile_btn.setStyleSheet("""
            QPushButton {
                padding: 6px;
                background-color: #e3f2fd;
                border: 1px solid #90caf9;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #bbdefb;
                border: 1px solid #64b5f6;
            }
            QPushButton:pressed {
                background-color: #90caf9;
            }
        """)
        self.outfile_btn = outfile_btn
        param_layout.addWidget(outfile_btn)

        # DPI选择（仅影响扫描件处理）
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("扫描件处理精度："))
        self.dpi_combo = QComboBox()
        self.dpi_combo.addItem("标准 (200 DPI)", 200)
        self.dpi_combo.addItem("快速 (100 DPI)", 100)
        self.dpi_combo.setCurrentIndex(0)  # 默认200dpi
        self.dpi_combo.setToolTip("选择扫描件图像提取的DPI精度\n200DPI：更高质量，处理较慢\n100DPI：更快处理，较低质量")
        dpi_layout.addWidget(self.dpi_combo)
        dpi_layout.addStretch()
        param_layout.addLayout(dpi_layout)
        


        parent_layout.addWidget(param_group)

    def _build_button_group(self, parent_layout):
        """构建操作按钮分组"""
        # 主转换按钮 - 单独一行，更加突出
        convert_layout = QHBoxLayout()
        
        self.convert_btn = QPushButton("🚀 开始转换")
        self.convert_btn.clicked.connect(self.start_conversion)
        self.convert_btn.setEnabled(True)  # 默认启用
        self.convert_btn.setMinimumHeight(60)  # 增加高度
        self.convert_btn.setStyleSheet("""
            QPushButton {
                padding: 16px 32px;
                background-color: #4caf50;
                color: white;
                border: 3px solid #45a049;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                min-width: 200px;
            }
            QPushButton:hover:!disabled {
                background-color: #45a049;
                border: 3px solid #3d8b40;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
                border: 3px solid #bdbdbd;
            }
        """)
        convert_layout.addStretch()
        convert_layout.addWidget(self.convert_btn)
        convert_layout.addStretch()
        parent_layout.addLayout(convert_layout)

    def _build_log_group(self, parent_layout):
        """构建日志分组（移到左侧）- 占据剩余空间"""
        log_group = QGroupBox("转换日志")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #f9f9f9;")
        log_layout.addWidget(self.log_text)

        parent_layout.addWidget(log_group, 1)  # 设置 stretch factor 为 1，占据剩余空间

    def _build_right_panel(self, main_layout):
        """构建右侧预览面板（占据整个右侧）"""
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)  # 去除边距
        main_layout.addLayout(right_layout)

        # 预览分组 - 占据整个右侧
        preview_group = QGroupBox("文件预览")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(5, 5, 5, 5)  # 减小内边距

        # 顶部状态标签（显示选中数量）- 独立于图片
        self.selection_status_label = QLabel("")
        self.selection_status_label.setAlignment(Qt.AlignCenter)
        self.selection_status_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        self.selection_status_label.setMaximumHeight(30)
        preview_layout.addWidget(self.selection_status_label)

        # 预览标签（支持鼠标滚轮和右键选择）- 占据整个右侧区域
        self.preview_label = QLabel("暂无预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border:1px solid #ccc; background-color: #fafafa;")
        self.preview_label.setMinimumHeight(600)
        self.preview_label.setMouseTracking(True)
        self.preview_label.wheelEvent = self.preview_wheel_event
        self.preview_label.mousePressEvent = self.preview_mouse_press_event
        self.preview_label.mouseReleaseEvent = self.preview_mouse_release_event
        self.preview_label.mouseMoveEvent = self.preview_mouse_move_event
        self.preview_label.keyPressEvent = self.preview_key_press_event
        self.preview_label.keyReleaseEvent = self.preview_key_release_event
        preview_layout.addWidget(self.preview_label)

        # 底部状态标签（显示当前页选中状态）- 独立于图片
        self.page_status_label = QLabel("")
        self.page_status_label.setAlignment(Qt.AlignCenter)
        self.page_status_label.setMaximumHeight(30)
        preview_layout.addWidget(self.page_status_label)

        # 页码显示和导航
        page_nav_layout = QHBoxLayout()
        self.page_label = QLabel("页码：0/0")
        self.page_label.setAlignment(Qt.AlignCenter)
        page_nav_layout.addWidget(self.page_label)
        
        # 快捷切换按钮（显示当前预览页的处理方法，点击切换当前页）
        self.quick_switch_btn = QPushButton("🖼️ 图像识别")
        self.quick_switch_btn.setToolTip("当前预览页使用图像识别处理\n点击切换当前页的处理方法")
        self.quick_switch_btn.setEnabled(False)
        self.quick_switch_btn.clicked.connect(self.quick_switch_method)
        self.quick_switch_btn.setStyleSheet("""
            QPushButton {
                padding: 4px 10px;
                background-color: #4caf50;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover:!disabled {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }
        """)
        page_nav_layout.addWidget(self.quick_switch_btn)
        
        # 批量设置按钮（设置所有选中页面的处理方法）
        self.batch_method_btn = QPushButton("批量设置")
        self.batch_method_btn.setToolTip("批量设置所有选中电子档页面的处理方法")
        self.batch_method_btn.setEnabled(False)
        self.batch_method_btn.clicked.connect(self.show_batch_method_menu)
        self.batch_method_btn.setStyleSheet("""
            QPushButton {
                padding: 4px 10px;
                background-color: #ff9800;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover:!disabled {
                background-color: #f57c00;
            }
            QPushButton:pressed {
                background-color: #e65100;
            }
            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }
        """)
        page_nav_layout.addWidget(self.batch_method_btn)
        
        # 第一行：上一页/下一页 + 跳转（放一起）
        nav_row1 = QHBoxLayout()
        
        self.prev_btn = QPushButton("◀ 上一页")
        self.prev_btn.clicked.connect(self.prev_page)
        self.prev_btn.setEnabled(False)
        self.prev_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                background-color: #fff3e0;
                border: 1px solid #ffb74d;
                border-radius: 3px;
                min-width: 70px;
            }
            QPushButton:hover:!disabled {
                background-color: #ffe0b2;
                border: 1px solid #ffa726;
            }
            QPushButton:pressed {
                background-color: #ffcc80;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #bdbdbd;
            }
        """)
        nav_row1.addWidget(self.prev_btn)
        
        # 页面跳转功能（放在上一页/下一页之间）
        jump_layout = QHBoxLayout()
        jump_layout.addWidget(QLabel("跳转:"))
        self.page_input = QLineEdit()
        self.page_input.setFixedWidth(45)
        self.page_input.setAlignment(Qt.AlignCenter)
        self.page_input.returnPressed.connect(self.jump_to_page)
        jump_layout.addWidget(self.page_input)
        
        jump_btn = QPushButton("Go")
        jump_btn.clicked.connect(self.jump_to_page)
        jump_btn.setFixedWidth(40)
        jump_btn.setStyleSheet("""
            QPushButton {
                padding: 4px;
                background-color: #e3f2fd;
                border: 1px solid #90caf9;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #bbdefb;
                border: 1px solid #64b5f6;
            }
            QPushButton:pressed {
                background-color: #90caf9;
            }
        """)
        jump_layout.addWidget(jump_btn)
        nav_row1.addLayout(jump_layout)
        
        self.next_btn = QPushButton("下一页 ▶")
        self.next_btn.clicked.connect(self.next_page)
        self.next_btn.setEnabled(False)
        self.next_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                background-color: #fff3e0;
                border: 1px solid #ffb74d;
                border-radius: 3px;
                min-width: 70px;
            }
            QPushButton:hover:!disabled {
                background-color: #ffe0b2;
                border: 1px solid #ffa726;
            }
            QPushButton:pressed {
                background-color: #ffcc80;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #bdbdbd;
            }
        """)
        nav_row1.addWidget(self.next_btn)
        nav_row1.addStretch()
        
        page_nav_layout.addLayout(nav_row1)
        
        # 第二行：范围选择（单独放一起）
        nav_row2 = QHBoxLayout()
        
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("范围:"))
        self.range_start_input = QLineEdit()
        self.range_start_input.setFixedWidth(45)
        self.range_start_input.setAlignment(Qt.AlignCenter)
        self.range_start_input.setPlaceholderText("起")
        self.range_start_input.returnPressed.connect(self.select_range)
        range_layout.addWidget(self.range_start_input)
        
        range_layout.addWidget(QLabel("-"))
        
        self.range_end_input = QLineEdit()
        self.range_end_input.setFixedWidth(45)
        self.range_end_input.setAlignment(Qt.AlignCenter)
        self.range_end_input.setPlaceholderText("止")
        self.range_end_input.returnPressed.connect(self.select_range)
        range_layout.addWidget(self.range_end_input)
        
        range_select_btn = QPushButton("选择")
        range_select_btn.clicked.connect(self.select_range)
        range_select_btn.setFixedWidth(50)
        range_select_btn.setStyleSheet("""
            QPushButton {
                padding: 4px 8px;
                background-color: #e8f5e9;
                border: 1px solid #a5d6a7;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #c8e6c9;
                border: 1px solid #81c784;
            }
            QPushButton:pressed {
                background-color: #a5d6a7;
            }
        """)
        range_layout.addWidget(range_select_btn)
        
        nav_row2.addLayout(range_layout)
        nav_row2.addStretch()
        
        page_nav_layout.addLayout(nav_row2)
        preview_layout.addLayout(page_nav_layout)

        right_layout.addWidget(preview_group)
        # 日志已移到左侧，右侧仅保留预览

    # ---------------------- 窗口事件处理 ----------------------
    def resizeEvent(self, event):
        """窗口大小改变事件 - 确保预览正确重绘"""
        super().resizeEvent(event)
        # 窗口大小改变时，如果已有图像，更新预览以适应新尺寸
        # 注意：初始化时可能还没有 image_list，需要检查
        if hasattr(self, 'image_list') and self.image_list and self.current_preview_idx < len(self.image_list):
            # 使用延迟调用，避免频繁重绘
            from PySide6.QtCore import QTimer
            QTimer.singleShot(100, self.update_preview)
    
    # ---------------------- 拖拽事件处理 ----------------------
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖拽进入窗口事件"""
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ALL_SUPPORTED_FORMATS:
                event.acceptProposedAction()
                self.status_bar.showMessage(f"即将加载：{os.path.basename(file_path)}")
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        """拖拽释放事件"""
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.load_file(file_path)

    # ---------------------- 文件选择与加载 ----------------------
    def select_file(self):
        """打开文件选择对话框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择支持的文件", "",
            f"所有支持文件 ({' '.join(['*' + fmt for fmt in ALL_SUPPORTED_FORMATS])});;"
            f"PDF文件 (*.pdf);;图片文件 ({' '.join(['*' + fmt for fmt in SUPPORTED_IMG_FORMATS])})"
        )
        if file_path:
            self.load_file(file_path)

    def load_file(self, file_path):
        """加载选择的 PDF/图片文件"""
        # 如果之前有加载线程在运行，先停止它
        if hasattr(self, 'loader_thread') and self.loader_thread is not None:
            if self.loader_thread.isRunning():
                self.loader_thread.stop()
                self.loader_thread.wait(1000)  # 等待 1 秒
        
        # 关闭旧的 PDF 加载器
        if self.pdf_loader:
            self.pdf_loader.shutdown()
        
        self.clear_all()
        self.selected_file = file_path
        self.file_edit.setText(file_path)

        # 设置默认输出路径（与源文件同目录）
        dir_path = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        self.output_path = os.path.normpath(os.path.join(dir_path, f"{base_name}_output.xlsx"))
        self.outfile_edit.setText(self.output_path)

        # 清空 data 文件夹下的旧图片
        import glob
        data_dir = os.path.join(os.path.dirname(sys.executable), "data")
        existing_files = glob.glob(os.path.join(data_dir, "pdf_page_*.png"))
        for f in existing_files:
            try:
                os.remove(f)
            except Exception:
                pass

        # 初始化图像列表
        self.image_list = []
        self.temp_files = []
        self.total_pages = 0  # 总页数

        # 如果是 PDF 文件，创建按需加载器
        if file_path.lower().endswith('.pdf'):
            self.status_bar.showMessage("正在初始化 PDF 加载器...")
            QApplication.processEvents()
            
            # 直接调用异步初始化方法
            self._init_pdf_loader_async(file_path)
        else:
            # 图片文件，使用原有逻辑
            self.status_bar.showMessage("加载中...")
            self.loader_thread = PDFLoaderThread(file_path, load_all=False)
            self.loader_thread.progress.connect(self.update_load_progress)
            self.loader_thread.page_loaded.connect(self.on_page_loaded)
            self.loader_thread.total_pages_known.connect(self.on_total_pages_known)
            self.loader_thread.finished.connect(self.on_load_finished)
            self.loader_thread.start()
    
    def update_load_progress(self, current, total):
        """更新加载进度"""
        self.status_bar.showMessage(f"加载中... {current}/{total} 页")
    
    def on_demand_page_loaded(self, page_idx: int, image_path: str, success: bool):
        """按需加载页面完成回调"""
        if success and image_path:
            # 将新加载的页面插入到 image_list 中
            if page_idx < len(self.image_list):
                self.image_list[page_idx] = image_path
            else:
                while len(self.image_list) <= page_idx:
                    self.image_list.append(None)
                self.image_list[page_idx] = image_path
            
            # 标记为已加载
            self.loaded_pages.add(page_idx)
            
            # 如果当前预览的页面刚好是这个，刷新预览
            if self.current_preview_idx == page_idx:
                print(f"DEBUG callback: 刷新预览 page_idx={page_idx}")
                self.update_preview()
            
            # 检查是否有等待的旋转任务
            self._check_pending_rotation()
            
            # 按钮默认启用，不需要在这里启用
    
    def _init_pdf_loader_async(self, file_path: str):
        """初始化 PDF 加载器 V2 - 完全按需"""
        try:
            self.status_bar.showMessage("正在打开 PDF 文件...")
            QApplication.processEvents()
            
            # 清理旧的加载器
            if hasattr(self, 'pdf_loader_v2') and self.pdf_loader_v2:
                self.pdf_loader_v2.close()
                self.pdf_loader_v2 = None
            
            # 清理缓存的图像文件
            if hasattr(self, 'temp_files') and self.temp_files:
                for temp_file in self.temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except:
                        pass
                self.temp_files = []
            
            # 清理 data 目录下的旧缓存文件（防止旧PDF的图像干扰）
            try:
                import glob
                data_dir = os.path.join(PROJECT_ROOT, "data")
                if os.path.exists(data_dir):
                    # 删除所有页面缓存文件
                    for pattern in ["page_*_preview.png", "page_*_dpi*.png", "pdf_page_*.png"]:
                        for old_file in glob.glob(os.path.join(data_dir, pattern)):
                            try:
                                os.unlink(old_file)
                            except:
                                pass
            except Exception:
                pass
            
            # 创建 V2 加载器（完全按需，不立即分析任何页面）
            self.pdf_loader_v2 = PDFLoaderV2(file_path, max_workers=3)
            
            self.total_pdf_pages = self.pdf_loader_v2.total_pages
            self.pdf_path = file_path
            
            # 初始化 image_list
            self.image_list = [None] * self.total_pdf_pages
            self.current_preview_idx = 0
            
            # 清空已加载页面集合
            self.loaded_pages.clear()
            
            # 清空处理管理器
            self.process_manager.clear()
            
            # 清空上次的处理结果
            self.last_result = {}
            
            # 自动全选所有页面
            self.selected_pages = set(range(self.total_pdf_pages))
            self._update_range_inputs()
            
            # 显示初始信息
            self.log_text.append(f"📄 已打开 PDF，共 {self.total_pdf_pages} 页")
            self.log_text.append(f"✅ 页面类型和预览将按需加载")
            self.status_bar.showMessage(f"✅ 已加载 PDF，共{self.total_pdf_pages}页（按需分析）")
            
            # 立即显示第一页（按需生成）
            self.update_preview()
                
        except Exception as e:
            import traceback
            error_msg = f"无法加载 PDF 文件：{str(e)}\n\n详细信息：\n{traceback.format_exc()}\n\n请确保文件没有被其他程序占用"
            QMessageBox.critical(self, "错误", error_msg)
            self.clear_all()
    
    def on_page_loaded_v2(self, page_idx: int, image_path: str, success: bool):
        """V2 加载器页面加载完成回调"""
        if success:
            if image_path:
                # 有图像路径，直接更新
                if page_idx < len(self.image_list):
                    self.image_list[page_idx] = image_path
                else:
                    # 如果 image_list 长度不够，扩展它
                    while len(self.image_list) <= page_idx:
                        self.image_list.append(None)
                    self.image_list[page_idx] = image_path
                
                # 标记为已加载
                self.loaded_pages.add(page_idx)
                
                # 如果当前预览的页面是这个，刷新预览
                if self.current_preview_idx == page_idx:
                    self.update_preview()
            else:
                # image_path 为 None，表示页面已准备好但需按需加载
                # 确保 image_list 长度足够
                if page_idx >= len(self.image_list):
                    while len(self.image_list) <= page_idx:
                        self.image_list.append(None)
                
                # 如果当前预览的是这一页，触发按需加载
                if self.current_preview_idx == page_idx:
                    self.update_preview()
    
    def on_total_pages_known(self, total_pages):
        """总页数已知处理"""
        self.total_pages = total_pages
        # 更新页码显示
        if self.image_list:
            current_page = self.current_preview_idx + 1
            self.page_label.setText(f"页码：{current_page}/{total_pages}")
    
    def quick_switch_method(self):
        """快捷切换：切换当前预览页的处理方法"""
        current_page = self.current_preview_idx
        
        if not hasattr(self, 'pdf_loader_v2') or not self.pdf_loader_v2:
            return
        
        # 检查当前预览页是否是电子档
        page_type = self.pdf_loader_v2.get_page_type(current_page)
        if page_type != 'electronic':
            QMessageBox.information(self, "提示", f"第{current_page+1}页是扫描件，无需切换处理方法")
            return
        
        # 确保页面已注册到 process_manager
        current_config = self.process_manager.get_config(current_page)
        if not current_config:
            # 页面未注册，先注册
            current_config = self.process_manager.register_page(current_page, PageType.ELECTRONIC)
        
        # 根据当前方法切换到另一种
        if current_config.is_image_processing():
            # 当前是图像识别，切换为矢量提取
            self.process_manager.set_method(current_page, ProcessMethod.VECTOR)
            target_name = "矢量提取"
            target_icon = "📄"
            btn_text = "📄 矢量提取"
            btn_color = "#2196f3"  # 蓝色
        else:
            # 当前是矢量提取，切换为图像识别
            self.process_manager.set_method(current_page, ProcessMethod.IMAGE)
            target_name = "图像识别"
            target_icon = "🖼️"
            btn_text = "🖼️ 图像识别"
            btn_color = "#4caf50"  # 绿色
        
        # 显示结果
        self.log_text.append(f"{target_icon} 已将第{current_page+1}页切换为「{target_name}」")
        self.status_bar.showMessage(f"第{current_page+1}页已切换为{target_name}")
        
        # 更新按钮显示
        self.quick_switch_btn.setText(btn_text)
        self.quick_switch_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 4px 10px;
                background-color: {btn_color};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover:!disabled {{
                background-color: {btn_color};
                opacity: 0.9;
            }}
            QPushButton:pressed {{
                background-color: {btn_color};
                opacity: 0.8;
            }}
            QPushButton:disabled {{
                background-color: #e0e0e0;
                color: #9e9e9e;
            }}
        """)
        self.quick_switch_btn.setToolTip(f"当前预览页（第{current_page+1}页）使用{target_name}处理\n点击切换处理方法")
    
    def show_batch_method_menu(self):
        """显示批量设置处理方法的菜单"""
        if not self.selected_pages:
            return
        
        if not hasattr(self, 'pdf_loader_v2') or not self.pdf_loader_v2:
            return
        
        # 筛选出电子档页面
        text_pages = []
        for page_idx in self.selected_pages:
            page_type = self.pdf_loader_v2.get_page_type(page_idx)
            if page_type == 'electronic':
                text_pages.append(page_idx)
        
        if not text_pages:
            QMessageBox.information(self, "提示", "选中的页面中没有电子档页面")
            return
        
        # 创建菜单
        menu = QMenu(self)
        
        # 图像识别选项
        action_image = menu.addAction("🖼️ 设置为图像识别")
        action_image.triggered.connect(lambda: self.batch_set_method(text_pages, ProcessMethod.IMAGE, "图像识别"))
        
        # 矢量提取选项
        action_vector = menu.addAction("📄 设置为矢量提取")
        action_vector.triggered.connect(lambda: self.batch_set_method(text_pages, ProcessMethod.VECTOR, "矢量提取"))
        
        # 显示菜单在按钮下方
        menu.exec(self.batch_method_btn.mapToGlobal(self.batch_method_btn.rect().bottomLeft()))
    
    def batch_set_method(self, page_indices, method, method_name):
        """批量设置页面的处理方法"""
        for page_idx in page_indices:
            # 确保页面已注册
            config = self.process_manager.get_config(page_idx)
            if not config:
                self.process_manager.register_page(page_idx, PageType.ELECTRONIC)
            self.process_manager.set_method(page_idx, method)
        
        page_numbers = [p + 1 for p in page_indices]
        icon = "🖼️" if method == ProcessMethod.IMAGE else "📄"
        self.log_text.append(f"{icon} 已将第{', '.join(map(str, page_numbers))}页批量设置为「{method_name}」")
        self.status_bar.showMessage(f"已将 {len(page_indices)} 页设置为{method_name}")
        
        # 更新当前预览页的按钮显示
        self.update_quick_switch_button()
    
    def update_quick_switch_button(self):
        """更新快捷切换按钮的状态和显示 - 根据当前预览页的处理方法"""
        # 检查是否有选中的电子档页面
        has_text_pages = False
        
        # 如果没有 pdf_loader_v2，说明是图片文件，直接按扫描件处理
        if not hasattr(self, 'pdf_loader_v2') or not self.pdf_loader_v2:
            self.quick_switch_btn.setEnabled(False)
            self.quick_switch_btn.setText("—")
            self.quick_switch_btn.setToolTip("当前是图片文件，按扫描件处理")
            self.batch_method_btn.setEnabled(False)
            return
        
        if self.selected_pages:
            for page_idx in self.selected_pages:
                page_type = self.pdf_loader_v2.get_page_type(page_idx)
                if page_type == 'electronic':
                    has_text_pages = True
                    break
        
        if not has_text_pages:
            self.quick_switch_btn.setEnabled(False)
            self.quick_switch_btn.setText("🖼️ 图像识别")
            self.quick_switch_btn.setToolTip("选中的页面中没有电子档页面")
            self.batch_method_btn.setEnabled(False)
            return
        
        # 有电子档页面，启用按钮
        self.quick_switch_btn.setEnabled(True)
        self.batch_method_btn.setEnabled(True)
        
        # 根据当前预览页的处理方法显示按钮
        current_page = self.current_preview_idx
        if hasattr(self, 'pdf_loader_v2') and self.pdf_loader_v2:
            current_page_type = self.pdf_loader_v2.get_page_type(current_page)
            
            if current_page_type == 'electronic':
                # 当前预览页是电子档，显示其处理方法
                config = self.process_manager.get_config(current_page)
                # 如果页面未注册，创建一个临时配置用于显示（默认矢量提取）
                if not config:
                    config = PageProcessConfig(page_idx=current_page, page_type=PageType.ELECTRONIC, method=ProcessMethod.AUTO)
                
                if config.is_image_processing():
                    # 当前是图像识别
                    self.quick_switch_btn.setText("🖼️ 图像识别")
                    self.quick_switch_btn.setToolTip(f"当前预览页（第{current_page+1}页）使用图像识别处理\n点击切换为矢量提取")
                    self.quick_switch_btn.setStyleSheet("""
                        QPushButton {
                            padding: 4px 10px;
                            background-color: #4caf50;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            font-size: 11px;
                            font-weight: bold;
                        }
                        QPushButton:hover:!disabled {
                            background-color: #45a049;
                        }
                        QPushButton:pressed {
                            background-color: #3d8b40;
                        }
                        QPushButton:disabled {
                            background-color: #e0e0e0;
                            color: #9e9e9e;
                        }
                    """)
                else:
                    # 当前是矢量提取
                    self.quick_switch_btn.setText("📄 矢量提取")
                    self.quick_switch_btn.setToolTip(f"当前预览页（第{current_page+1}页）使用矢量提取处理\n点击切换为图像识别")
                    self.quick_switch_btn.setStyleSheet("""
                        QPushButton {
                            padding: 4px 10px;
                            background-color: #2196f3;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            font-size: 11px;
                            font-weight: bold;
                        }
                        QPushButton:hover:!disabled {
                            background-color: #1e88e5;
                        }
                        QPushButton:pressed {
                            background-color: #1976d2;
                        }
                        QPushButton:disabled {
                            background-color: #e0e0e0;
                            color: #9e9e9e;
                        }
                    """)
            else:
                # 当前预览页是扫描件，禁用按钮
                self.quick_switch_btn.setEnabled(False)
                self.quick_switch_btn.setText("—")
                self.quick_switch_btn.setToolTip(f"当前预览页（第{current_page+1}页）是扫描件，无需切换")
                self.quick_switch_btn.setStyleSheet("""
                    QPushButton {
                        padding: 4px 10px;
                        background-color: #4caf50;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        font-size: 11px;
                        font-weight: bold;
                    }
                    QPushButton:hover:!disabled {
                        background-color: #45a049;
                    }
                    QPushButton:pressed {
                        background-color: #3d8b40;
                    }
                    QPushButton:disabled {
                        background-color: #e0e0e0;
                        color: #9e9e9e;
                    }
                """)
        else:
            # 没有 pdf_loader_v2，说明是图片文件，显示为扫描件
            self.quick_switch_btn.setEnabled(False)
            self.quick_switch_btn.setText("—")
            self.quick_switch_btn.setToolTip(f"当前是图片文件，按扫描件处理")
            self.quick_switch_btn.setStyleSheet("""
                QPushButton {
                    padding: 4px 10px;
                    background-color: #4caf50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover:!disabled {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
                QPushButton:disabled {
                    background-color: #e0e0e0;
                    color: #9e9e9e;
                }
            """)
    
    def update_page_selection_ui(self):
        """更新页面选择相关的UI状态"""
        # 更新快捷切换按钮
        self.update_quick_switch_button()
    
    def preview_wheel_event(self, event):
        """预览区域鼠标滚轮事件"""
        if not self.image_list:
            return
        
        # 滚轮向上滚动（delta > 0）- 上一页
        # 滚轮向下滚动（delta < 0）- 下一页
        if event.angleDelta().y() > 0:
            self.prev_page()
        else:
            self.next_page()
    
    def preview_mouse_press_event(self, event):
        """预览区域鼠标按下事件"""
        if not self.image_list:
            return
        
        # 获取预览区域的尺寸
        width = self.preview_label.width()
        height = self.preview_label.height()
        x = event.position().x()
        y = event.position().y()
        
        # 左键点击
        if event.button() == Qt.LeftButton:
            # 左侧 1/4 区域：上一页
            if x < width / 4:
                self.prev_page()
                # 如果 Ctrl 按下，连续选择
                if self.ctrl_pressed and self.selection_start_page >= 0:
                    self._update_continuous_selection()
                return
            
            # 右侧 1/4 区域：下一页
            if x > width * 3 / 4:
                self.next_page()
                # 如果 Ctrl 按下，连续选择
                if self.ctrl_pressed and self.selection_start_page >= 0:
                    self._update_continuous_selection()
                return
            
            # 中间区域
            # 上半部分：选中当前页面
            if y < height / 2:
                if self.current_preview_idx not in self.selected_pages:
                    self.selected_pages.add(self.current_preview_idx)
                    self.status_bar.showMessage(f"已选中第{self.current_preview_idx + 1}页")
                else:
                    self.status_bar.showMessage(f"第{self.current_preview_idx + 1}页已在选中状态")
                self.update_preview()
                return
            
            # 下半部分：取消选中当前页面
            if y >= height / 2:
                if self.current_preview_idx in self.selected_pages:
                    self.selected_pages.discard(self.current_preview_idx)
                    self.status_bar.showMessage(f"已取消选中第{self.current_preview_idx + 1}页")
                else:
                    self.status_bar.showMessage(f"第{self.current_preview_idx + 1}页未选中")
                self.update_preview()
                return
        
        # 右键按下：开始选择模式
        elif event.button() == Qt.RightButton:
            self.selection_mode = True
            self.selection_start_page = self.current_preview_idx
            self.selected_pages = {self.current_preview_idx}
            self.update_preview()
            self.status_bar.showMessage(f"选择模式：已选中第{self.current_preview_idx + 1}页")
    
    def preview_mouse_release_event(self, event):
        """预览区域鼠标释放事件"""
        # 右键释放：结束选择模式
        if event.button() == Qt.RightButton and self.selection_mode:
            self.selection_mode = False
            self.hover_region = None
            self.update_preview()
            
            selected_count = len(self.selected_pages)
            if selected_count > 1:
                self.status_bar.showMessage(f"选择完成：已选中{selected_count}页")
            else:
                self.status_bar.showMessage(f"选择完成：已选中 1 页")
    
    def preview_mouse_move_event(self, event):
        """预览区域鼠标移动事件"""
        if not self.image_list:
            return
        
        # 计算鼠标在预览区域的位置
        width = self.preview_label.width()
        height = self.preview_label.height()
        x = event.position().x()
        y = event.position().y()
        
        # 确定当前悬停区域
        if x < width / 4:
            new_region = "left"  # 左侧：上一页区域
        elif x > width * 3 / 4:
            new_region = "right"  # 右侧：下一页区域
        elif y < height / 2:
            new_region = "top"  # 上半部分：选中区域
        else:
            new_region = "bottom"  # 下半部分：取消选中区域
        
        # 如果悬停区域变化，更新视觉反馈
        if new_region != self.hover_region:
            self.hover_region = new_region
            self.update_preview()  # 立即重绘以显示箭头
        
        # 选择模式下的翻页逻辑
        if self.selection_mode:
            if y < height / 3:
                # 上部区域：上一页
                if self.current_preview_idx > 0:
                    self.current_preview_idx -= 1
            elif y > height * 2 / 3:
                # 下部区域：下一页
                if self.current_preview_idx < len(self.image_list) - 1:
                    self.current_preview_idx += 1
            
            # 更新选择范围
            if self.selection_start_page != -1:
                self._update_continuous_selection()
    
    def _update_continuous_selection(self):
        """更新连续选择"""
        start = min(self.selection_start_page, self.current_preview_idx)
        end = max(self.selection_start_page, self.current_preview_idx)
        self.selected_pages = set(range(start, end + 1))
        self.status_bar.showMessage(f"已选中 {len(self.selected_pages)} 页")
        self.update_preview()
    
    def _update_hover_visual(self):
        """更新悬停视觉反馈"""
        # 现在视觉反馈直接在 update_preview 中绘制
        # 此方法保留用于未来扩展
        pass
    
    def preview_key_press_event(self, event):
        """预览区域键盘按下事件"""
        # Ctrl 键按下
        if event.key() == Qt.Key_Control:
            self.ctrl_pressed = True
            self.status_bar.showMessage("Ctrl 已按下：进入连续选择模式")
            self.update_preview()  # 重绘以显示状态
        
        # Ctrl + A：全选
        elif event.key() == Qt.Key_A and self.ctrl_pressed:
            self.selected_pages = set(range(len(self.image_list)))
            self.status_bar.showMessage(f"已全选 {len(self.image_list)} 页")
            self.update_preview()
        
        # Ctrl + D：取消全选
        elif event.key() == Qt.Key_D and self.ctrl_pressed:
            self.selected_pages.clear()
            self.status_bar.showMessage("已取消所有选择")
            self.update_preview()
    
    def preview_key_release_event(self, event):
        """预览区域键盘释放事件"""
        # Ctrl 键释放
        if event.key() == Qt.Key_Control:
            self.ctrl_pressed = False
            if self.selection_mode:
                self.status_bar.showMessage(f"选择模式：已选中 {len(self.selected_pages)} 页")
            else:
                self.status_bar.showMessage(f"已选中 {len(self.selected_pages)} 页")
    
    def eventFilter(self, obj, event):
        """全局事件过滤器，支持全局快捷键"""
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent
        
        if event.type() == QEvent.Type.KeyPress:
            # 检查是否是 Ctrl 键
            if event.key() == Qt.Key_Control:
                self.ctrl_pressed = True
                self.status_bar.showMessage("Ctrl 已按下：进入连续选择模式")
                self.update_preview()
                return True
            
            # Ctrl + A：全选
            elif event.key() == Qt.Key_A and self.ctrl_pressed:
                if self.image_list:
                    self.selected_pages = set(range(len(self.image_list)))
                    self.status_bar.showMessage(f"已全选 {len(self.image_list)} 页")
                    self.update_preview()
                return True
            
            # Ctrl + D：取消全选
            elif event.key() == Qt.Key_D and self.ctrl_pressed:
                self.selected_pages.clear()
                self.status_bar.showMessage("已取消所有选择")
                self.update_preview()
                return True
        
        elif event.type() == QEvent.Type.KeyRelease:
            if event.key() == Qt.Key_Control:
                self.ctrl_pressed = False
                if self.selection_mode:
                    self.status_bar.showMessage(f"选择模式：已选中 {len(self.selected_pages)} 页")
                else:
                    self.status_bar.showMessage(f"已选中 {len(self.selected_pages)} 页")
                return True
        
        # 其他事件交给基类处理
        return super().eventFilter(obj, event)
    
    def on_page_loaded(self, image_path):
        """单页加载完成处理"""
        # 添加到图像列表
        if image_path not in self.image_list:
            self.image_list.append(image_path)
            
        # 如果是第一页，更新预览
        if len(self.image_list) == 1:
            self.current_preview_idx = 0
            self.update_preview()
            
            # 按钮默认启用，不需要在这里启用
        else:
            # 更新预览，保持当前页码
            self.update_preview()
    
    def on_load_finished(self, image_list, success, message, temp_files):
        """加载完成处理"""
        if success:
            self.image_list = image_list
            self.temp_files = temp_files
            
            # 保存已加载的页面索引
            for i in range(len(image_list)):
                self.loaded_pages.add(i)
            
            # 保存 PDF 文档信息以便后续按需加载
            if self.selected_file.lower().endswith(SUPPORTED_PDF_FORMATS):
                # 只保存文件路径，不保存文档对象（避免内存问题）
                self.pdf_path = self.selected_file
                self.total_pdf_pages = self.total_pages
            else:
                self.pdf_path = None
                self.total_pdf_pages = len(self.image_list)
                
                # 图片文件：将所有页面注册为扫描件（强制图像处理）
                for page_idx in range(len(self.image_list)):
                    self.process_manager.register_page(page_idx, PageType.SCANNED)
            
            # 更新页码显示
            total_pages = len(self.image_list)
            
            # 只有在没有手动选择过的情况下才全选（通过检查日志判断）
            # 如果用户已经手动选择了页面，保持用户的选择
            has_manual_selection = len(self.selected_pages) > 0 and self.selected_pages != set(range(total_pages))
            
            if not has_manual_selection:
                # 默认全选所有已加载的页面
                self.selected_pages = set(range(len(self.image_list)))
                self.status_bar.showMessage(f"已加载：{os.path.basename(self.selected_file)}（{total_pages}/{self.total_pdf_pages} 页，已全选）")
            else:
                # 保持用户的选择，只更新状态栏
                self.status_bar.showMessage(f"已加载：{os.path.basename(self.selected_file)}（{total_pages}/{self.total_pdf_pages} 页，已选中{len(self.selected_pages)}页）")
            
            # 更新预览
            if self.image_list:
                self.update_preview()
            
            # 日志与状态栏更新
            self.log_text.append(f"✅ {message}")
            
            # 检查是否有等待的转换任务
            self._check_pending_conversion()
        else:
            QMessageBox.critical(self, "加载失败", message)
            self.status_bar.showMessage("加载失败")
    
    def load_page_on_demand(self, page_idx):
        """
        按需加载单页 PDF
        
        Args:
            page_idx: 页面索引（0-based）
        """
        if page_idx < 0 or page_idx >= self.total_pdf_pages:
            return
        
        # 如果页面已经加载过，直接返回
        if page_idx in self.loaded_pages:
            return
        
        # 如果已经有后台加载线程在运行，跳过
        if self.background_loader and self.background_loader.isRunning():
            return
        
        # 创建并启动后台加载线程
        data_dir = os.path.join(os.path.dirname(sys.executable), "data")
        self.background_loader = SinglePageLoaderThread(self.pdf_path, page_idx, data_dir)
        self.background_loader.page_loaded.connect(self.on_background_page_loaded)
        self.background_loader.start()
    
    def on_background_page_loaded(self, page_idx, image_path):
        """后台加载页面完成处理"""
        # 将新加载的页面插入到 image_list 中
        if page_idx < len(self.image_list):
            # 已经存在，替换
            self.image_list[page_idx] = image_path
        else:
            # 追加到列表
            while len(self.image_list) <= page_idx:
                self.image_list.append(None)
            self.image_list[page_idx] = image_path
        
        # 标记为已加载
        self.loaded_pages.add(page_idx)
        
        # 如果当前预览的页面刚好是这个，刷新预览
        if self.current_preview_idx == page_idx:
            self.update_preview()
        
        # 检查是否可以开始转换
        self._check_pending_conversion()
    
    def _check_pending_conversion(self):
        """检查是否有等待的转换任务（当加载完成时自动触发）"""
        if not self.selected_pages:
            return
        
        # 检查是否还有未加载的页面
        unloaded_pages = [p for p in self.selected_pages if p not in self.loaded_pages]
        
        # 如果所有选中的页面都已加载完成，且转换按钮是禁用状态（说明之前在等待）
        if not unloaded_pages and not self.convert_btn.isEnabled():
            self.log_text.append("")
            self.log_text.append("✅ 所有页面已加载完成，自动开始处理...")
            # 恢复用户原始选择的页面范围（如果之前选择了 273 页，现在应该处理 273 页）
            # 不需要修改 selected_pages，因为它一直保持着用户的选择
            self.start_conversion(allow_partial=False)

    # ---------------------- 输出文件路径选择 ----------------------
    def select_outfile(self):
        """选择Excel文件保存路径"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存Excel文件", self.output_path,
            "Excel文件 (*.xlsx);;所有文件 (*.*)"
        )
        if file_path:
            # 自动补充 .xlsx 后缀
            if not file_path.lower().endswith(".xlsx"):
                file_path += ".xlsx"
            self.output_path = file_path
            self.outfile_edit.setText(self.output_path)

    # ---------------------- 预览功能 ----------------------
    def update_preview(self):
        """更新预览窗口内容（按需生成预览）"""
        if not self.image_list:
            return
        
        # 确保当前预览索引在有效范围内
        if self.current_preview_idx >= len(self.image_list):
            self.current_preview_idx = len(self.image_list) - 1

        # 获取当前预览图像的路径
        img_path = self.image_list[self.current_preview_idx]
        
        # 如果预览尚未生成（None 或空字符串），按需生成
        if not img_path and hasattr(self, 'pdf_loader_v2') and self.pdf_loader_v2:
            self.status_bar.showMessage(f"正在生成页面 {self.current_preview_idx + 1} 预览...")
            QApplication.processEvents()  # 更新UI
            
            img_path = self.pdf_loader_v2.get_preview(self.current_preview_idx)
            if img_path:
                self.image_list[self.current_preview_idx] = img_path
        
        # 检查图像是否正在加载中
        if not img_path:
            self.preview_label.setText(f"页面 {self.current_preview_idx + 1} 正在加载...")
            return
        
        # 读取图像（cv2 支持多种格式）
        img = cv2.imread(img_path)
        if img is None:
            self.preview_label.setText("无法预览该图像")
            return
        
        # 调试信息：显示图像尺寸
        h, w = img.shape[:2]
        self.status_bar.showMessage(f"预览: 页面 {self.current_preview_idx + 1} ({w}x{h})")
        
        # 使用新的按需加载器加载缓冲页
        if self.pdf_loader:
            # 加载当前页 + 前后各 1 页
            self.pdf_loader.load_for_preview(self.current_preview_idx, buffer_pages=1)

        # 格式转换（cv2 BGR → Qt RGB）
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 创建带提示箭头的图像
        pixmap = QPixmap.fromImage(qt_img)
        
        # 获取预览区域尺寸（使用父容器尺寸，避免被之前的pixmap影响）
        # 获取预览分组的大小作为参考
        preview_group = self.preview_label.parentWidget()
        if preview_group:
            # 减去边距和控件占用空间
            label_width = preview_group.width() - 20
            label_height = preview_group.height() - 80  # 预留空间给页码导航等
        else:
            label_width = self.preview_label.width()
            label_height = self.preview_label.height()
        
        # 确保最小尺寸
        label_width = max(label_width, 400)
        label_height = max(label_height, 300)
        
        # 计算缩放比例
        scaled_pixmap = pixmap.scaled(
            label_width,
            label_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # 在缩放后的图像上绘制箭头
        painter = QPainter(scaled_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 根据悬停区域绘制箭头提示
        if self.hover_region == "left":
            # 左侧箭头（绿色三角形）
            arrow_width = int(scaled_pixmap.width() * 0.15)
            arrow_height = int(scaled_pixmap.height() * 0.15)
            arrow_x = int(scaled_pixmap.width() * 0.05)
            arrow_y = scaled_pixmap.height() // 2
            
            painter.setBrush(QColor(76, 175, 80, 180))  # 绿色半透明
            painter.setPen(Qt.NoPen)
            left_arrow = QPolygon([
                QPoint(arrow_x, arrow_y),
                QPoint(arrow_x + arrow_width, arrow_y - arrow_height // 2),
                QPoint(arrow_x + arrow_width, arrow_y + arrow_height // 2)
            ])
            painter.drawPolygon(left_arrow)
            
        elif self.hover_region == "right":
            # 右侧箭头（蓝色三角形）
            arrow_width = int(scaled_pixmap.width() * 0.15)
            arrow_height = int(scaled_pixmap.height() * 0.15)
            arrow_x = int(scaled_pixmap.width() * 0.95)
            arrow_y = scaled_pixmap.height() // 2
            
            painter.setBrush(QColor(33, 150, 243, 180))  # 蓝色半透明
            painter.setPen(Qt.NoPen)
            right_arrow = QPolygon([
                QPoint(arrow_x, arrow_y),
                QPoint(arrow_x - arrow_width, arrow_y - arrow_height // 2),
                QPoint(arrow_x - arrow_width, arrow_y + arrow_height // 2)
            ])
            painter.drawPolygon(right_arrow)
        
        # 不再在图片上绘制选中状态，改为更新独立的状态标签
        painter.end()
        self.preview_label.setPixmap(scaled_pixmap)

        # 更新顶部状态标签（显示选中数量）
        if self.selected_pages:
            total_pages = len(self.image_list)
            if total_pages > 0:
                status_text = f"已选中：{len(self.selected_pages)}/{total_pages} 页"
            else:
                status_text = f"已选中：{len(self.selected_pages)} 页"
            self.selection_status_label.setText(status_text)
            self.selection_status_label.setVisible(True)
        else:
            self.selection_status_label.setVisible(False)

        # 更新底部状态标签（显示当前页选中状态）
        if self.current_preview_idx in self.selected_pages:
            self.page_status_label.setText(f"✓ 第{self.current_preview_idx + 1}页 (已选中)")
            self.page_status_label.setStyleSheet("""
                QLabel {
                    background-color: #4caf50;
                    color: white;
                    padding: 5px;
                    border-radius: 3px;
                    font-weight: bold;
                    font-size: 12px;
                }
            """)
        else:
            self.page_status_label.setText(f"✗ 第{self.current_preview_idx + 1}页 (未选中)")
            self.page_status_label.setStyleSheet("""
                QLabel {
                    background-color: #f44336;
                    color: white;
                    padding: 5px;
                    border-radius: 3px;
                    font-weight: bold;
                    font-size: 12px;
                }
            """)

        # 更新页码显示
        current_page = self.current_preview_idx + 1
        display_total = self.total_pages if self.total_pages > 0 else len(self.image_list)
        self.page_label.setText(f"页码：{current_page}/{display_total}")

        # 更新导航按钮状态
        self.prev_btn.setEnabled(self.current_preview_idx > 0)
        self.next_btn.setEnabled(self.current_preview_idx < len(self.image_list) - 1)
        
        # 更新快捷切换按钮状态（根据当前预览页）
        self.update_quick_switch_button()
    
    def prev_page(self):
        """上一页（按需生成预览）"""
        if self.current_preview_idx > 0:
            self.current_preview_idx -= 1
            # 如果 Ctrl 按下，连续选择
            if self.ctrl_pressed and self.selection_start_page >= 0:
                self._update_continuous_selection()
            else:
                # 更新预览（会自动按需生成）
                self.update_preview()
                # 后台预加载相邻页面
                self._preload_adjacent_pages()
        elif self.ctrl_pressed and self.selection_start_page >= 0:
            # 即使在第一页，如果 Ctrl 按下也要更新选择
            self._update_continuous_selection()

    def next_page(self):
        """下一页（按需生成预览）"""
        if self.current_preview_idx < len(self.image_list) - 1:
            self.current_preview_idx += 1
            # 如果 Ctrl 按下，连续选择
            if self.ctrl_pressed and self.selection_start_page >= 0:
                self._update_continuous_selection()
            else:
                # 更新预览（会自动按需生成）
                self.update_preview()
                # 后台预加载相邻页面
                self._preload_adjacent_pages()
        elif self.ctrl_pressed and self.selection_start_page >= 0:
            # 即使在最后一页，如果 Ctrl 按下也要更新选择
            self._update_continuous_selection()

    def _preload_adjacent_pages(self):
        """后台预加载相邻页面预览"""
        if not hasattr(self, 'pdf_loader_v2') or not self.pdf_loader_v2:
            return
        
        def preload():
            """在后台线程预加载"""
            try:
                # 预加载前后各2页
                for offset in [-2, -1, 1, 2]:
                    page_idx = self.current_preview_idx + offset
                    if 0 <= page_idx < len(self.image_list):
                        if self.image_list[page_idx] is None:
                            img_path = self.pdf_loader_v2.get_preview(page_idx)
                            if img_path:
                                self.image_list[page_idx] = img_path
            except Exception as e:
                print(f"预加载页面失败: {e}")
        
        # 在后台线程执行预加载
        import threading
        threading.Thread(target=preload, daemon=True).start()
    
    def _ensure_page_loaded(self, page_idx: int):
        """确保页面已加载（V2）- 加载当前页和前后各1页（100dpi预览）"""
        if hasattr(self, 'pdf_loader_v2') and self.pdf_loader_v2:
            # 在完全按需架构下，不需要预加载
            # 页面会在需要时自动按需生成
            pass
    
    def jump_to_page(self):
        """跳转到指定页面（按需生成预览）"""
        try:
            page_num = int(self.page_input.text())
            if page_num < 1:
                page_num = 1
            max_page = self.total_pdf_pages if self.total_pdf_pages > 0 else len(self.image_list)
            if page_num > max_page:
                page_num = max_page
            
            self.current_preview_idx = page_num - 1
            
            # 使用 V2 加载器 - 按需生成预览
            if hasattr(self, 'pdf_loader_v2') and self.pdf_loader_v2:
                # 直接更新预览（会自动按需生成）
                self.update_preview()
                # 后台预加载相邻页面
                self._preload_adjacent_pages()
            else:
                self.update_preview()
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的页码")
    
    def _update_range_inputs(self):
        """根据选中的页面更新范围输入框"""
        if not self.selected_pages:
            return
        
        # 获取选中的页面范围
        start = min(self.selected_pages) + 1  # 转换为1-based
        end = max(self.selected_pages) + 1
        
        # 更新输入框
        self.range_start_input.setText(str(start))
        self.range_end_input.setText(str(end))
        
        # 更新页面选择UI状态
        self.update_page_selection_ui()
    
    def select_range(self):
        """选择指定范围的页面"""
        try:
            # 用户输入的是页码（从 1 开始）
            start_page = int(self.range_start_input.text())
            end_page = int(self.range_end_input.text())
            
            # 使用 PDF 总页数（如果已知），否则使用已加载页数
            total_pages = self.total_pdf_pages if self.total_pdf_pages > 0 else len(self.image_list)
            max_loaded_page = len(self.image_list)
            
            print(f"DEBUG select_range: start={start_page}, end={end_page}, total_pdf_pages={self.total_pdf_pages}, total_pages={total_pages}, max_loaded_page={max_loaded_page}")
            
            # 如果没有加载任何页面
            if max_loaded_page == 0:
                QMessageBox.information(
                    self, 
                    "提示", 
                    "文件正在加载中...\n\n"
                    "请等待文件加载完成后再选择"
                )
                return
            
            if start_page < 1:
                start_page = 1
            if end_page < 1:
                end_page = 1
            
            # 确保 start_page <= end_page
            if start_page > end_page:
                # 自动交换并提示用户
                start_page, end_page = end_page, start_page
                QMessageBox.information(
                    self,
                    "提示",
                    f"已自动调整范围：{start_page}-{end_page}"
                )
            
            # 检查是否超出总页数
            if end_page > total_pages:
                # 文件已加载完成，但选择的页码超出总页数
                # 询问用户是否要调整为最大页数
                reply = QMessageBox.question(
                    self,
                    "页码超出范围",
                    f"该文件总共只有 {total_pages} 页\n\n"
                    f"您选择了第 {start_page}-{end_page} 页，超出范围。\n\n"
                    f"是否自动调整为第 {start_page}-{total_pages} 页？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    end_page = total_pages
                else:
                    self.log_text.append(f"❌ 选择取消：页码超出范围（文件共{total_pages}页）")
                    self.status_bar.showMessage(f"选择取消：文件共{total_pages}页")
                    return
            
            # 转换为索引（从 0 开始）
            start_idx = start_page - 1
            end_idx = end_page - 1
            
            # 选中范围内的所有页面（存储索引，从 0 开始）
            self.selected_pages = set(range(start_idx, end_idx + 1))
            self.current_preview_idx = start_idx  # 跳转到起始页（索引）
            
            # 使用新的按需加载器加载选中的所有页面
            if self.pdf_loader:
                self.pdf_loader.load_for_selection(self.selected_pages)
                self.status_bar.showMessage(f"⏳ 正在加载第{start_page}页...")
            elif self.current_preview_idx not in self.loaded_pages and self.pdf_path:
                # 旧逻辑兼容
                self.load_page_on_demand(self.current_preview_idx)
                self.status_bar.showMessage(f"⏳ 正在加载第{start_page}页...")
            
            self.update_preview()
            
            count = end_page - start_page + 1
            self.status_bar.showMessage(f"已选中第{start_page}页到第{end_page}页，共{count}页")
            self.log_text.append(f"✅ 已选中页面范围：{start_page}-{end_page}（共{count}页）")
            
            # 如果选择的页码超出已加载范围，提示用户
            if end_page > max_loaded_page and self.pdf_path:
                unloaded_count = len([p for p in self.selected_pages if p not in self.loaded_pages])
                if unloaded_count > 0:
                    self.log_text.append(f"⚠️ 已选择第{start_page}-{end_page}页，目前仅加载到第{max_loaded_page}页")
                    self.log_text.append(f"   未加载：{unloaded_count} 页")
                    self.log_text.append(f"💡 请点击开始转换按钮开始处理")
                
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的页码范围")

    # ---------------------- 转换功能 ----------------------
    def start_conversion(self, allow_partial=True):
        """
        开始转换（V2 - 自动加载后执行）
        """
        if not self.selected_file:
            QMessageBox.warning(self, "警告", "请先加载有效的 PDF/图片 文件")
            return
        
        if not self.selected_pages:
            QMessageBox.warning(self, "警告", "请先选择要处理的页面")
            return
        
        # 获取用户选择的DPI
        selected_dpi = self.dpi_combo.currentData() if hasattr(self, 'dpi_combo') else 200
        
        # 检查DPI是否变化
        dpi_changed = (selected_dpi != self.current_dpi)
        if dpi_changed:
            self.log_text.append(f"📐 DPI 从 {self.current_dpi} 改为 {selected_dpi}，将使用新DPI生成图像...")
            self.current_dpi = selected_dpi
        
        # 直接开始转换（处理时会按需使用正确的DPI生成图像）
        self._do_conversion(list(self.selected_pages), dpi=selected_dpi)
    
    def _load_unloaded_pages(self, page_indices):
        """
        后台加载未加载的页面
        
        Args:
            page_indices: 页面索引列表
        """
        if not page_indices:
            return
        
        self.log_text.append(f"⏳ 开始后台加载 {len(page_indices)} 个未加载的页面...")
        
        # 简单实现：依次加载每个页面
        # 更好的方式是使用线程池并发加载
        for page_idx in sorted(page_indices):
            if self.pdf_path:
                self.load_page_on_demand(page_idx)
    
    def _do_conversion(self, page_indices, skip_image_check=False, dpi=200):
        """
        执行实际的转换处理
        
        Args:
            page_indices: 要处理的页面索引列表
            skip_image_check: 是否跳过图像检查（用于矢量提取模式）
            dpi: 图像渲染DPI
        """
        if not page_indices:
            return
        
        excel_path = self.outfile_edit.text().strip()  # 始终保存 Excel
        
        self.convert_btn.setEnabled(False)
        self.log_text.append("=" * 60)
        
        # 如果是矢量提取模式，直接处理 PDF 文件，不需要检查图像
        if skip_image_check and self.pdf_path and os.path.exists(self.pdf_path):
            # 纯矢量提取不需要模型，直接处理
            self.log_text.append(f"🚀 开始矢量提取处理（共 {len(page_indices)} 页）...")
            self.status_bar.showMessage(f"矢量提取处理中...（{len(page_indices)} 页）")
            
            # 启动纯矢量提取线程（不需要模型）
            self.thread = VectorExtractThread(
                pdf_path=self.pdf_path,
                page_indices=page_indices,
                excel_output_path=excel_path
            )
            self.thread.log_signal.connect(self.log_text.append)
            self.thread.finish_signal.connect(self.on_conversion_finish)
            self.thread.start()
            return
        
        # 分类页面：扫描件（需加载）vs 电子档（直接完成）
        scan_pages = []  # 扫描件页面索引（强制图像处理）
        text_vector_pages = []  # 电子档页面索引（矢量提取）
        text_image_pages = []  # 电子档页面索引（图像识别）
        valid_selected_pages = []  # 有效的图像路径列表（仅图片文件）
        page_idx_to_image = {}  # 页面索引到图像路径的映射（仅图片文件）
        
        for page_idx in sorted(page_indices):
            # 获取页面配置
            config = self.process_manager.get_config(page_idx)
            
            # 根据页面类型分类（按需分析）
            if hasattr(self, 'pdf_loader_v2') and self.pdf_loader_v2:
                # 按需获取页面类型（会触发分析）
                page_type_str = self.pdf_loader_v2.get_page_type(page_idx)
                page_type = PageType.SCANNED if page_type_str == 'scanned' else PageType.ELECTRONIC
                
                # 注册页面（如果尚未注册）
                if not config:
                    config = self.process_manager.register_page(page_idx, page_type)
                
                # 获取实际处理方法
                effective_method = config.get_effective_method()
                method_name = "图像识别" if effective_method == ProcessMethod.IMAGE else "矢量提取"
                
                if page_type == PageType.SCANNED:  # 扫描件 - 强制图像处理
                    scan_pages.append(page_idx)
                    # 图像将在后台线程中生成，不在主线程中阻塞UI
                        
                else:  # 电子档
                    if effective_method == ProcessMethod.IMAGE:
                        text_image_pages.append(page_idx)
                        # 图像将在后台线程中生成，不在主线程中阻塞UI
                            
                    else:
                        text_vector_pages.append(page_idx)
                        # 矢量提取不需要图像
            else:
                # 无法判断页面类型（通常是图片文件），强制使用图像识别
                if not config:
                    config = self.process_manager.register_page(page_idx, PageType.SCANNED)
                
                # 图片文件强制使用图像识别，不支持矢量提取
                scan_pages.append(page_idx)
                
                # 图片文件直接使用 image_list 中的路径
                if page_idx < len(self.image_list):
                    img_path = self.image_list[page_idx]
                    valid_selected_pages.append(img_path)
                    page_idx_to_image[page_idx] = img_path
        
        # 检查是否有任何页面需要处理
        total_pages = len(scan_pages) + len(text_vector_pages) + len(text_image_pages)
        if total_pages == 0:
            self.log_text.append("⚠️ 没有可处理的页面")
            return
        
        # 判断是否需要模型（扫描件或电子档选择图像识别）
        need_model = len(scan_pages) > 0 or len(text_image_pages) > 0
        
        # 只有需要模型时才检查模型加载
        if need_model and not self._ensure_model_loaded():
            QMessageBox.warning(self, "警告", "模型尚未加载完成，请稍后再试")
            self.convert_btn.setEnabled(True)
            return
        
        # 开始处理 - 禁用按钮
        self.convert_btn.setEnabled(False)
        
        # 显示处理策略
        total_scan = len(scan_pages) + len(text_image_pages)  # 所有图像处理的页面
        total_vector = len(text_vector_pages)  # 矢量提取的页面
        
        if total_scan > 0 and total_vector > 0:
            self.log_text.append(f"🚀 混合处理：🖼️ {total_scan} 页图像识别 + 📄 {total_vector} 页矢量提取")
            if scan_pages:
                self.log_text.append(f"   （其中 {len(scan_pages)} 页扫描件强制图像处理）")
        elif total_scan > 0:
            self.log_text.append(f"🖼️ 处理 {total_scan} 页（图像识别）...")
            if scan_pages:
                self.log_text.append(f"   （其中 {len(scan_pages)} 页扫描件强制图像处理）")
        elif total_vector > 0:
            self.log_text.append(f"📄 处理 {total_vector} 页（矢量提取）...")
        else:
            self.log_text.append(f"🚀 开始执行处理（共 {len(valid_selected_pages)} 页）...")
        
        self.status_bar.showMessage(f"处理中...（共 {total_pages} 页：🖼️{len(scan_pages)+len(text_image_pages)} + 📄{len(text_vector_pages)}）")
        
        self.thread = ConversionThread(
            processor=self.processor, 
            image_list=valid_selected_pages, 
            excel_output_path=excel_path,
            pdf_path=self.pdf_path if hasattr(self, 'pdf_path') else None,
            page_indices=page_indices,
            scan_pages=scan_pages,
            text_vector_pages=text_vector_pages,
            text_image_pages=text_image_pages,
            page_idx_to_image=page_idx_to_image,
            pdf_loader_v2=self.pdf_loader_v2 if hasattr(self, 'pdf_loader_v2') else None,
            dpi=dpi
        )
        self.thread.log_signal.connect(self.log_text.append)
        self.thread.finish_signal.connect(self.on_conversion_finish)
        self.thread.start()

    def on_conversion_finish(self, result):
        import os
        import platform
        
        if not result:
            self.convert_btn.setEnabled(True)
            QMessageBox.warning(self, "处理失败", "⚠️ 处理失败，请查看日志")
            self.status_bar.showMessage("处理失败")
            return

        self.last_result = result
        regions = result.get("regions", [])
        tables = result.get("tables", [])
        excel_path = result.get("excel_path")
        non_table_text = result.get("non_table_text", "")
        
        self.log_text.append("")
        self.log_text.append(f"✅ 处理完成，区域数：{len(regions)}")
        self.log_text.append(f"✅ 表格数量：{len(tables)}")
        if excel_path:
            # 规范化路径
            excel_path = os.path.normpath(excel_path)
            self.log_text.append(f"✅ Excel已保存：{excel_path}")
            # 自动打开Excel文件
            try:
                if platform.system() == 'Windows':
                    os.startfile(excel_path)
                elif platform.system() == 'Darwin':  # macOS
                    import subprocess
                    subprocess.call(['open', excel_path])
                else:  # Linux
                    import subprocess
                    subprocess.call(['xdg-open', excel_path])
                self.log_text.append("✅ 已自动打开Excel文件")
            except Exception as e:
                self.log_text.append(f"⚠️ 自动打开Excel失败：{e}")
        self.status_bar.showMessage("处理完成")
        
        # Excel 保存和自动打开完成后，重新启用按钮
        self.convert_btn.setEnabled(True)

    # ---------------------- 清空功能 ----------------------
    def clear_all(self):
        """清空所有输入与状态"""
        self.selected_file = ""
        self.image_list = []
        self.temp_files = []
        self.current_preview_idx = 0
        
        # 清空页面选择状态
        self.selected_pages = set()
        self.selection_start_page = -1
        self.selection_mode = False
        self.ctrl_pressed = False
        self.hover_region = None

        # 清空 UI 控件
        self.file_edit.clear()
        self.outfile_edit.setText("output.xlsx")
        self.log_text.clear()
        self.preview_label.setText("暂无预览")
        self.page_label.setText("页码：0/0")
        self.last_result = {}
        
        # 清空页面处理管理器
        self.process_manager.clear()
        
        # 重置DPI记录
        self.current_dpi = 200

        # 重置状态栏
        self.status_bar.showMessage("就绪 - 支持拖拽 PDF/图片文件到窗口")

    def rotate_image(self, angle):
        """旋转当前预览图像，并记录旋转角度到 pdf_loader_v2"""
        if not self.image_list or self.current_preview_idx >= len(self.image_list):
            QMessageBox.warning(self, "警告", "请先加载文件并选择要旋转的页面")
            return
        
        try:
            # 标准化角度
            angle = angle % 360
            
            # 读取当前图像
            img_path = self.image_list[self.current_preview_idx]
            if not img_path:
                QMessageBox.warning(self, "警告", "当前页面尚未加载，请等待加载完成")
                return
            img = cv2.imread(img_path)
            if img is None:
                QMessageBox.warning(self, "警告", "无法读取当前图像")
                return
            
            # 根据角度选择最优旋转方法
            if angle == 90:
                rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(img, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                # 其他角度使用仿射变换
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # 计算旋转后的图像大小
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                
                # 调整旋转矩阵的平移部分
                M[0, 2] += (new_w / 2) - center[0]
                M[1, 2] += (new_h / 2) - center[1]
                
                rotated = cv2.warpAffine(img, M, (new_w, new_h), 
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
            
            # 保存旋转后的图像（覆盖原文件）
            cv2.imwrite(img_path, rotated)
            
            # 记录旋转角度到 pdf_loader_v2，确保处理时使用旋转后的图像
            if hasattr(self, 'pdf_loader_v2') and self.pdf_loader_v2:
                current_page = self.current_preview_idx
                current_rotation = self.pdf_loader_v2.get_page_rotation(current_page)
                new_rotation = (current_rotation + angle) % 360
                self.pdf_loader_v2.set_page_rotation(current_page, new_rotation)
            
            # 更新预览
            self.update_preview()
            
            # 日志更新
            self.log_text.append(f"✅ 已将第{self.current_preview_idx + 1}页旋转{angle}度")
            self.status_bar.showMessage(f"已旋转{angle}度")
            
        except Exception as e:
            QMessageBox.critical(self, "旋转失败", f"旋转出错：{str(e)}")
    
    # 旋转按钮已移除，使用 RapidTableDetection 自动处理表格旋转

    def _start_rotate_thread(self, pages_to_rotate, angle):
        """启动旋转线程"""

        self.status_bar.showMessage(f"正在旋转 {len(pages_to_rotate)} 页...")
        
        # 确保所有要旋转的页面路径都已加载到 image_list
        if hasattr(self, 'pdf_loader_v2') and self.pdf_loader_v2:
            for page_idx in pages_to_rotate:
                if not self.image_list[page_idx]:  # 如果路径为空
                    preview_path = self.pdf_loader_v2.get_preview(page_idx)
                    if preview_path:
                        self.image_list[page_idx] = preview_path
        
        # 显示页码（索引 + 1）
        page_nums = [str(p + 1) for p in sorted(pages_to_rotate)]
        self.log_text.append(f"准备旋转的页面：{', '.join(page_nums)} 页")

        # 创建并启动旋转线程
        self.rotate_thread = RotateThread(self.image_list, pages_to_rotate, angle)
        self.rotate_thread.log_signal.connect(self.log_text.append)
        self.rotate_thread.progress_signal.connect(self._on_rotate_progress)
        self.rotate_thread.finish_signal.connect(self._on_rotate_finished)
        self.rotate_thread.start()

    def _wait_and_rotate(self, loaded_pages, unloaded_pages, angle):
        """等待页面加载完成后旋转"""
        self._rotate_after_load = {
            'loaded_pages': loaded_pages,
            'unloaded_pages': set(unloaded_pages),
            'angle': angle
        }
        self.log_text.append(f"⏳ 正在加载中，请稍候...")

    def _on_rotate_progress(self, current, total):
        """旋转进度更新"""
        self.status_bar.showMessage(f"正在旋转... {current}/{total}")

    def _on_rotate_finished(self, rotated_count, total_pages, angle):
        """旋转完成处理"""
        # 更新预览
        self.update_preview()

        # 记录旋转角度到 pdf_loader_v2，确保处理时使用旋转后的图像
        if hasattr(self, 'pdf_loader_v2') and self.pdf_loader_v2:
            # 获取实际旋转的页面（从 RotateThread 的 pages_to_rotate）
            if hasattr(self, 'rotate_thread') and self.rotate_thread:
                for page_idx in self.rotate_thread.pages_to_rotate:
                    # 累加旋转角度
                    current_rotation = self.pdf_loader_v2.get_page_rotation(page_idx)
                    new_rotation = (current_rotation + angle) % 360
                    self.pdf_loader_v2.set_page_rotation(page_idx, new_rotation)

        # 日志更新（不弹窗）
        self.log_text.append(f"✅ 已旋转{rotated_count}/{total_pages}页 {angle}度")
        self.status_bar.showMessage(f"已旋转{rotated_count}页")

    def _on_load_complete(self):
        """页面加载完成后的回调"""
        self.log_text.append(f"✅ 页面加载完成")
        
        # 更新 image_list 中的路径（从 pdf_loader_v2 获取）
        if hasattr(self, 'pdf_loader_v2') and self.pdf_loader_v2:
            for page_idx in range(len(self.image_list)):
                if not self.image_list[page_idx]:  # 如果路径为空
                    # 尝试从 pdf_loader_v2 获取预览路径
                    preview_path = self.pdf_loader_v2.get_preview(page_idx)
                    if preview_path:
                        self.image_list[page_idx] = preview_path
        
        self._check_pending_operations()
    
    def _check_pending_operations(self):
        """检查并执行等待的操作"""
        if not hasattr(self, '_pending_operation') or not self._pending_operation:
            return
        
        op = self._pending_operation
        op_type = op.get('type')
        
        if op_type == 'rotate':
            # 执行旋转
            pages = op['pages']
            angle = op['angle']
            self._pending_operation = None
            self._start_rotate_thread(pages, angle)
            
        elif op_type == 'convert':
            # 执行转换
            pages = op['pages']
            self._pending_operation = None
            self._do_conversion(pages)
    
    def _check_pending_rotation(self):
        """检查是否有等待的旋转任务（旧版兼容）"""
        self._check_pending_operations()

# ---------------------- 程序入口 ----------------------
def main():
    # 解决多进程打包问题
    freeze_support()

    # 启用高 DPI 支持（Qt6 中这些属性已默认启用，无需设置）
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # 启动 Qt 应用
    app = QApplication(sys.argv)
    
    # 设置应用图标
    if getattr(sys, 'frozen', False):
        # 打包后的环境
        icon_path = os.path.join(sys._MEIPASS, "favicon.ico")
    else:
        # 开发环境
        icon_path = os.path.join(PROJECT_ROOT, "favicon.ico")
    
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    window = TableConverterWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
