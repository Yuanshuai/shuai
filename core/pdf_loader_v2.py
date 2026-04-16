"""
PDF 智能加载管理器 V2
- 完全按需分析：页面类型在需要时才分析
- 完全按需加载：预览图像在需要时才生成
"""
import os
import fitz
import cv2
import numpy as np
from typing import List, Set, Callable, Optional, Dict, Tuple
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor


class PageStatus(Enum):
    """页面状态"""
    UNKNOWN = "unknown"        # 尚未分析
    NO_IMAGE = "no_image"      # 无内嵌图像，直接完成
    UNLOADED = "unloaded"      # 有内嵌图像，未加载
    LOADING = "loading"        # 正在加载
    LOADED = "loaded"          # 已加载完成


class PDFLoaderV2:
    """PDF 智能加载管理器 V2 - 完全按需"""
    
    def __init__(self, pdf_path: str, max_workers: int = 3):
        """
        Args:
            pdf_path: PDF 文件路径
            max_workers: 最大并发加载线程数
        """
        self.pdf_path = pdf_path
        self.max_workers = max_workers
        
        # 打开 PDF 文档
        self.doc = fitz.open(pdf_path)
        self.total_pages = len(self.doc)
        
        # 创建 data 文件夹（在项目根目录下）
        project_root = os.path.dirname(os.path.dirname(__file__))
        self.data_dir = os.path.join(project_root, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 页面状态管理（完全按需，初始状态为 UNKNOWN）
        self.page_status: Dict[int, PageStatus] = {}  # 页面索引 -> 状态
        self.page_has_image: Dict[int, bool] = {}     # 页面索引 -> 是否有内嵌图像
        self.page_image_path: Dict[int, str] = {}     # 页面索引 -> 图像路径
        
        # 页面旋转角度记录
        self.page_rotation: Dict[int, int] = {}       # 页面索引 -> 旋转角度（0, 90, 180, 270）
        
        # 加载状态
        self.loaded_pages: Set[int] = set()           # 已加载的页面索引
        self.loading_pages: Set[int] = set()          # 正在加载的页面索引
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        
        # 回调函数
        self.on_page_loaded: Optional[Callable[[int, str, bool], None]] = None
        self.on_all_loaded: Optional[Callable[[], None]] = None
        
        # 初始化所有页面为 UNKNOWN 状态
        for i in range(self.total_pages):
            self.page_status[i] = PageStatus.UNKNOWN
    
    def set_callbacks(self, 
                      on_page_loaded: Optional[Callable[[int, str, bool], None]] = None,
                      on_all_loaded: Optional[Callable[[], None]] = None):
        """设置回调函数"""
        self.on_page_loaded = on_page_loaded
        self.on_all_loaded = on_all_loaded
    
    def _analyze_page(self, page_idx: int) -> Tuple[bool, bool]:
        """
        分析页面类型
        
        Returns:
            (has_image, has_text): 是否有图像，是否有文本
        """
        page = self.doc[page_idx]
        
        # 检查是否有内嵌图像
        has_image = len(page.get_images(full=True)) > 0
        
        # 检查是否有文本（电子档通常有文本层）
        text = page.get_text().strip()
        has_text = len(text) > 50  # 至少50个字符才认为是电子档
        
        return has_image, has_text
    
    def _ensure_page_analyzed(self, page_idx: int) -> PageStatus:
        """
        确保页面已分析，如果未分析则立即分析
        
        Returns:
            页面状态
        """
        with self.lock:
            current_status = self.page_status.get(page_idx, PageStatus.UNKNOWN)
            
            # 如果已经分析过，直接返回
            if current_status != PageStatus.UNKNOWN:
                return current_status
        
        # 需要分析页面
        has_image, has_text = self._analyze_page(page_idx)
        
        with self.lock:
            self.page_has_image[page_idx] = has_image
            
            if has_image:
                # 有内嵌图像，需要加载（扫描件或混合）
                self.page_status[page_idx] = PageStatus.UNLOADED
            elif has_text:
                # 无图像但有文本，是电子档
                self.page_status[page_idx] = PageStatus.NO_IMAGE
                self.loaded_pages.add(page_idx)
            else:
                # 无图像无文本，可能是空白页或特殊页面
                self.page_status[page_idx] = PageStatus.UNLOADED
            
            return self.page_status[page_idx]
    
    def get_preview(self, page_idx: int) -> Optional[str]:
        """
        按需获取页面预览图像
        如果页面尚未分析，会先分析；如果预览尚未生成，则立即生成
        
        Args:
            page_idx: 页面索引
            
        Returns:
            预览图像路径，如果失败则返回None
        """
        if page_idx < 0 or page_idx >= self.total_pages:
            return None
        
        # 第一步：确保页面已分析
        status = self._ensure_page_analyzed(page_idx)
        
        # 第二步：如果已经生成过预览，直接返回
        if page_idx in self.page_image_path:
            return self.page_image_path[page_idx]
        
        # 第三步：按需生成预览
        try:
            return self._render_page_preview(page_idx)
        except Exception as e:
            print(f"生成预览失败 (页面 {page_idx + 1}): {e}")
            return None
    
    def _render_page_preview(self, page_idx: int, dpi: int = 100) -> str:
        """
        渲染页面为预览图像
        
        Args:
            page_idx: 页面索引
            dpi: 渲染DPI
            
        Returns:
            图像文件路径
        """
        page = self.doc[page_idx]
        
        # 使用指定DPI渲染
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        
        # 转换为numpy数组
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # 转换为BGR（OpenCV默认格式）
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 保存图像
        image_path = os.path.join(self.data_dir, f"page_{page_idx}_preview.png")
        cv2.imwrite(image_path, img)
        
        # 更新状态
        with self.lock:
            self.page_image_path[page_idx] = image_path
            
            # 更新状态（扫描件从 UNLOADED 变为 LOADED）
            current_status = self.page_status.get(page_idx)
            if current_status == PageStatus.UNLOADED:
                self.page_status[page_idx] = PageStatus.LOADED
                self.loaded_pages.add(page_idx)
        
        return image_path
    
    def get_page_for_processing(self, page_idx: int, dpi: int = 200) -> Optional[str]:
        """
        获取用于处理的页面图像（更高DPI），会自动应用旋转
        
        Args:
            page_idx: 页面索引
            dpi: 渲染DPI（处理时使用更高DPI）
            
        Returns:
            图像文件路径
        """
        if page_idx < 0 or page_idx >= self.total_pages:
            return None
        
        # 确保页面已分析
        self._ensure_page_analyzed(page_idx)
        
        # 获取旋转角度
        rotation = self.page_rotation.get(page_idx, 0)
        
        # 检查是否是电子档（无内嵌图像）
        has_image = self.page_has_image.get(page_idx, True)
        
        # 使用包含DPI和旋转信息的缓存键
        cache_key = f"page_{page_idx}_dpi{dpi}_rot{rotation}.png"
        image_path = os.path.join(self.data_dir, cache_key)
        
        # 检查缓存是否存在
        if os.path.exists(image_path):
            return image_path
        
        # 渲染新DPI图像
        page = self.doc[page_idx]
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        # 转换为BGR（OpenCV默认格式）
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 应用自动纠偏（角度微调）
        img = self._auto_deskew(img)
        
        # 应用旋转
        if rotation > 0:
            img = self._apply_rotation(img, rotation)
        
        cv2.imwrite(image_path, img)
        
        # 如果是电子档，同时更新预览图路径（用于UI显示）
        if not has_image:
            with self.lock:
                self.page_image_path[page_idx] = image_path
        
        return image_path
    
    def is_page_loaded(self, page_idx: int) -> bool:
        """检查页面是否已加载"""
        # 先确保页面已分析
        status = self._ensure_page_analyzed(page_idx)
        return status in (PageStatus.LOADED, PageStatus.NO_IMAGE)
    
    def get_page_type(self, page_idx: int) -> str:
        """
        获取页面类型
        
        Returns:
            'scanned' - 扫描件（有内嵌图像）
            'electronic' - 电子档（无内嵌图像）
        """
        # 确保页面已分析
        self._ensure_page_analyzed(page_idx)
        
        has_image = self.page_has_image.get(page_idx, True)
        return 'scanned' if has_image else 'electronic'
    
    def set_page_rotation(self, page_idx: int, angle: int):
        """
        设置页面旋转角度
        
        Args:
            page_idx: 页面索引
            angle: 旋转角度（0, 90, 180, 270）
        """
        # 标准化角度
        angle = angle % 360
        self.page_rotation[page_idx] = angle
        
        # 清除该页面的处理图像缓存，强制重新生成
        self._clear_page_cache(page_idx)
    
    def get_page_rotation(self, page_idx: int) -> int:
        """
        获取页面旋转角度
        
        Returns:
            旋转角度（0, 90, 180, 270）
        """
        return self.page_rotation.get(page_idx, 0)
    
    def _clear_page_cache(self, page_idx: int):
        """清除页面的缓存图像"""
        # 清除各种DPI的缓存
        for dpi in [100, 150, 200, 300]:
            cache_key = f"page_{page_idx}_dpi{dpi}.png"
            image_path = os.path.join(self.data_dir, cache_key)
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
    
    def _apply_rotation(self, img: np.ndarray, angle: int) -> np.ndarray:
        """
        应用旋转到图像
        
        Args:
            img: 输入图像
            angle: 旋转角度（0, 90, 180, 270）
            
        Returns:
            旋转后的图像
        """
        if angle == 0:
            return img
        elif angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # 其他角度使用仿射变换
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            return cv2.warpAffine(img, M, (new_w, new_h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    
    def _auto_deskew(self, img: np.ndarray) -> np.ndarray:
        """
        自动纠偏（角度微调）- 使用投影法检测并纠正小角度倾斜
        
        Args:
            img: 输入图像（BGR格式，OpenCV默认）
            
        Returns:
            纠偏后的图像（BGR格式）
        """
        try:
            # 转换为灰度图（图像是BGR格式，但BGR2GRAY和RGB2GRAY效果相同）
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
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
            
            # 如果角度很小，认为不需要纠偏
            if abs(best_angle) < 0.5:
                return img
            
            # 应用纠偏
            corrected = rotate_image(img, best_angle)
            return corrected
            
        except Exception as e:
            # 纠偏失败，返回原图
            return img
    
    def get_unloaded_pages(self, page_indices: List[int]) -> List[int]:
        """
        获取未加载的页面列表（兼容旧接口）
        
        Args:
            page_indices: 页面索引列表
            
        Returns:
            未加载的页面索引列表
        """
        unloaded = []
        for page_idx in page_indices:
            if not self.is_page_loaded(page_idx):
                unloaded.append(page_idx)
        return unloaded
    
    def load_pages_async(self, page_indices: List[int], on_complete: Optional[Callable] = None, 
                         dpi: int = 200, force_reload: bool = False):
        """
        异步加载页面（兼容旧接口，实际同步执行）
        
        Args:
            page_indices: 页面索引列表
            on_complete: 加载完成回调
            dpi: 加载DPI
            force_reload: 是否强制重新加载
        """
        def load_worker():
            for page_idx in page_indices:
                # 按需获取页面图像
                self.get_page_for_processing(page_idx, dpi)
            
            # 调用完成回调
            if on_complete:
                on_complete()
        
        # 在后台线程执行
        threading.Thread(target=load_worker, daemon=True).start()
    
    def close(self):
        """关闭加载器，清理资源"""
        self.executor.shutdown(wait=True)
        if self.doc:
            self.doc.close()
