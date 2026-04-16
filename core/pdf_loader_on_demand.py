"""
PDF 按需加载管理器
支持智能调度、并发加载、优先级队列
"""
import os
import fitz
import cv2
import numpy as np
from typing import List, Set, Callable, Optional, Dict
from collections import OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import IntEnum


class LoadPriority(IntEnum):
    """加载优先级"""
    URGENT = 0      # 紧急：当前预览页
    HIGH = 1        # 高：预览缓冲页（上下页）
    NORMAL = 2      # 普通：选中的页面
    LOW = 3         # 低：预加载页面


@dataclass
class LoadRequest:
    """加载请求"""
    page_idx: int           # 页面索引
    priority: LoadPriority  # 优先级
    request_id: str = ""    # 请求 ID（用于追踪）
    
    def __lt__(self, other):
        # 用于优先级队列排序
        return self.priority < other.priority


@dataclass
class LoadResult:
    """加载结果"""
    page_idx: int       # 页面索引
    image_path: str     # 图片路径
    success: bool       # 是否成功
    error: str = ""     # 错误信息


class PDFOnDemandLoader:
    """PDF 按需加载管理器"""
    
    def __init__(self, pdf_path: str, cache_size: int = 50, max_workers: int = 3):
        """
        Args:
            pdf_path: PDF 文件路径
            cache_size: 缓存的页面数量
            max_workers: 最大并发加载线程数
        """
        self.pdf_path = pdf_path
        self.cache_size = cache_size
        self.max_workers = max_workers
        
        # 打开 PDF 文档
        self.doc = fitz.open(pdf_path)
        self.total_pages = len(self.doc)
        
        # 创建 data 文件夹
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 缓存管理（LRU）
        self.cache = OrderedDict()  # page_idx -> image_path
        self.loaded_pages: Set[int] = set()  # 已加载的页面索引
        self.loading_pages: Set[int] = set()  # 正在加载的页面索引
        
        # 加载队列（按优先级排序）
        self.load_queue: List[LoadRequest] = []
        self.queue_lock = threading.Lock()
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: Dict[int, Future] = {}  # page_idx -> Future
        
        # 回调函数
        self.on_page_loaded: Optional[Callable[[int, str, bool], None]] = None
        
        # 线程锁
        self.lock = threading.Lock()
    
    def set_callback(self, callback: Callable[[int, str, bool], None]):
        """
        设置页面加载完成回调
        
        Args:
            callback: 回调函数 (page_idx, image_path, success)
        """
        self.on_page_loaded = callback
    
    def get_page_path(self, page_idx: int) -> str:
        """获取页面图片路径"""
        return os.path.join(self.data_dir, f"pdf_page_{page_idx + 1}.png")
    
    def _extract_embedded_image(self, page) -> Optional[np.ndarray]:
        """
        尝试提取页面中内嵌的图像
        
        Args:
            page: fitz.Page 对象
            
        Returns:
            提取的图像数据，如果没有则返回 None
        """
        try:
            # 获取页面中的图像列表
            img_list = page.get_images(full=True)
            
            if not img_list:
                return None
            
            # 找到最大的图像（通常是主要内容）
            max_area = 0
            best_img = None
            
            for img_index, img in enumerate(img_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # 尝试解码图像
                try:
                    img_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if img_array is not None:
                        h, w = img_array.shape[:2]
                        area = h * w
                        # 选择面积最大的图像（通常是要提取的内容）
                        if area > max_area:
                            max_area = area
                            best_img = img_array
                except:
                    continue
            
            return best_img
            
        except Exception as e:
            return None
    
    def _has_embedded_image(self, page) -> bool:
        """检查页面是否有内嵌图像"""
        img_list = page.get_images(full=True)
        return len(img_list) > 0
    
    def load_single_page(self, page_idx: int, force_render: bool = False) -> Optional[str]:
        """
        加载单页 PDF
        - 有内嵌图像：提取图像（需要加载）
        - 无内嵌图像：标记为加载完成（不需要预览图）
        
        Args:
            page_idx: 页面索引（0-based）
            force_render: 是否强制渲染（用于预览）
            
        Returns:
            加载后的图片路径，失败返回 None
        """
        if page_idx < 0 or page_idx >= self.total_pages:
            return None
        
        try:
            # 加载页面
            page = self.doc[page_idx]
            out_path = self.get_page_path(page_idx)
            
            # 检查是否有内嵌图像
            has_embedded = self._has_embedded_image(page)
            
            if has_embedded:
                # 有内嵌图像：提取图像
                embedded_img = self._extract_embedded_image(page)
                if embedded_img is not None:
                    cv2.imwrite(out_path, embedded_img)
                else:
                    # 提取失败，渲染页面（100dpi）
                    pix = page.get_pixmap(dpi=100, alpha=False)
                    pix.save(out_path)
            else:
                # 无内嵌图像：渲染页面（100dpi低质量预览）
                pix = page.get_pixmap(dpi=100, alpha=False)
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
            
            # 更新缓存
            with self.lock:
                self.cache[page_idx] = out_path
                self.loaded_pages.add(page_idx)
                self.loading_pages.discard(page_idx)
                
                # LRU 淘汰
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            
            return out_path
            
        except Exception as e:
            with self.lock:
                self.loading_pages.discard(page_idx)
            return None
    
    def _load_page_worker(self, page_idx: int) -> LoadResult:
        """
        加载页面的工作线程
        
        Args:
            page_idx: 页面索引
            
        Returns:
            加载结果
        """
        image_path = self.load_single_page(page_idx)
        success = image_path is not None
        
        # 触发回调
        if self.on_page_loaded:
            self.on_page_loaded(page_idx, image_path or "", success)
        
        # 任务完成后，继续处理队列中的其他请求
        self._process_queue()
        
        return LoadResult(
            page_idx=page_idx,
            image_path=image_path or "",
            success=success
        )
    
    def _process_queue(self):
        """处理加载队列（带锁）"""
        with self.queue_lock:
            self._process_queue_unlocked()
    
    def request_pages(self, requests: List[LoadRequest]):
        """
        批量请求加载页面
        
        Args:
            requests: 加载请求列表
        """
        with self.queue_lock:
            # 添加请求到队列
            for req in requests:
                # 跳过已加载或正在加载的页面
                if req.page_idx in self.loaded_pages or req.page_idx in self.loading_pages:
                    continue
                
                # 添加到队列
                self.load_queue.append(req)
                
                # 按优先级排序
                self.load_queue.sort()
            
            # 处理队列（在锁内直接调用，不需要再次获取锁）
            self._process_queue_unlocked()
    
    def _process_queue_unlocked(self):
        """处理加载队列（假设已经持有锁）"""
        # 检查是否有空闲的线程
        active_futures = {idx: f for idx, f in self.futures.items() if not f.done()}
        self.futures = active_futures
        
        available_workers = self.max_workers - len(active_futures)
        
        if available_workers <= 0:
            return
        
        # 从队列中取出优先级最高的请求
        to_load = []
        while self.load_queue and len(to_load) < available_workers:
            req = self.load_queue.pop(0)
            
            # 再次检查是否已被加载
            if req.page_idx not in self.loaded_pages and req.page_idx not in self.loading_pages:
                to_load.append(req.page_idx)
                self.loading_pages.add(req.page_idx)
        
        # 提交到线程池
        for page_idx in to_load:
            future = self.executor.submit(self._load_page_worker, page_idx)
            self.futures[page_idx] = future
    
    # ==================== 高级接口 ====================
    
    def load_for_preview(self, current_page: int, buffer_pages: int = 1):
        """
        为预览加载页面（当前页 + 缓冲页）
        
        Args:
            current_page: 当前预览页索引
            buffer_pages: 缓冲页数（前后各加载多少页）
        """
        requests = []
        
        # 当前页（紧急优先级）
        if current_page not in self.loaded_pages:
            requests.append(LoadRequest(
                page_idx=current_page,
                priority=LoadPriority.URGENT,
                request_id=f"preview_current_{current_page}"
            ))
        
        # 缓冲页（高优先级）
        for offset in range(1, buffer_pages + 1):
            # 上一页
            prev_page = current_page - offset
            if prev_page >= 0 and prev_page not in self.loaded_pages:
                requests.append(LoadRequest(
                    page_idx=prev_page,
                    priority=LoadPriority.HIGH,
                    request_id=f"preview_prev_{prev_page}"
                ))
            
            # 下一页
            next_page = current_page + offset
            if next_page < self.total_pages and next_page not in self.loaded_pages:
                requests.append(LoadRequest(
                    page_idx=next_page,
                    priority=LoadPriority.HIGH,
                    request_id=f"preview_next_{next_page}"
                ))
        
        self.request_pages(requests)
    
    def load_for_jump(self, target_page: int, load_count: int = 3):
        """
        为跳转加载页面（目标页及附近页面）
        
        Args:
            target_page: 目标页面索引
            load_count: 加载页面数量（目标页前后各加载多少页）
        """
        requests = []
        
        # 目标页（紧急优先级）
        if target_page not in self.loaded_pages:
            requests.append(LoadRequest(
                page_idx=target_page,
                priority=LoadPriority.URGENT,
                request_id=f"jump_target_{target_page}"
            ))
        
        # 附近页面（高优先级）
        for offset in range(1, load_count):
            # 上一页
            prev_page = target_page - offset
            if prev_page >= 0 and prev_page not in self.loaded_pages:
                requests.append(LoadRequest(
                    page_idx=prev_page,
                    priority=LoadPriority.HIGH,
                    request_id=f"jump_prev_{prev_page}"
                ))
            
            # 下一页
            next_page = target_page + offset
            if next_page < self.total_pages and next_page not in self.loaded_pages:
                requests.append(LoadRequest(
                    page_idx=next_page,
                    priority=LoadPriority.HIGH,
                    request_id=f"jump_next_{next_page}"
                ))
        
        self.request_pages(requests)
    
    def load_for_selection(self, selected_pages: Set[int]):
        """
        为选中的页面加载
        
        Args:
            selected_pages: 选中的页面索引集合
        """
        requests = []
        
        for page_idx in sorted(selected_pages):
            if page_idx not in self.loaded_pages:
                requests.append(LoadRequest(
                    page_idx=page_idx,
                    priority=LoadPriority.NORMAL,
                    request_id=f"select_{page_idx}"
                ))
        
        self.request_pages(requests)
    
    def preload_pages(self, start_idx: int, count: int):
        """
        预加载页面（低优先级）
        
        Args:
            start_idx: 起始页面索引
            count: 加载数量
        """
        requests = []
        
        for i in range(count):
            page_idx = start_idx + i
            if page_idx < self.total_pages and page_idx not in self.loaded_pages:
                requests.append(LoadRequest(
                    page_idx=page_idx,
                    priority=LoadPriority.LOW,
                    request_id=f"preload_{page_idx}"
                ))
        
        self.request_pages(requests)
    
    # ==================== 查询接口 ====================
    
    def is_loaded(self, page_idx: int) -> bool:
        """检查页面是否已加载"""
        return page_idx in self.loaded_pages
    
    def get_loaded_pages(self) -> Set[int]:
        """获取所有已加载的页面索引"""
        return self.loaded_pages.copy()
    
    def get_loaded_count(self) -> int:
        """获取已加载的页面数量"""
        return len(self.loaded_pages)
    
    def get_image_path(self, page_idx: int) -> Optional[str]:
        """获取已加载页面的图片路径"""
        return self.cache.get(page_idx)
    
    # ==================== 清理接口 ====================
    
    def shutdown(self):
        """关闭加载器"""
        self.executor.shutdown(wait=False)
        if self.doc:
            self.doc.close()
    
    def clear_cache(self):
        """清空缓存（不关闭文档）"""
        with self.lock:
            self.cache.clear()
            self.loaded_pages.clear()
            self.load_queue.clear()
