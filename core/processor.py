import atexit
import os
import re
import sys
import time
import uuid
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import fitz
import numpy as np
import requests
from rapid_layout import RapidLayout, EngineType, ModelType
from rapidocr import RapidOCR
from rapidocr.utils.typings import EngineType as RapidOCREngineType, LangDet, LangRec, ModelType as RapidOCRModelType, OCRVersion
from core.orientation_detector import OrientationDetector
from core.rapid_orientation_detector import RapidOrientationDetector

from core.settings import Settings
from core.pdf_vector_table import PDFVectorTableExtractor, extract_tables_from_pdf
from algorithms.table_recognition import (
    TableResult,
    build_table_grid,
    parse_table_structure,
    save_tables_to_single_workbook,
)


@dataclass
class Region:
    page_index: int
    label: str
    score: float
    bbox: List[float]
    text: str
    page_span: Optional[List[int]] = None
    meta: Optional[Dict[str, Any]] = None
    crop: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_index": self.page_index,
            "label": self.label,
            "score": self.score,
            "bbox": self.bbox,
            "text": self.text,
            "page_span": self.page_span,
            "meta": self.meta,
        }


class SpinCorrector:
    """基于投影的文档倾斜校正器"""
    
    def __init__(self):
        pass
    
    def correct(self, image_bgr: np.ndarray, auto_rotate: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        校正图像倾斜
        只进行小角度校正（-10 到 10 度），大角度由用户手动调整
        使用双向搜索 + 贪心策略，步长 0.1 度
        """
        # 小角度校正
        if auto_rotate:
            small_angle, corrected = self._correct_small_angle_by_projection(image_bgr)
        else:
            small_angle = 0.0
            corrected = image_bgr
        
        result = {
            "小角度校正": round(small_angle, 2),
            "标准方向": "0",  # 大角度由用户手动调整
            "最终校正角度": str(round(small_angle, 2)),
            "置信度": 1.0
        }
        return corrected, result
    
    def _correct_small_angle_by_projection(self, image_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        基于投影方差的小角度校正
        在-10到10度范围内搜索最佳角度，步长0.1度
        原理：文字行水平时，水平投影的方差最大
        """
        try:
            # 参数设置
            max_angle = 10.0  # 最大搜索角度
            step = 0.1  # 步长
            
            # 转换为灰度图
            if len(image_bgr.shape) == 3:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_bgr
            
            # 自适应二值化（使用高斯加权平均）
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 高斯加权平均
                cv2.THRESH_BINARY,
                blockSize=15,  # 邻域大小（奇数）
                C=10  # 常数，从均值中减去
            )
            
            # 反转图像（使文字为白色，背景为黑色）
            binary = 255 - binary
            
            h, w = binary.shape
            center = (w / 2, h / 2)
            
            # 生成所有测试角度（-10到10度，步长0.1）
            angles = np.arange(-max_angle, max_angle + step, step)
            variances = []
            
            # 对每个角度进行测试
            for angle in angles:
                # 旋转图像
                if abs(angle) < 0.01:
                    # 角度接近0，不旋转
                    rotated = binary
                else:
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(binary, M, (w, h), borderValue=0)
                
                # 水平投影：统计每行的白色像素数
                h_proj = np.sum(rotated, axis=1)
                
                # 计算方差（方差越大，说明文字行越清晰）
                variance = np.var(h_proj)
                variances.append(variance)
            
            # 找到方差最大的角度
            best_idx = np.argmax(variances)
            best_angle = angles[best_idx]
            best_variance = variances[best_idx]
            
            # 如果最佳角度接近0，说明不需要调整
            if abs(best_angle) < 0.5:
                return 0.0, image_bgr
            
            # 使用最佳角度旋转原图
            M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
            rotated = cv2.warpAffine(image_bgr, M, (w, h), borderValue=(255, 255, 255))
            
            return best_angle, rotated
        except Exception:
            return 0.0, image_bgr
    
    def _rotate_image(self, image_bgr: np.ndarray, angle: float) -> np.ndarray:
        """
        旋转图像（支持90度倍数的角度）
        使用transpose和flip，无插值损失，更高效
        
        Args:
            image_bgr: 输入图像
            angle: 当前角度（0, 90, 180, 270），函数会自动计算需要旋转的角度来转正
        """
        angle = angle % 360
        
        # 计算需要旋转的角度来转正（顺时针旋转）
        # 如果当前是270°，需要顺时针旋转90°来转正
        rotate_angle = (360 - angle) % 360
        
        if rotate_angle == 0:
            return image_bgr
        elif rotate_angle == 90:
            # 顺时针90度 = 转置 + 垂直翻转
            return cv2.flip(cv2.transpose(image_bgr), 1)
        elif rotate_angle == 180:
            # 180度 = 水平翻转 + 垂直翻转
            return cv2.flip(image_bgr, -1)
        elif rotate_angle == 270:
            # 顺时针270度 = 转置 + 水平翻转
            return cv2.flip(cv2.transpose(image_bgr), 0)
        else:
            # 非90度倍数，使用传统旋转
            (h, w) = image_bgr.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
            rotated = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
    
    def _evaluate_image_quality(self, image_bgr: np.ndarray) -> float:
        """
        评估图像质量（基于投影法）
        返回质量分数，分数越高表示图像越正
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            
            # 二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # 反转图像
            binary = 255 - binary
            
            # 水平投影
            h_proj = np.sum(binary, axis=1)
            
            # 计算投影的方差（方差越大，说明文字行越清晰）
            variance = np.var(h_proj)
            
            return variance
        except Exception:
            return 0.0


class DocumentProcessor:
    def __init__(
        self,
        layout_model_type: str = "pp_doc_layoutv3",
    ):
        cfg = Settings.from_env()
        self.layout = RapidLayout(
            engine_type=EngineType.ONNXRUNTIME,
            model_type=ModelType.PP_DOC_LAYOUTV3,
        )
        self.ocr = RapidOCR(
            params={
                "Global.use_det": False,  # 禁用检测模型
                "Global.use_cls": False,  # 禁用分类模型
                "Rec.engine_type": RapidOCREngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.CH,
                "Rec.model_type": RapidOCRModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
            }
        )
        self.table_exclude_pad_ratio = float(cfg.table_exclude_pad_ratio)
        self.table_exclude_pad_px = int(cfg.table_exclude_pad_px)
        self.pdf_dpi = int(cfg.pdf_dpi)

        self.spin = SpinCorrector()
        # 初始化方向检测器（使用 rapid-orientation）
        self.orientation_detector = RapidOrientationDetector()

    def _rotate_image(self, image_bgr: np.ndarray, angle: float) -> np.ndarray:
        """
        旋转图像（支持90度倍数的角度）
        使用transpose和flip，无插值损失，更高效
        
        Args:
            image_bgr: 输入图像
            angle: 当前角度（0, 90, 180, 270），函数会自动计算需要旋转的角度来转正
        """
        angle = angle % 360
        
        # 计算需要旋转的角度来转正（顺时针旋转）
        # 如果当前是270°，需要顺时针旋转90°来转正
        rotate_angle = (360 - angle) % 360
        
        if rotate_angle == 0:
            return image_bgr
        elif rotate_angle == 90:
            # 顺时针90度 = 转置 + 垂直翻转
            return cv2.flip(cv2.transpose(image_bgr), 1)
        elif rotate_angle == 180:
            # 180度 = 水平翻转 + 垂直翻转
            return cv2.flip(image_bgr, -1)
        elif rotate_angle == 270:
            # 顺时针270度 = 转置 + 水平翻转
            return cv2.flip(cv2.transpose(image_bgr), 0)
        else:
            # 非90度倍数，使用传统旋转
            (h, w) = image_bgr.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
            rotated = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated

    def _detect_page_orientation(self, image_bgr: np.ndarray, table_items: List[Dict]) -> Dict[str, Any]:
        """
        检测页面整体方向
        
        策略：
        1. 从每个表格中采样多个单元格（至少表格长+宽个单元格）
        2. 对这些单元格进行方向检测
        3. 取众数（最常见的角度）作为页面整体方向
        
        Args:
            image_bgr: 页面图像
            table_items: 检测到的表格列表
            
        Returns:
            方向检测结果 {"angle": int, "confidence": float}
        """
        if not table_items:
            return {"angle": 0, "confidence": 1.0}
        
        from collections import Counter
        
        # 收集所有待检测的单元格图像
        cells_to_detect = []
        cell_info = []
        
        for table_idx, table in enumerate(table_items):
            crop = table.get("crop")
            if crop is None:
                continue
                
            h, w = crop.shape[:2]
            
            # 计算需要采样的单元格数量：至少长+宽个
            num_samples = max(4, (h // 100) + (w // 100))
            
            # 将表格划分为网格采样
            rows = max(2, int(np.sqrt(num_samples * h / w)))
            cols = max(2, int(np.sqrt(num_samples * w / h)))
            
            cell_h = h // rows
            cell_w = w // cols
            
            for row in range(rows):
                for col in range(cols):
                    y1 = row * cell_h
                    y2 = (row + 1) * cell_h if row < rows - 1 else h
                    x1 = col * cell_w
                    x2 = (col + 1) * cell_w if col < cols - 1 else w
                    
                    cell = crop[y1:y2, x1:x2]
                    if cell.size > 0 and cell.shape[0] > 20 and cell.shape[1] > 20:  # 过滤太小的单元格
                        cells_to_detect.append(cell)
                        cell_info.append({"table": table_idx, "row": row, "col": col})
        
        if not cells_to_detect:
            return {"angle": 0, "confidence": 1.0}
        
        # 批量检测方向
        print(f"[DEBUG] 检测 {len(cells_to_detect)} 个单元格的方向...")
        results = self.orientation_detector.detect_batch(cells_to_detect)
        
        # 收集所有角度
        angles = [r["angle"] for r in results]
        
        # 计算众数（最常见的角度）
        angle_counts = Counter(angles)
        most_common_angle, count = angle_counts.most_common(1)[0]
        
        # 计算众数的置信度（出现频率）
        confidence = count / len(angles)
        
        # 找到该角度对应的所有结果，计算平均loss
        angle_results = [r for r in results if r["angle"] == most_common_angle]
        avg_loss = np.mean([r["loss"] for r in angle_results])
        avg_confidence = np.mean([r["confidence"] for r in angle_results])
        
        # 打印统计信息
        print(f"[DEBUG] 角度统计: {dict(angle_counts)}")
        print(f"[DEBUG] 众数角度: {most_common_angle}° (出现 {count}/{len(angles)} 次, 频率: {confidence:.2%})")
        print(f"[DEBUG] 平均loss: {avg_loss:.4f}, 平均confidence: {avg_confidence:.3f}")
        
        return {
            "angle": most_common_angle,
            "loss": avg_loss,
            "confidence": avg_confidence,
            "label": {0: 0, 90: 1, 180: 2, 270: 3}.get(most_common_angle, 0),
            "vote_count": count,
            "total_count": len(angles)
        }

    def _process_single_image(self, image_bgr: np.ndarray, page_index: int) -> Tuple[List[Region], List[Region], str]:
        """处理单张图片，只检测和识别表格"""
        from algorithms.table_recognition import build_table_grid
        import os
        
        corrected, _ = self.spin.correct(image_bgr)
        layout_items = self._detect_layout(corrected)
        table_items = [it for it in layout_items if it["label"] == "table"]

        regions: List[Region] = []
        tables: List[Region] = []
        
        for idx, it in enumerate(table_items):
            crop = it["crop"]
            
            # 使用独立的方向检测器自动检测和纠正表格方向
            rotation_applied = False
            
            try:
                # 检测表格方向
                orientation_result = self.orientation_detector.detect_orientation(crop)
                
                # 如果检测到需要旋转且置信度足够高
                if orientation_result["angle"] != 0 and orientation_result["confidence"] > 0.5:
                    # 通过坐标映射实现虚拟旋转
                    rotated_crop = self._crop_with_orientation(crop, orientation_result)
                    if rotated_crop is not None:
                        crop = rotated_crop
                        # 更新原始数据
                        it["crop"] = crop
                        rotation_applied = True
                        print(f"表格 {idx} 坐标映射旋转 {orientation_result['angle']} 度（置信度: {orientation_result['confidence']:.3f}）")
                    else:
                        print(f"表格 {idx} 坐标映射旋转失败，使用原始图像")
                else:
                    # 方向正常或置信度不足
                    print(f"表格 {idx} 方向正常（角度: {orientation_result['angle']}°, 置信度: {orientation_result['confidence']:.3f}）")
                        
            except Exception as e:
                print(f"表格方向检测失败: {e}")
            
            structure = self._extract_table_meta(crop) or {"rows": 0, "cols": 0, "cells": []}
            
            # 直接使用rec模型识别每个单元格（不使用det模型）
            table = build_table_grid(
                crop,
                structure,
                rec_ocr_fn=lambda im: self._ocr_text(im, use_det=False, use_cls=False, use_rec=True),
                min_text_height=100,
            )
            meta = structure
            meta["rows"] = table.rows
            meta["cols"] = table.cols
            meta["preview"] = {"header": table.header, "first_row": table.first_row}
            meta["grid"] = table.grid
            tr = Region(
                page_index=page_index,
                label="table",
                score=float(it.get("score", 0.0)),
                bbox=it.get("bbox", [0, 0, 0, 0]),
                text="",
                meta=meta,
                crop=crop,
            )
            regions.append(tr)
            tables.append(tr)

        regions.sort(key=lambda r: (r.page_index, r.bbox[1], r.bbox[0]))
        return regions, tables, ""

    def process_path(self, path: str, excel_output_path: Optional[str] = None, max_pages: Optional[int] = None) -> Dict[str, Any]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self.process_pdf_path(path, excel_output_path=excel_output_path, max_pages=max_pages)
        return self.process_image_path(path, excel_output_path=excel_output_path)

    def process_images(self, image_paths: List[str], excel_output_path: Optional[str] = None, page_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """处理多个图片文件（用于Qt版本，直接使用已转换的图片）
        
        新的处理流程：
        1. 小角度微调
        2. 表格结构检测
        3. 对角线单元格方向检测 -> 确定页面整体方向
        4. 页面整体旋转到正方向
        5. 重新检测表格结构
        6. 单元格处理（二值化、切割、Y轴投影）
        7. 表格旋转至正方向
        
        Args:
            image_paths: 图片路径列表
            excel_output_path: Excel输出路径
            page_indices: 对应的页面索引列表（0-based），如果为None则使用枚举索引
        """
        from algorithms.table_recognition import build_table_grid
        
        rotate_preds: List[Dict[str, Any]] = []
        table_regions: List[Region] = []
        
        for i, image_path in enumerate(image_paths):
            # 使用实际的页面索引（如果提供），否则使用枚举索引
            page_idx = page_indices[i] if page_indices and i < len(page_indices) else i
            img = self._imread_any(image_path)
            if img is None:
                continue
            
            # 步骤1: 小角度微调
            img_corrected, spin_angle = self.spin.correct(img)
            rotate_preds.append({"angle": spin_angle, "method": "spin_corrector"})
            
            # 步骤2: 表格结构检测
            layout_items = self._detect_layout(img_corrected)
            table_items = [it for it in layout_items if it["label"] == "table"]
            
            if not table_items:
                continue
            
            # 步骤3: 对角线单元格方向检测 -> 确定页面整体方向
            page_orientation = self._detect_page_orientation(img_corrected, table_items)
            
            # 步骤4: 页面整体旋转到正方向
            if page_orientation["angle"] != 0 and page_orientation["confidence"] > 0.3:
                # 计算实际旋转角度（顺时针旋转来转正）
                rotate_angle = (360 - page_orientation["angle"]) % 360
                img_rotated = self._rotate_image(img_corrected, page_orientation["angle"])
                
                # 步骤5: 重新检测表格结构（因为页面旋转了）
                layout_items = self._detect_layout(img_rotated)
                table_items = [it for it in layout_items if it["label"] == "table"]
            else:
                img_rotated = img_corrected
            
            # 步骤6 & 7: 处理每个表格
            for idx, it in enumerate(table_items):
                crop = it["crop"]
                
                # 提取表格结构
                structure = self._extract_table_meta(crop) or {"rows": 0, "cols": 0, "cells": []}
                
                # 使用rec模型识别每个单元格
                table = build_table_grid(
                    crop,
                    structure,
                    rec_ocr_fn=lambda im: self._ocr_text(im, use_det=False, use_cls=False, use_rec=True),
                    min_text_height=100,
                )
                
                meta = structure.copy()
                meta["rows"] = table.rows
                meta["cols"] = table.cols
                meta["grid"] = table.grid
                meta["header"] = table.header
                meta["first_row"] = table.first_row
                
                table_regions.append(
                    Region(
                        page_index=page_idx,
                        label="table",
                        score=it["score"],
                        bbox=it["bbox"],
                        text="",
                        page_span=None,
                        meta=meta,
                        crop=None,
                    )
                )
                print(f"[DEBUG] 表格 {idx} 处理完成")
        
        # 导出表格到Excel（在转换为字典之前导出，因为to_dict后meta会被_strip_table_grids影响）
        excel_path = self._export_tables_to_excel(table_regions, excel_output_path, base_name="output")
        
        # 转换为字典后再清理grid（避免影响返回的数据）
        import copy
        regions_dict = [copy.deepcopy(r.to_dict()) for r in table_regions]
        
        self._strip_table_grids(table_regions)

        return {
            "type": "pdf",
            "regions": regions_dict,
            "tables": regions_dict,
            "excel_path": excel_path,
            "non_table_text": "",
            "rotate_preds": rotate_preds,
        }

    def process_pdf_path(self, pdf_path: str, excel_output_path: Optional[str] = None, max_pages: Optional[int] = None) -> Dict[str, Any]:
        with open(pdf_path, "rb") as f:
            data = f.read()
        return self.process_pdf_bytes(data, filename=os.path.basename(pdf_path), excel_output_path=excel_output_path, max_pages=max_pages)

    def process_image_path(self, image_path: str, excel_output_path: Optional[str] = None) -> Dict[str, Any]:
        """处理单张图片，使用新的页面级方向检测流程"""
        result = self.process_images([image_path], excel_output_path=excel_output_path, page_indices=[0])
        return result

    def _analyze_page_content(self, page: fitz.Page) -> Tuple[bool, bool]:
        """
        分析页面内容类型
        
        Returns:
            (has_embedded_image, has_text_content)
        """
        # 检查是否有内嵌图像
        img_list = page.get_images(full=True)
        has_embedded_image = len(img_list) > 0
        
        # 检查是否有文本内容
        text = page.get_text().strip()
        has_text_content = len(text) > 10
        
        return has_embedded_image, has_text_content
    
    def _extract_embedded_images_from_page(self, page: fitz.Page) -> List[np.ndarray]:
        """从页面提取所有内嵌图像"""
        images = []
        try:
            img_list = page.get_images(full=True)
            for img in img_list:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                img_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img_array is not None:
                    images.append(img_array)
        except Exception as e:
            print(f"提取内嵌图像失败: {e}")
        return images
    
    def process_pdf_bytes(
        self, pdf_bytes: bytes, filename: str = "input.pdf", excel_output_path: Optional[str] = None, max_pages: Optional[int] = None
    ) -> Dict[str, Any]:
        """处理PDF，根据每页内容选择最佳处理方式"""
        from algorithms.table_recognition import build_table_grid
        
        rotate_preds: List[Dict[str, Any]] = []
        all_tables: List[Dict[str, Any]] = []

        print(f"\n{'='*60}")
        print(f"📄 开始处理 PDF: {filename}")
        print(f"{'='*60}")
        
        # 打开PDF进行逐页分析
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            total_pages = len(doc)
            pages_to_process = range(min(total_pages, max_pages or total_pages))
            
            for page_idx in pages_to_process:
                page = doc[page_idx]
                has_embedded_image, has_text_content = self._analyze_page_content(page)
                
                print(f"\n📄 处理第 {page_idx + 1} 页:")
                print(f"   内嵌图像: {'✓' if has_embedded_image else '✗'}")
                print(f"   文本内容: {'✓' if has_text_content else '✗'}")
                
                # 情况1: 只有内嵌图像 → 图像处理
                if has_embedded_image and not has_text_content:
                    print(f"   → 图像处理（扫描件）")
                    images = self._extract_embedded_images_from_page(page)
                    for img in images:
                        img2, pred = self.spin.correct(img)
                        rotate_preds.append(pred)
                        tables = self._process_image_for_tables(img2, page_idx)
                        all_tables.extend(tables)

                # 情况2: 只有文本 → 矢量提取
                elif not has_embedded_image and has_text_content:
                    print(f"   → 矢量提取（电子档）")
                    vector_tables = self._extract_vector_tables_from_page(page, page_idx)
                    if vector_tables:
                        all_tables.extend(vector_tables)
                        rotate_preds.append({"小角度校正": 0, "标准方向": "0", "最终校正角度": "0", "置信度": 1.0})
                    else:
                        # 矢量提取失败，回退到图像处理
                        print(f"   → 矢量提取失败，回退到图像处理")
                        pix = page.get_pixmap(dpi=self.pdf_dpi)
                        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                        if pix.n == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                        elif pix.n == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        img2, pred = self.spin.correct(img)
                        rotate_preds.append(pred)
                        tables = self._process_image_for_tables(img2, page_idx)
                        all_tables.extend(tables)

                # 情况3: 混合内容 → 分别处理
                elif has_embedded_image and has_text_content:
                    print(f"   → 混合处理（图像+矢量）")
                    # 先处理内嵌图像
                    images = self._extract_embedded_images_from_page(page)
                    for img in images:
                        img2, pred = self.spin.correct(img)
                        rotate_preds.append(pred)
                        tables = self._process_image_for_tables(img2, page_idx)
                        all_tables.extend(tables)

                    # 再处理矢量表格
                    vector_tables = self._extract_vector_tables_from_page(page, page_idx)
                    if vector_tables:
                        all_tables.extend(vector_tables)
                        rotate_preds.append({"小角度校正": 0, "标准方向": "0", "最终校正角度": "0", "置信度": 1.0})
                
                # 情况4: 什么都没有
                else:
                    print(f"   → 跳过（无内容）")

        # 重建Region对象
        table_regions = []
        for t in all_tables:
            table_regions.append(Region(
                page_index=t.get("page_index", 0),
                label=t.get("label", "table"),
                score=t.get("score", 0.0),
                bbox=t.get("bbox", [0, 0, 0, 0]),
                text=t.get("text", ""),
                meta=t.get("meta", {}),
                crop=None,
            ))

        excel_path = self._export_tables_to_excel(table_regions, excel_output_path, base_name=os.path.splitext(filename)[0])
        self._strip_table_grids(table_regions)
        
        return {
            "type": "pdf",
            "filename": filename,
            "rotate": rotate_preds,
            "regions": [r.to_dict() for r in table_regions],
            "tables": [r.to_dict() for r in table_regions],
            "excel_path": excel_path,
            "non_table_text": "",
        }
    
    def _process_image_for_tables(self, img: np.ndarray, page_idx: int) -> List[Dict]:
        """处理图像提取表格"""
        from algorithms.table_recognition import build_table_grid
        tables = []
        layout_items = self._detect_layout(img)
        table_items = [it for it in layout_items if it["label"] == "table"]

        for idx, it in enumerate(table_items):
            crop = it["crop"]
            structure = self._extract_table_meta(crop) or {"rows": 0, "cols": 0, "cells": []}
            table = build_table_grid(
                crop,
                structure,
                rec_ocr_fn=lambda im: self._ocr_text(im, use_det=False, use_cls=False, use_rec=True),
                min_text_height=100,
            )
            meta = structure.copy()
            meta["rows"] = table.rows
            meta["cols"] = table.cols
            meta["preview"] = {"header": table.header, "first_row": table.first_row}
            meta["grid"] = table.grid
            # 更新 cells 信息，包含正确的 rowspan 和 colspan
            meta["cells"] = [
                {
                    "row": c.row,
                    "col": c.col,
                    "rowspan": c.rowspan,
                    "colspan": c.colspan,
                    "text": c.text,
                    "bbox": c.bbox
                }
                for c in table.cells
            ]
            table_region = Region(
                page_index=page_idx,
                label="table",
                score=float(it.get("score", 0.0)),
                bbox=it.get("bbox", [0, 0, 0, 0]),
                text="",
                meta=meta,
                crop=None,
            )
            tables.append(table_region.to_dict())
        return tables

    def _extract_vector_tables_from_page(self, page: fitz.Page, page_idx: int) -> List[Dict]:
        """从页面提取矢量表格"""
        from core.pdf_vector_table import PDFVectorTableExtractor
        tables = []
        extractor = PDFVectorTableExtractor()
        table = extractor.extract_table_from_page(page, debug=False)
        
        if table and table.cells:
            # 检查是否有文本内容
            total_text = sum(len(c.text.strip()) for c in table.cells)
            if total_text > 0:
                table_region = Region(
                    page_index=page_idx,
                    label="table",
                    score=1.0,
                    bbox=[
                        min(c.x1 for c in table.cells),
                        min(c.y1 for c in table.cells),
                        max(c.x2 for c in table.cells),
                        max(c.y2 for c in table.cells)
                    ],
                    text="",
                    meta={
                        "rows": table.rows,
                        "cols": table.cols,
                        "grid": table.to_grid(),
                        "cells": [
                            {
                                "row": c.row,
                                "col": c.col,
                                "rowspan": c.rowspan,
                                "colspan": c.colspan,
                                "text": c.text,
                                "bbox": [c.x1, c.y1, c.x2, c.y2]
                            }
                            for c in table.cells
                        ]
                    },
                    crop=None,
                )
                tables.append(table_region.to_dict())
        return tables
    
    def _try_extract_vector_tables(self, pdf_bytes: bytes, max_pages: Optional[int] = None) -> Dict[int, Any]:
        """
        尝试从 PDF 中提取矢量表格
        
        Args:
            pdf_bytes: PDF 文件字节
            max_pages: 最大处理页数
            
        Returns:
            页面索引到表格的映射，如果没有提取到则返回空字典
        """
        print(f"\n🔄 尝试矢量表格提取...")
        try:
            import tempfile
            
            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            
            try:
                # 尝试矢量提取
                tables = extract_tables_from_pdf(tmp_path, max_pages=max_pages)
                
                if not tables:
                    print(f"⚠️ 矢量提取未找到表格，将使用图像处理")
                    return {}
                
                # 检查是否真正提取到了文本内容（不只是表格结构）
                total_text_chars = 0
                for page_idx, table in tables.items():
                    for cell in table.cells:
                        total_text_chars += len(cell.text.strip())
                
                # 如果表格单元格中有文本内容，认为是电子档
                if total_text_chars > 0:
                    print(f"✅ 矢量提取成功，共 {len(tables)} 页包含表格，总字符数: {total_text_chars}")
                    return tables
                else:
                    # 有表格结构但没有文本，可能是扫描件（图片中的表格）
                    print(f"⚠️ 检测到表格结构但没有文本内容，可能是扫描件，切换到图像处理")
                    return {}
                    
            finally:
                # 清理临时文件
                os.unlink(tmp_path)
                
        except Exception as e:
            print(f"❌ 矢量提取失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def process_image_bytes(
        self, image_bytes: bytes, filename: str = "input.png", excel_output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """处理图片字节，使用新的页面级方向检测流程"""
        # 保存临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        try:
            result = self.process_images([tmp_path], excel_output_path=excel_output_path, page_indices=[0])
            result["filename"] = filename
            return result
        finally:
            # 清理临时文件
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _detect_layout(self, image_bgr: np.ndarray, padding: int = 20) -> List[Dict[str, Any]]:
        layout_res = self.layout(image_bgr)
        items: List[Dict[str, Any]] = []
        if not layout_res.boxes:
            return items
        h, w = image_bgr.shape[:2]
        for bbox, label, score in zip(layout_res.boxes, layout_res.class_names, layout_res.scores):
            x1, y1, x2, y2 = self._clamp_box(bbox, w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            # 添加留白（padding）
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(w, x2 + padding)
            y2_padded = min(h, y2 + padding)
            crop = image_bgr[y1_padded:y2_padded, x1_padded:x2_padded]
            items.append(
                {
                    "label": str(label),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(score) if score is not None else 0.0,
                    "crop": crop,
                    "crop_offset": [int(x1_padded), int(y1_padded)],  # padding偏移量
                }
            )
        items.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))
        return items

    def _extract_table_meta(self, table_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        try:
            result = parse_table_structure(table_bgr)
            return result
        except Exception:
            return None

    def _export_tables_to_excel(self, tables: List[Region], excel_output_path: Optional[str], base_name: str) -> Optional[str]:
        if not excel_output_path or not tables:
            return None
        
        # 按页面索引和位置排序，确保表格按页面顺序合并
        tables = sorted(tables, key=lambda r: (r.page_index if r.page_index is not None else 0, r.bbox[1] if r.bbox else 0, r.bbox[0] if r.bbox else 0))
        
        resolved = os.path.abspath(excel_output_path)
        if resolved.lower().endswith(".xlsx"):
            out_path = resolved
        else:
            Path(resolved).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(resolved, f"{base_name}.xlsx")

        out_dir = os.path.dirname(out_path) or os.getcwd()
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        sheet_tables: List[Tuple[str, TableResult]] = []
        last_header_key: Optional[Tuple[str, ...]] = None
        last_table_structure: Optional[Tuple[int, int]] = None  # (rows, cols)
        
        for r in tables:
            if not r.meta or not isinstance(r.meta, dict):
                r.meta = {"rows": 0, "cols": 0, "cells": []}
            grid = r.meta.get("grid") if isinstance(r.meta.get("grid"), list) else None
            if not grid or not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
                continue
            header = [str(v or "") for v in grid[0]]
            first_row = [str(v or "") for v in grid[1]] if len(grid) > 1 and isinstance(grid[1], list) else []
            table = TableResult(rows=len(grid), cols=len(grid[0]), cells=[], grid=grid, header=header, first_row=first_row)
            header_key = self._header_key(table.header)
            current_structure = (table.rows, table.cols)
            
            span = ""
            if isinstance(r.page_span, list) and len(r.page_span) == 2:
                span = f"p{r.page_span[0]}-{r.page_span[1]}_"
            
            # 判断是否应该合并
            should_merge = False
            merge_mode = "none"  # "header" - 表头相同, "continuous" - 连续数据
            
            if sheet_tables:
                # 情况 1：表头完全相同（当前页也有表头）
                if header_key is not None and header_key == last_header_key:
                    should_merge = True
                    merge_mode = "header"
                # 情况 2：表头不同，但列数相同，检查是否有连续关系
                elif last_table_structure is not None:
                    prev_rows, prev_cols = last_table_structure
                    curr_rows, curr_cols = current_structure
                    # 列数必须相同
                    if prev_cols == curr_cols:
                        # 检查上一页最后一行和当前页第一行是否有连续关系
                        prev_name, prev_table = sheet_tables[-1]
                        continuity = self._check_row_continuity(prev_table, table)
                        if continuity["is_continuous"]:
                            should_merge = True
                            merge_mode = "continuous"
            
            if should_merge and sheet_tables:
                prev_name, prev_table = sheet_tables[-1]
                merged_grid = list(prev_table.grid)
                
                if merge_mode == "header":
                    # 表头相同，跳过当前表格的表头行
                    if len(table.grid) >= 2:
                        merged_grid.extend(table.grid[1:])
                else:
                    # 结构相似，保留所有行（当前页没有表头）
                    merged_grid.extend(table.grid)
                
                merged_table = TableResult(
                    rows=len(merged_grid),
                    cols=prev_table.cols,
                    cells=[],
                    grid=merged_grid,
                    header=prev_table.header,
                    first_row=prev_table.first_row,
                )
                sheet_tables[-1] = (prev_name, merged_table)
                r.meta["merged_sheet"] = prev_name
            else:
                name = f"{span}table_{len(sheet_tables) + 1}"
                sheet_tables.append((name, table))
                last_header_key = header_key
                last_table_structure = current_structure
                r.meta["merged_sheet"] = name
            
            r.meta["rows"] = table.rows
            r.meta["cols"] = table.cols
            r.meta["preview"] = {"header": table.header, "first_row": table.first_row}

        if not sheet_tables:
            return None
        save_tables_to_single_workbook(sheet_tables, out_path)
        for r in tables:
            if r.meta is None:
                r.meta = {}
            r.meta["excel_path"] = out_path
        return out_path

    def _header_key(self, header: Any) -> Optional[Tuple[str, ...]]:
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

    def _check_row_continuity(self, prev_table: TableResult, curr_table: TableResult) -> Dict[str, Any]:
        """
        检查上一页最后一行和当前页第一行是否有连续关系
        返回: {"is_continuous": bool, "type": str}
        """
        result = {"is_continuous": False, "type": "none"}
        
        if not prev_table.grid or not curr_table.grid:
            return result
        
        # 获取上一页的最后一行数据
        prev_last_row = prev_table.grid[-1] if prev_table.grid else None
        # 获取当前页的第一行
        curr_first_row = curr_table.grid[0] if curr_table.grid else None
        
        if not prev_last_row or not curr_first_row:
            return result
        
        # 检查每一列是否有连续关系
        continuous_cols = []
        
        for prev_val, curr_val in zip(prev_last_row, curr_first_row):
            prev_str = str(prev_val or "").strip()
            curr_str = str(curr_val or "").strip()
            
            # 检查1：完全相同
            if prev_str == curr_str and prev_str:
                continuous_cols.append("same")
            # 检查2：连续数字（如 1->2, 10->11）
            elif self._is_continuous_number(prev_str, curr_str):
                continuous_cols.append("number_seq")
            # 检查3：连续字母（如 A->B, AA->AB）
            elif self._is_continuous_alpha(prev_str, curr_str):
                continuous_cols.append("alpha_seq")
            else:
                continuous_cols.append("none")
        
        # 统计连续关系的列数
        same_count = continuous_cols.count("same")
        num_seq_count = continuous_cols.count("number_seq")
        alpha_seq_count = continuous_cols.count("alpha_seq")
        total_continuous = same_count + num_seq_count + alpha_seq_count
        
        # 如果有至少一列有连续关系，认为是连续表格
        if total_continuous >= 1:
            result["is_continuous"] = True
            if num_seq_count >= 1:
                result["type"] = "数字连续"
            elif alpha_seq_count >= 1:
                result["type"] = "字母连续"
            elif same_count >= 1:
                result["type"] = "内容相同"
        
        return result

    def _is_continuous_number(self, prev: str, curr: str) -> bool:
        """检查两个字符串是否是连续的数字"""
        # 提取数字部分
        prev_nums = re.findall(r'\d+', prev)
        curr_nums = re.findall(r'\d+', curr)
        
        if not prev_nums or not curr_nums:
            return False
        
        # 比较最后一组数字
        try:
            prev_num = int(prev_nums[-1])
            curr_num = int(curr_nums[-1])
            # 检查是否是连续的数字（差值为1）
            if curr_num == prev_num + 1:
                # 检查前缀是否相同
                prev_prefix = prev[:prev.rfind(prev_nums[-1])]
                curr_prefix = curr[:curr.rfind(curr_nums[-1])]
                return prev_prefix == curr_prefix
        except ValueError:
            pass
        
        return False

    def _is_continuous_alpha(self, prev: str, curr: str) -> bool:
        """检查两个字符串是否是连续的字母（如 A->B, AA->AB）"""
        # 提取字母部分
        prev_alpha = re.findall(r'[a-zA-Z]+', prev)
        curr_alpha = re.findall(r'[a-zA-Z]+', curr)
        
        if not prev_alpha or not curr_alpha:
            return False
        
        # 比较最后一组字母
        prev_str = prev_alpha[-1]
        curr_str = curr_alpha[-1]
        
        # 简单的字母递增检查
        if len(prev_str) == len(curr_str):
            # 从右向左比较
            for i in range(len(prev_str) - 1, -1, -1):
                p_char = prev_str[i].upper()
                c_char = curr_str[i].upper()
                
                if ord(c_char) == ord(p_char) + 1:
                    # 检查左边部分是否相同
                    if i == 0 or prev_str[:i] == curr_str[:i]:
                        return True
                    break
                elif c_char != p_char:
                    break
        
        return False

    def _strip_table_grids(self, tables: List[Region]) -> None:
        for r in tables:
            if not r.meta or not isinstance(r.meta, dict):
                continue
            if "grid" in r.meta:
                try:
                    del r.meta["grid"]
                except Exception:
                    pass

    def _ocr_output(
        self,
        image_bgr: np.ndarray,
        use_det: Optional[bool] = None,
        use_cls: Optional[bool] = None,
        use_rec: Optional[bool] = None,
    ) -> Any:
        return self.ocr(image_bgr, use_det=use_det, use_cls=use_cls, use_rec=use_rec)

    def _ocr_text(
        self,
        image_bgr: np.ndarray,
        use_det: Optional[bool] = None,
        use_cls: Optional[bool] = None,
        use_rec: Optional[bool] = None,
    ) -> str:
        try:
            res = self._ocr_output(image_bgr, use_det=use_det, use_cls=use_cls, use_rec=use_rec)
        except Exception:
            tmp_dir = os.path.join(os.getcwd(), "output", "tmp_ocr")
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.png")
            cv2.imencode(".png", image_bgr)[1].tofile(tmp_path)
            res = self._ocr_output(tmp_path, use_det=use_det, use_cls=use_cls, use_rec=use_rec)

        if hasattr(res, "txts") and isinstance(res.txts, (list, tuple)):
            parts = [str(t).strip() for t in list(res.txts) if str(t).strip()]
            return "\n".join(parts)
        return ""

    def _extract_text_outside_tables(
        self, ocr_out: Any, hw: Tuple[int, int], table_bboxes: List[List[float]]
    ) -> str:
        if not hasattr(ocr_out, "txts") or not hasattr(ocr_out, "boxes") or ocr_out.txts is None or ocr_out.boxes is None:
            return ""
        txts = list(ocr_out.txts)
        boxes = np.array(ocr_out.boxes)

        h, w = hw
        padded = [self._pad_bbox(b, w=w, h=h) for b in table_bboxes]
        items = []
        for text, box in zip(txts, boxes):
            t = str(text).strip()
            if not t:
                continue
            pts = np.array(box, dtype=np.float32).reshape(-1, 2)
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            if self._point_in_any_table(cx, cy, padded):
                continue
            items.append((cy, cx, t))
        items.sort(key=lambda x: (x[0], x[1]))
        return "\n".join([t for _, _, t in items])

    def _point_in_any_table(self, x: float, y: float, table_bboxes: List[List[float]]) -> bool:
        for b in table_bboxes:
            try:
                x1, y1, x2, y2 = b
                if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                    return True
            except Exception:
                continue
        return False

    def _pad_bbox(self, bbox: List[float], w: int, h: int) -> List[float]:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            return bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad_x = max(float(self.table_exclude_pad_px), bw * float(self.table_exclude_pad_ratio))
        pad_y = max(float(self.table_exclude_pad_px), bh * float(self.table_exclude_pad_ratio))
        x1 = max(0.0, x1 - pad_x)
        y1 = max(0.0, y1 - pad_y)
        x2 = min(float(w), x2 + pad_x)
        y2 = min(float(h), y2 + pad_y)
        return [x1, y1, x2, y2]

    def _pdf_page_generator(self, pdf_bytes: bytes, dpi: int = 150):
        """
        按需生成PDF页面的生成器，节省内存
        每次只加载一个页面到内存，处理完后立即释放
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                yield img
                # 释放当前页面的资源
                pix = None
                page = None
        finally:
            doc.close()

    def _imread_any(self, path: str) -> np.ndarray:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        return img

    def _clamp_box(self, bbox: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))
        return x1, y1, x2, y2

    def _x_overlap_ratio(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        ax1, ax2 = a
        bx1, bx2 = b
        inter = max(0, min(ax2, bx2) - max(ax1, bx1))
        denom = max(1, min(ax2 - ax1, bx2 - bx1))
        return inter / denom

    def _vconcat_with_gap(self, imgs: List[np.ndarray], gap: int = 10) -> np.ndarray:
        if not imgs:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        widths = [im.shape[1] for im in imgs]
        target_w = max(widths)
        padded = []
        for im in imgs:
            if im.shape[1] == target_w:
                padded.append(im)
            else:
                pad = target_w - im.shape[1]
                padded.append(cv2.copyMakeBorder(im, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        if gap > 0:
            spacer = np.full((gap, target_w, 3), 255, dtype=np.uint8)
            out = padded[0]
            for im in padded[1:]:
                out = np.vstack([out, spacer, im])
            return out
        return np.vstack(padded)
