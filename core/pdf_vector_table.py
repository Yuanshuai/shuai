"""
PDF 矢量表格提取模块
从 PDF 页面中提取矢量图形和文本，构建表格结构
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import fitz
import numpy as np


@dataclass
class Point:
    """点"""
    x: float
    y: float
    
    def __hash__(self):
        return hash((round(self.x, 2), round(self.y, 2)))
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return abs(self.x - other.x) < 0.1 and abs(self.y - other.y) < 0.1
        return False


@dataclass
class Line:
    """线段"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def is_horizontal(self) -> bool:
        """是否为水平线"""
        return abs(self.y2 - self.y1) < 1.0
    
    @property
    def is_vertical(self) -> bool:
        """是否为垂直线"""
        return abs(self.x2 - self.x1) < 1.0
    
    @property
    def length(self) -> float:
        """线段长度"""
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)


@dataclass
class Cell:
    """表格单元格"""
    row: int
    col: int
    x1: float
    y1: float
    x2: float
    y2: float
    text: str = ""
    rowspan: int = 1
    colspan: int = 1


@dataclass
class Table:
    """表格对象"""
    cells: List[Cell] = field(default_factory=list)
    rows: int = 0
    cols: int = 0
    
    def to_grid(self) -> List[List[str]]:
        """转换为二维网格"""
        if not self.cells:
            return []
        
        # 创建空网格
        grid = [["" for _ in range(self.cols)] for _ in range(self.rows)]
        
        # 填充文本
        for cell in self.cells:
            if 0 <= cell.row < self.rows and 0 <= cell.col < self.cols:
                grid[cell.row][cell.col] = cell.text
        
        return grid


class PDFVectorTableExtractor:
    """PDF 矢量表格提取器"""
    
    def __init__(self, tolerance: float = 2.0, intersection_tolerance: float = 3.0):
        """
        Args:
            tolerance: 线条过滤容差（像素）
            intersection_tolerance: 交点检测容差（像素）
        """
        self.tolerance = tolerance
        self.intersection_tolerance = intersection_tolerance
    
    def extract_table_from_page(self, page: fitz.Page, debug: bool = True) -> Optional[Table]:
        """
        从 PDF 页面中提取表格
        
        Args:
            page: fitz.Page 对象
            debug: 是否输出调试信息
            
        Returns:
            Table 对象，如果没有检测到表格则返回 None
        """
        if debug:
            print(f"\n{'='*60}")
            print(f"🔍 开始矢量表格提取")
            print(f"{'='*60}")
        
        # 1. 提取可绘制对象
        drawings = page.get_drawings()
        if debug:
            print(f"📊 提取到 {len(drawings)} 个绘制对象")
        
        # 2. 提取文本（使用 rawdict 获取字符级信息）
        text_dict = page.get_text("rawdict")
        text_blocks = len(text_dict.get("blocks", []))
        if debug:
            print(f"📝 提取到 {text_blocks} 个文本块（字符级）")
        
        # 3. 解析线条
        lines = self._extract_lines(drawings)
        if debug:
            print(f"📏 解析到 {len(lines)} 条线段")
        
        if not lines:
            if debug:
                print(f"❌ 未检测到线条，放弃矢量提取")
            return None
        
        # 4. 过滤线条（保留水平线和垂直线）
        h_lines, v_lines = self._filter_lines(lines)
        if debug:
            print(f"➡️  水平线: {len(h_lines)} 条")
            print(f"⬇️  垂直线: {len(v_lines)} 条")
            for i, h in enumerate(h_lines[:5]):
                print(f"    水平线 {i+1}: ({h.x1:.1f}, {h.y1:.1f}) -> ({h.x2:.1f}, {h.y2:.1f}), 长度={h.length:.1f}")
            for i, v in enumerate(v_lines[:5]):
                print(f"    垂直线 {i+1}: ({v.x1:.1f}, {v.y1:.1f}) -> ({v.x2:.1f}, {v.y2:.1f}), 长度={v.length:.1f}")
        
        if not h_lines or not v_lines:
            if debug:
                print(f"❌ 缺少水平线或垂直线，无法构建表格")
            return None
        
        # 5. 找交点
        intersections = self._find_intersections(h_lines, v_lines)
        if debug:
            print(f"✂️  找到 {len(intersections)} 个交点")
            for i, p in enumerate(list(intersections)[:10]):
                print(f"    交点 {i+1}: ({p.x:.1f}, {p.y:.1f})")
        
        if len(intersections) < 4:
            if debug:
                print(f"❌ 交点数量不足（{len(intersections)} < 4），无法构建单元格")
            return None
        
        # 6. 构建单元格
        cells = self._build_cells(intersections, h_lines, v_lines)
        if debug:
            print(f"🔲 构建到 {len(cells)} 个单元格")
            for i, c in enumerate(cells[:10]):
                print(f"    单元格 {i+1}: 行{c.row} 列{c.col} 位置({c.x1:.1f}, {c.y1:.1f}) -> ({c.x2:.1f}, {c.y2:.1f})")
        
        if not cells:
            if debug:
                print(f"❌ 未能构建单元格")
            return None
        
        # 7. 分配文本到单元格
        self._assign_text_to_cells(cells, text_dict, debug)
        
        # 8. 确定表格维度
        rows = max(c.row for c in cells) + 1 if cells else 0
        cols = max(c.col for c in cells) + 1 if cells else 0
        
        if debug:
            print(f"📐 表格维度: {rows} 行 x {cols} 列")
            print(f"✅ 矢量表格提取成功")
            print(f"{'='*60}\n")
        
        return Table(cells=cells, rows=rows, cols=cols)
    
    def _extract_lines(self, drawings: List[Dict]) -> List[Line]:
        """
        从绘制对象中提取线段
        
        Args:
            drawings: 绘制对象列表
            
        Returns:
            线段列表
        """
        lines = []
        
        for drawing in drawings:
            items = drawing.get("items", [])
            
            for item in items:
                if item[0] == "l":  # 线段
                    # item = ("l", p1, p2)
                    p1, p2 = item[1], item[2]
                    lines.append(Line(p1.x, p1.y, p2.x, p2.y))
                    
                elif item[0] == "re":  # 矩形
                    # item = ("re", rect)
                    rect = item[1]
                    x1, y1, x2, y2 = rect.x0, rect.y0, rect.x1, rect.y1
                    # 矩形的四条边
                    lines.append(Line(x1, y1, x2, y1))  # 上边
                    lines.append(Line(x2, y1, x2, y2))  # 右边
                    lines.append(Line(x2, y2, x1, y2))  # 下边
                    lines.append(Line(x1, y2, x1, y1))  # 左边
        
        return lines
    
    def _filter_lines(self, lines: List[Line]) -> Tuple[List[Line], List[Line]]:
        """
        过滤线条，保留水平线和垂直线
        
        Args:
            lines: 线段列表
            
        Returns:
            (水平线列表, 垂直线列表)
        """
        h_lines = []
        v_lines = []
        
        for line in lines:
            if line.is_horizontal and line.length > 5:  # 过滤太短的线
                h_lines.append(line)
            elif line.is_vertical and line.length > 5:
                v_lines.append(line)
        
        # 合并相近的线条
        h_lines = self._merge_horizontal_lines(h_lines)
        v_lines = self._merge_vertical_lines(v_lines)
        
        return h_lines, v_lines
    
    def _merge_horizontal_lines(self, lines: List[Line]) -> List[Line]:
        """合并相近的水平线"""
        if not lines:
            return []
        
        # 按 y 坐标分组
        lines_by_y: Dict[int, List[Line]] = {}
        for line in lines:
            y_key = int(round((line.y1 + line.y2) / 2))
            if y_key not in lines_by_y:
                lines_by_y[y_key] = []
            lines_by_y[y_key].append(line)
        
        merged = []
        for y, group in lines_by_y.items():
            # 按 x 排序并合并重叠的线段
            group.sort(key=lambda l: l.x1)
            
            current = group[0]
            for line in group[1:]:
                if line.x1 <= current.x2 + self.tolerance:  # 重叠或接近
                    current = Line(current.x1, current.y1, max(current.x2, line.x2), current.y1)
                else:
                    merged.append(current)
                    current = line
            merged.append(current)
        
        return merged
    
    def _merge_vertical_lines(self, lines: List[Line]) -> List[Line]:
        """合并相近的垂直线"""
        if not lines:
            return []
        
        # 按 x 坐标分组
        lines_by_x: Dict[int, List[Line]] = {}
        for line in lines:
            x_key = int(round((line.x1 + line.x2) / 2))
            if x_key not in lines_by_x:
                lines_by_x[x_key] = []
            lines_by_x[x_key].append(line)
        
        merged = []
        for x, group in lines_by_x.items():
            # 按 y 排序并合并重叠的线段
            group.sort(key=lambda l: l.y1)
            
            current = group[0]
            for line in group[1:]:
                if line.y1 <= current.y2 + self.tolerance:  # 重叠或接近
                    current = Line(current.x1, current.y1, current.x1, max(current.y2, line.y2))
                else:
                    merged.append(current)
                    current = line
            merged.append(current)
        
        return merged
    
    def _find_intersections(self, h_lines: List[Line], v_lines: List[Line]) -> Set[Point]:
        """
        找水平线与垂直线的交点
        
        Args:
            h_lines: 水平线列表
            v_lines: 垂直线列表
            
        Returns:
            交点集合
        """
        intersections = set()
        
        for h in h_lines:
            for v in v_lines:
                # 检查垂直线是否穿过水平线
                if (v.y1 - self.intersection_tolerance <= h.y1 <= v.y2 + self.intersection_tolerance and
                    h.x1 - self.intersection_tolerance <= v.x1 <= h.x2 + self.intersection_tolerance):
                    intersections.add(Point(v.x1, h.y1))
        
        return intersections
    
    def _build_cells(self, intersections: Set[Point], h_lines: List[Line], v_lines: List[Line]) -> List[Cell]:
        """
        从交点构建单元格
        
        Args:
            intersections: 交点集合
            h_lines: 水平线列表
            v_lines: 垂直线列表
            
        Returns:
            单元格列表
        """
        if len(intersections) < 4:
            return []
        
        # 获取唯一的 x 和 y 坐标
        x_coords = sorted(set(p.x for p in intersections))
        y_coords = sorted(set(p.y for p in intersections))
        
        if len(x_coords) < 2 or len(y_coords) < 2:
            return []
        
        cells = []
        
        # 构建单元格网格
        for row in range(len(y_coords) - 1):
            for col in range(len(x_coords) - 1):
                x1, x2 = x_coords[col], x_coords[col + 1]
                y1, y2 = y_coords[row], y_coords[row + 1]
                
                # 检查四个角是否都存在
                corners = [
                    Point(x1, y1) in intersections,
                    Point(x2, y1) in intersections,
                    Point(x1, y2) in intersections,
                    Point(x2, y2) in intersections
                ]
                
                if all(corners):
                    cells.append(Cell(
                        row=row,
                        col=col,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2
                    ))
        
        return cells
    
    def _assign_text_to_cells(self, cells: List[Cell], text_dict: Dict, debug: bool = False):
        """
        将文本分配到对应的单元格
        
        Args:
            cells: 单元格列表
            text_dict: 文本字典（来自 page.get_text("rawdict")）
            debug: 是否输出调试信息
        """
        assigned_count = 0
        total_chars = 0
        
        # 收集所有字符用于调试
        all_chars = []
        
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # 跳过非文本块
                continue
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    # 使用 rawdict，span 中包含 chars（字符列表）
                    chars = span.get("chars", [])
                    if not chars:
                        # 兼容模式：如果没有 chars，使用 span 级别的 text
                        text = span.get("text", "").strip()
                        if text:
                            bbox = span.get("bbox")
                            if bbox:
                                chars = [{'c': text, 'bbox': bbox}]
                    
                    for char_info in chars:
                        char = char_info.get('c', '')
                        if not char or char.strip() == '':
                            continue
                        
                        total_chars += 1
                        
                        # 获取字符的 bbox
                        bbox = char_info.get('bbox')
                        if not bbox:
                            continue
                        
                        char_x = (bbox[0] + bbox[2]) / 2  # 字符中心 x
                        char_y = (bbox[1] + bbox[3]) / 2  # 字符中心 y
                        
                        # 找到包含该字符的单元格
                        assigned_cell = None
                        for cell in cells:
                            if (cell.x1 <= char_x <= cell.x2 and 
                                cell.y1 <= char_y <= cell.y2):
                                # 直接拼接字符
                                if cell.text:
                                    cell.text += char
                                else:
                                    cell.text = char
                                assigned_count += 1
                                assigned_cell = cell
                                break
                        
                        # 记录字符信息用于调试
                        all_chars.append({
                            'char': char,
                            'bbox': bbox,
                            'cell': assigned_cell
                        })
        
        # 调试输出：显示所有字符（只显示前30个）
        if debug and all_chars:
            print(f"\n🔤 字符级详情（共 {len(all_chars)} 个字符，显示前30个）:")
            current_word = ""
            current_cell = None
            for i, char_info in enumerate(all_chars[:30]):
                char = char_info['char']
                cell = char_info['cell']
                
                # 按单元格分组显示
                if cell != current_cell:
                    if current_word and current_cell:
                        print(f"    '{current_word}' →[{current_cell.row},{current_cell.col}]")
                    current_word = char
                    current_cell = cell
                else:
                    current_word += char
            
            if current_word and current_cell:
                print(f"    '{current_word}' →[{current_cell.row},{current_cell.col}]")
        
        if debug:
            print(f"📄 文本分配: {assigned_count}/{total_chars} 个字符被分配到单元格")
            non_empty_cells = sum(1 for c in cells if c.text)
            print(f"🎯 有文本的单元格: {non_empty_cells}/{len(cells)}")
            # 显示前几个单元格的内容
            if non_empty_cells > 0:
                print(f"📋 部分单元格内容预览:")
                for i, cell in enumerate(cells[:15]):
                    if cell.text:
                        print(f"    [{cell.row},{cell.col}]: '{cell.text[:50]}'")


def extract_tables_from_pdf(pdf_path: str, page_indices: Optional[List[int]] = None, max_pages: Optional[int] = None) -> Dict[int, Table]:
    """
    从 PDF 中提取表格
    
    Args:
        pdf_path: PDF 文件路径
        page_indices: 要处理的页面索引列表，None 表示所有页面
        max_pages: 最大处理页数，None 表示不限制
        
    Returns:
        页面索引到 Table 的映射
    """
    tables = {}
    extractor = PDFVectorTableExtractor()
    
    with fitz.open(pdf_path) as doc:
        if page_indices is None:
            page_indices = range(len(doc))
        
        for i, page_idx in enumerate(page_indices):
            if max_pages is not None and i >= max_pages:
                break
            
            if page_idx < 0 or page_idx >= len(doc):
                continue
            
            page = doc[page_idx]
            table = extractor.extract_table_from_page(page)
            
            if table and table.cells:
                tables[page_idx] = table
    
    return tables
