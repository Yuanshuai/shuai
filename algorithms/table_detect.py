"""表格检测底层模块，负责从图像中定位表格与单元格结构。"""

import cv2
import numpy as np


class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __str__(self):
        return f"BBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"


def get_table_structure(tables):
    """获取每个单元格的行列信息，处理跨行跨列（支持合并单元格）"""
    if not tables:
        return []

    x_coords = sorted(list(set([t[0] for t in tables] + [t[0] + t[2] for t in tables])))
    y_coords = sorted(list(set([t[1] for t in tables] + [t[1] + t[3] for t in tables])))

    # 动态计算合并阈值：根据表格框的尺寸调整
    # 估算图像尺寸（从表格坐标推断）
    if tables:
        max_x = max(t[0] + t[2] for t in tables)
        max_y = max(t[1] + t[3] for t in tables)
        min_dim = min(max_x, max_y)
        # 小图像：threshold ≈ 20，大图像：threshold ≈ 40
        # 增加阈值以更好地合并表格的上下边界线（表格行高通常在10-20像素）
        dynamic_threshold = max(20, min(40, min_dim // 60))
    else:
        dynamic_threshold = 20  # 默认值

    def merge_close_coords(coords, threshold=None):
        if threshold is None:
            threshold = dynamic_threshold
        if not coords:
            return []
        merged = [coords[0]]
        for coord in coords[1:]:
            # 检查与最后一个合并坐标的距离（只与最后一个比较，保持顺序）
            if coord - merged[-1] > threshold:
                merged.append(coord)
        return merged

    x_coords_merged = merge_close_coords(x_coords)
    y_coords_merged = merge_close_coords(y_coords)

    total_rows = len(y_coords_merged) - 1
    total_cols = len(x_coords_merged) - 1

    # 检查是否有有效的行列数
    if total_rows <= 0 or total_cols <= 0:
        return []

    # 第一步：为每个检测到的框确定起始行列位置
    cell_candidates = []
    for table in tables:
        x, y, w, h = table
        x2, y2 = x + w, y + h

        # 找到起始行（框的顶部最接近哪条横线）
        start_row = -1
        min_y_diff = float('inf')
        for i in range(len(y_coords_merged) - 1):
            y_line = y_coords_merged[i]
            y_diff = abs(y - y_line)
            if y_diff < min_y_diff:
                min_y_diff = y_diff
                start_row = i

        # 找到起始列（框的左边最接近哪条竖线）
        start_col = -1
        min_x_diff = float('inf')
        for i in range(len(x_coords_merged) - 1):
            x_line = x_coords_merged[i]
            x_diff = abs(x - x_line)
            if x_diff < min_x_diff:
                min_x_diff = x_diff
                start_col = i

        # 找到结束行（框的底部最接近哪条横线）
        end_row = -1
        min_y_diff = float('inf')
        for i in range(len(y_coords_merged)):  # 从0开始，检查所有线
            y_line = y_coords_merged[i]
            y_diff = abs(y2 - y_line)
            if y_diff < min_y_diff:
                min_y_diff = y_diff
                end_row = max(0, i - 1)  # 转换为行索引，确保不小于0

        # 找到结束列（框的右边最接近哪条竖线）
        end_col = -1
        min_x_diff = float('inf')
        for i in range(len(x_coords_merged)):  # 从0开始，检查所有线
            x_line = x_coords_merged[i]
            x_diff = abs(x2 - x_line)
            if x_diff < min_x_diff:
                min_x_diff = x_diff
                end_col = max(0, i - 1)  # 转换为列索引，确保不小于0

        if start_row >= 0 and start_col >= 0 and end_row >= start_row and end_col >= start_col:
            rowspan = end_row - start_row + 1
            colspan = end_col - start_col + 1
            cell_candidates.append({
                'row': start_row,
                'col': start_col,
                'rowspan': rowspan,
                'colspan': colspan,
                'bbox': BBox(x, y, x2, y2)
            })

    # 第二步：处理重叠，保留最大的框（合并单元格通常比单个单元格大）
    # 按面积排序，大框优先
    cell_candidates.sort(key=lambda c: (c['bbox'].x2 - c['bbox'].x1) * (c['bbox'].y2 - c['bbox'].y1), reverse=True)

    # 使用网格标记已被占用的单元格位置
    occupied = [[False] * total_cols for _ in range(total_rows)]
    final_cells = []

    for cell in cell_candidates:
        r, c = cell['row'], cell['col']
        rs, cs = cell['rowspan'], cell['colspan']

        # 检查该位置是否已被占用
        can_place = True
        for i in range(r, min(r + rs, total_rows)):
            for j in range(c, min(c + cs, total_cols)):
                if occupied[i][j]:
                    can_place = False
                    break
            if not can_place:
                break

        if can_place:
            # 标记占用
            for i in range(r, min(r + rs, total_rows)):
                for j in range(c, min(c + cs, total_cols)):
                    occupied[i][j] = True
            final_cells.append(cell)

    # 按行列排序
    final_cells.sort(key=lambda x: (x['row'], x['col']))

    return final_cells


def image2tables(image) -> list:
    """
    表格检测主函数：
    1. 如果最大外框占据了整个图像并且包含了所有其他框，则剔除该外框
    2. 正确判断包含关系，保留内部真实表格
    3. 一张图只返回一个表格分组
    """
    if image is None or image.size == 0:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, -10)

    rows, cols = binary.shape
    
    # 动态计算 scale：根据图像尺寸调整，确保形态学核大小合适
    # 小图像（如 800x600）：scale ≈ 10
    # 大图像（如 1600x1200）：scale ≈ 16
    min_dim = min(rows, cols)
    scale = max(8, min(20, min_dim // 80))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    dilatedcol = cv2.dilate(cv2.erode(binary, kernel_h, iterations=1), kernel_h, iterations=1)

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    dilatedrow = cv2.dilate(cv2.erode(binary, kernel_v, iterations=1), kernel_v, iterations=1)

    merge = cv2.add(dilatedcol, dilatedrow)
    merge = cv2.dilate(merge, np.ones((3,3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(merge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_shape = []
    img_area = image.shape[0] * image.shape[1]

    for item in contours:
        x, y, w, h = cv2.boundingRect(item)
        area_ratio = w * h / img_area
        if area_ratio < 0.0003:
            continue
        if (w // h > 6 or h // w > 6) and area_ratio < 0.001448:
            continue
        if (w // h > 15 or h // w > 15) and area_ratio < 0.0018:
            continue
        if w // h > 36 or h // w > 36:
            continue
        contours_shape.append([x, y, w, h])

    if len(contours_shape) >= 1:
        contours_shape_sorted = sorted(contours_shape, key=lambda b: b[2]*b[3], reverse=True)
        max_box = contours_shape_sorted[0]
        mx, my, mw, mh = max_box

        is_full_img = (mw * mh) / img_area > 0.7

        contains_all = True
        for i in range(1, len(contours_shape_sorted)):
            x, y, w, h = contours_shape_sorted[i]
            if not (x >= mx and y >= my and (x+w) <= (mx+mw) and (y+h) <= (my+mh)):
                contains_all = False
                break

        if is_full_img and contains_all:
            contours_shape = contours_shape_sorted[1:]
        else:
            contours_shape = contours_shape_sorted
    else:
        return []

    tables = []
    for idx, shape in enumerate(contours_shape):
        x, y, w, h = shape

        if w * h > 1080 * 2600:
            continue

        contain = False

        for idx1, s1 in enumerate(contours_shape):
            if idx == idx1:
                continue
            x1, y1, w1, h1 = s1

            is_contain = (x >= x1 and
                          y >= y1 and
                          (x + w) <= (x1 + w1) and
                          (y + h) <= (y1 + h1))

            if is_contain:
                contain = True
                break

        if not contain:
            tables.append(shape)

    if not tables:
        return []
    else:
        grouped_tables = [tables]
        return grouped_tables


def image2big_tables(image, min_table_area=2000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables = [cnt for cnt in contours if cv2.contourArea(cnt) > min_table_area]
    table_coordinates = []
    for table in tables:
        x, y, w, h = cv2.boundingRect(table)
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        table_coordinates.append((x1, y1, x2, y2))
    return table_coordinates
