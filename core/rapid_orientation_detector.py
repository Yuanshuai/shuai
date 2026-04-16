"""
基于 RapidOrientation 的方向检测器
用于检测文字方向（0°, 90°, 180°, 270°）

注意：RapidOrientation 返回的是损失值（loss），越小表示置信度越高
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List
from rapid_orientation import RapidOrientation


class RapidOrientationDetector:
    """使用 rapid-orientation 库的方向检测器"""
    
    def __init__(self):
        """初始化方向检测器"""
        self.detector = RapidOrientation()
        print("RapidOrientation 方向检测器初始化完成")
    
    def detect_orientation(self, image: np.ndarray) -> Dict[str, Any]:
        """
        检测图像的方向
        
        Args:
            image: 输入图像 (H, W, C) BGR格式
            
        Returns:
            方向检测结果字典:
            {
                "angle": 旋转角度 (0, 90, 180, 270),
                "loss": 损失值（越小越好）,
                "confidence": 转换后的置信度 [0, 1]（越大越好）,
                "label": 预测标签 (0, 1, 2, 3)
            }
        """
        # RapidOrientation 接受 BGR 图像
        angle_str, loss = self.detector(image)
        
        # 转换角度字符串为整数
        angle = int(angle_str)
        
        # 映射到标签
        angle_to_label = {0: 0, 90: 1, 180: 2, 270: 3}
        label = angle_to_label.get(angle, 0)
        
        # 将损失值转换为置信度（越大越好）
        # 使用指数衰减：confidence = exp(-loss * scale)
        # 当 loss = 0.05 时，confidence ≈ 0.6
        # 当 loss = 0.02 时，confidence ≈ 0.82
        # 当 loss = 0.01 时，confidence ≈ 0.90
        confidence = np.exp(-loss * 10)
        
        return {
            "angle": angle,
            "loss": float(loss),
            "confidence": float(confidence),
            "label": label
        }
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        批量检测图像方向
        
        Args:
            images: 图像列表
            
        Returns:
            方向检测结果列表
        """
        results = []
        for img in images:
            result = self.detect_orientation(img)
            results.append(result)
        return results
    
    def get_best_orientation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从多个检测结果中选择置信度最高的方向
        
        Args:
            results: 方向检测结果列表
            
        Returns:
            置信度最高的方向结果
        """
        if not results:
            return {"angle": 0, "loss": 1.0, "confidence": 0.0, "label": 0}
        
        # 选择置信度最高（loss最小）的结果
        best_result = max(results, key=lambda x: x["confidence"])
        return best_result
