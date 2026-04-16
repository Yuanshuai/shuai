"""
独立的方向检测器，基于 paddle_cls_det.onnx 模型
参考 RapidTableDetection 的实现，但不依赖其包
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import Dict, Any
import os


class OrientationDetector:
    """表格方向检测器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化方向检测器
        
        Args:
            model_path: ONNX 模型文件路径，如果为 None 则使用默认路径
        """
        if model_path is None:
            # 使用项目内的模型文件
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "paddle_cls_det.onnx")
            model_path = os.path.abspath(model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"方向检测模型文件不存在: {model_path}")
        
        # 初始化 ONNX Runtime 会话
        self.session = ort.InferenceSession(model_path)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 输入尺寸
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 224, 224]
        self.input_height = self.input_shape[2]  # 224
        self.input_width = self.input_shape[3]   # 224
        
        # PaddlePaddle 标准归一化参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        print(f"方向检测器初始化完成，模型: {os.path.basename(model_path)}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像 (H, W, C) BGR格式
            
        Returns:
            预处理后的张量 [1, 3, 224, 224]
        """
        # 1. 调整图像尺寸到 224×224
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # 2. BGR 转 RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 3. 转换为 float32 并归一化到 [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # 4. PaddlePaddle 标准归一化
        normalized = (normalized - self.mean) / self.std
        
        # 5. 转换维度顺序: HWC → CHW
        chw = normalized.transpose(2, 0, 1)  # [3, 224, 224]
        
        # 6. 添加批次维度
        batch = np.expand_dims(chw, axis=0)  # [1, 3, 224, 224]
        
        return batch
    
    def detect_orientation(self, image: np.ndarray) -> Dict[str, Any]:
        """
        检测图像的方向
        
        Args:
            image: 输入图像 (H, W, C) BGR格式
            
        Returns:
            方向检测结果字典:
            {
                "angle": 旋转角度 (0, 90, 180, 270),
                "confidence": 置信度 [0, 1],
                "label": 预测标签 (0, 1, 2, 3),
                "probabilities": 各方向概率列表
            }
        """
        # 1. 预处理
        input_tensor = self.preprocess(image)
        
        # 2. 模型推理
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        probabilities = outputs[0][0]  # [4] 形状的概率向量
        
        # 3. 后处理
        # 应用 softmax 将 logits 转换为概率
        exp_logits = np.exp(probabilities - np.max(probabilities))  # 数值稳定性
        probabilities_softmax = exp_logits / np.sum(exp_logits)
        
        pred_label = np.argmax(probabilities_softmax)
        confidence = float(probabilities_softmax[pred_label])
        
        # 4. 映射到角度
        angles = [0, 90, 180, 270]
        angle = angles[pred_label]
        
        return {
            "angle": angle,
            "confidence": confidence,
            "label": pred_label,
            "probabilities": probabilities_softmax.tolist(),
            "logits": probabilities.tolist()  # 保留原始 logits 用于调试
        }
    
    def add_edge_prior(self, image: np.ndarray, edge_points: np.ndarray) -> np.ndarray:
        """
        添加边缘先验信息（参考 RapidTableDetection）
        
        Args:
            image: 输入图像
            edge_points: 边缘点坐标 [[x1,y1], [x2,y2], ...]
            
        Returns:
            添加了边缘框的图像
        """
        img_copy = image.copy()
        
        if edge_points is not None and len(edge_points) > 0:
            # 在图像上绘制边缘框
            points = np.array(edge_points).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                img_copy,
                [points],
                True,
                color=(255, 0, 255),  # 紫色边框
                thickness=3
            )
        
        return img_copy
    
    def detect_with_edge_prior(self, image: np.ndarray, edge_points: np.ndarray = None) -> Dict[str, Any]:
        """
        带边缘先验信息的方向检测（更准确）
        
        Args:
            image: 输入图像
            edge_points: 边缘点坐标，可选
            
        Returns:
            方向检测结果
        """
        if edge_points is not None:
            # 添加边缘先验信息
            image_with_edges = self.add_edge_prior(image, edge_points)
            return self.detect_orientation(image_with_edges)
        else:
            # 直接检测
            return self.detect_orientation(image)


def test_orientation_detector():
    """测试方向检测器"""
    # 创建一个测试图像
    test_img = np.ones((300, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_img, (50, 50), (150, 250), (0, 0, 0), 2)
    
    # 初始化检测器
    detector = OrientationDetector()
    
    # 检测方向
    result = detector.detect_orientation(test_img)
    
    print("方向检测结果:")
    print(f"角度: {result['angle']}°")
    print(f"置信度: {result['confidence']:.3f}")
    print(f"标签: {result['label']}")
    print(f"概率分布: {result['probabilities']}")


if __name__ == "__main__":
    test_orientation_detector()