from paddleocr import PaddleOCR
from PIL import Image

class ImageTextRecognizer:
    def __init__(self, use_gpu=False):
        """
        初始化图像文本识别器。

        参数:
        use_gpu (bool): 是否使用GPU进行加速。
        """
        self.ocr_model = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=use_gpu)
        self.recognized_text = []  # 存储识别的文本结果

    def rotate_image(self, image, angle):
        """旋转图片"""
        return image.rotate(-angle, expand=True)

    def process_coordinates(self, result):
        """
        处理识别结果中的坐标。

        参数:
        result (list): 包含识别结果的列表。

        返回:
        None
        """
        for line in result[0]:
            coordinates, (text, confidence) = line
            d1, d2, _, d4 = coordinates
            x1, y1 = d1
            x2, y2 = d2
            x4, y4 = d4
            # 将文本和置信度添加到识别文本列表
            self.recognized_text.append((d1,text))

    def recognize_text_in_image(self, img_path):
        """
        使用PaddleOCR识别图像中的文本，并根据文本方向旋转图像以提高识别准确性。

        参数:
        img_path (str): 图像文件的路径。

        返回:
        result (list): 包含识别结果的列表。
        """
        # 读取图片
        with Image.open(img_path) as original_image:
            # 进行文本方向检测
            angle_result = self.ocr_model.ocr(img_path, cls=True)
            # 检查是否有检测到文本方向，并获取角度
            angle = None
            for line in angle_result:
                if isinstance(line, list) and len(line) > 1 and 'angle' in line[1]:
                    angle = line[1]['angle']
                    break

            # 如果检测到角度不为0，则旋转图片
            if angle is not None and angle != 0:
                corrected_image = self.rotate_image(original_image, angle)
                corrected_img_path = 'corrected_' + img_path
                corrected_image.save(corrected_img_path)
            else:
                corrected_img_path = img_path

            # 使用纠正后的图片进行文字识别
            result = self.ocr_model.ocr(corrected_img_path)



        # 处理坐标和文本
        self.process_coordinates(result)

        return result
