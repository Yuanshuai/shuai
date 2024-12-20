from ocr import *
from ceshi import *
from  Excel import *

class Chart_conversion:
    def __init__(self, image_path,outfile_name="output.xlsx",mer=True):
        self.img_path = image_path
        self.file_name = outfile_name
        self.merge=mer

# 使用示例
    def do(self):
        ocr = ImageTextRecognizer(self.img_path)
        print("文本识别器初始化完成。")
        ocr.recognize_text_in_image(self.img_path)
        print("文字识别完成")
        image_processor = ImageProcessor(self.img_path)
        print("图像处理器初始化完成。")
        image_processor.detect_lines_and_find_intersections()
        print("单元格结构识别完成。")
        if self.merge==False:
            image_processor.visualize_lines_and_intersections()
        ex=ExcelUtils(self.file_name)
        ex.create_excel_table(image_processor.results,ocr.recognized_text,image_processor.detail,self.merge)
        print("写入文件完成")
