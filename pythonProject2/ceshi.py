import cv2
import numpy as np
#工具类
class LineUtils:
    @staticmethod
    def line_from_polar(rho, theta):
        '''
         从极坐标转换为直线的端点坐标
        :param rho:
        :param theta:
        :return:
        '''
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * a)
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * a)
        return (x1, y1), (x2, y2)

    @staticmethod
    def is_horizontal_or_vertical(theta, threshold=1e-2):
        '''
         判断直线是否是水平或垂直的。
        :param theta:
        :param threshold:
        :return:
        '''
        return abs(theta) < threshold or abs(theta - np.pi) < threshold or \
               abs(theta - np.pi/2) < threshold or abs(theta - 3*np.pi/2) < threshold

    @staticmethod
    def find_intersection(p1, p2, p3, p4):
        '''
        计算两条直线的交点
        :param p1:
        :param p2:
        :param p3:
        :param p4:
        :return:
        '''
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None  # 平行线或者重合线没有交点
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return int(px), int(py)

    @staticmethod
    def merge_close_lines(lines, threshold=15):
        '''
        合并接近的直线
        :param lines:
        :param threshold:
        :return:
        '''
        unique_lines = []
        for line in lines:
            added = False
            for ul in unique_lines:
                ul_m, ul_c = ul
                m, c = line
                if abs(ul_m - m) < threshold and abs(ul_c - c) < 1:
                    added = True
                    break
            if not added:
                unique_lines.append(line)
        return unique_lines
#主要类
class ImageProcessor:
    def __init__(self, image_path):

        self.image_path = image_path
        self.image = None
        self.lines_eq = []  # 存储直线方程
        self.intersections_sorted = []#储存排序后的交点
        self.results = []  # 存储处理后的结果
        self.detail=[]#储存交点的特征
        self.point_to_lines = {}# 创建一个字典来存储点到直线的映射,每个交点对应两条直线，第一条垂直，第二条水平
#工具函数 加载图像
    def load_image(self):
        '''
        加载图像
        :return:
        '''
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print("图片未找到")
            return False
        return True
#工具函数 获取（交点的特征）交点的上下左右是否有直线连接
    def check_feature(self, x, y):
        '''
         检查交点的特征(交点的上下左右是否有直线连接)
        :param x:
        :param y:
        :return:
        '''
        xx, yy = self.intersections_sorted[y][x]
        binary = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 3)
        # 初始化四个方向的标志
        shang = False
        xia = False
        zuo = False
        you = False
        pp = 10
        # 检查上方是否连接着直线
        if y > 0:
            xs, ys = self.intersections_sorted[y - 1][x]
            for i in range(xx - pp, xx + pp):
                ok = 0

                for j in range(ys, yy):
                    if gray[j][i] != 0:
                        ok += 1

                if ok <= pp:
                    shang = True
                    break

        # 检查下方是否连接着直线
        if y < len(self.intersections_sorted) - 1:
            xa, ya = self.intersections_sorted[y + 1][x]
            for i in range(xx - pp, xx + pp):
                ok = 0
                for j in range(yy, ya):
                    if gray[j][i] != 0:
                        ok += 1

                if ok <= pp:
                    xia = True
                    break

        # 检查左侧是否连接着直线
        if x > 0:
            xl, yl = self.intersections_sorted[y][x - 1]
            for j in range(yy - pp, yy + pp):
                ok = 0
                for i in range(xl, xx):
                    if gray[j][i] != 0:
                        ok += 1
                if ok < pp:
                    zuo = True
                    break

        # 检查右侧是否连接着直线
        if x < len(self.intersections_sorted[0]) - 1:
            xr, yr = self.intersections_sorted[y][x + 1]
            for j in range(yy - pp, yy + pp):
                ok = 0
                for i in range(xx, xr):
                    if gray[j][i] != 0:
                        ok += 1

                if ok <= pp:
                    you = True
                    break
        dian = (shang, xia, zuo, you)
        return dian

    # 工具函数，读取当前直线，更新交点
    def find_intersections(self):
            '''
            读取当前直线列表，更新交点列表,存储交点与直线的关系
            :return:
            '''
        
            if not self.lines_eq:
                print("No lines detected.")
                self.intersections_sorted = []
                return False

            intersections = set()
            for i in range(len(self.lines_eq)):
                for j in range(i + 1, len(self.lines_eq)):
                    if abs(self.lines_eq[i][0][0]-self.lines_eq[i][1][0])<=1:
                         m1, c1 = self.lines_eq[i]
                         m2, c2 = self.lines_eq[j]
                    else:
                        m1, c1 = self.lines_eq[j]
                        m2, c2 = self.lines_eq[i]
                    intersection = LineUtils.find_intersection(m1, c1, m2, c2)
                    if intersection:
                        intersections.add(intersection)
                        # 将交点添加到点到直线的映射中
                        point = (int(intersection[0]), int(intersection[1]))
                        if point not in self.point_to_lines:
                            self.point_to_lines[point] = []
                            self.point_to_lines[point].append((m1, c1))
                            self.point_to_lines[point].append((m2, c2))
            self.intersections_sorted = []
            lin = sorted(intersections, key=lambda point: (point[1], point[0]))
            x0, y0 = lin[0]
            lino = []
            for x, y in lin:
                if y0 == y:
                    lino.append((x, y))
                else:
                    self.intersections_sorted.append(lino)
                    lino = [(x, y)]
                    y0 = y
            self.intersections_sorted.append(lino)

            return True

        # 工具函数，根据交点 匹配特定坐标范围和单元格之间的关系
    def process_intersections(self):
            '''
            (读取交点) 储存 特定坐标范围对应的单元格
            :return:
            '''

            # 处理交点获得单元格地址
            column = len(self.intersections_sorted[0])
            row = len(self.intersections_sorted)
            for i in range(row):
                lin = []
                for j in range(column):
                    if i + 1 < row and j + 1 < column:
                        lin.append([self.intersections_sorted[i][j], self.intersections_sorted[i + 1][j + 1],
                                             (i + 1, j + 1)])
                if lin:
                    self.results.append(lin)

    def linedetection(self):
        '''
        读取图像，使用霍夫变换检测图像中的直线，并做 直线拟合 后，储存直线（两个端点）
        :return:
        '''
        if not self.load_image():
            return
        intersections=set()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

        if lines is not None:
            unique_lines = LineUtils.merge_close_lines([line[0] for line in lines if LineUtils.is_horizontal_or_vertical(line[0][1])])
            for line in unique_lines:
                rho, theta = line
                d1, d2 = LineUtils.line_from_polar(rho, theta)
                self.lines_eq.append((d1, d2))
    #主要函数
    def detect_lines_and_find_intersections(self):

            self.linedetection()
            self.find_intersections()
            self.features()
            self.embellish()
            self.find_intersections()
            self.features()
            self.process_intersections()


    #调试用，根据当前成员函数，可视化直线和交点
    def visualize_lines_and_intersections(self):
        '''
         #调试用，根据当前成员函数，可视化直线和交点
        :return:
        '''
        if self.image is None:
            print("图像未加载，请先加载图像。")
            return

        # 在图像上绘制直线
        for d1,d2 in self.lines_eq:
            (x1,y1)=d1
            (x2,y2)=d2
            if abs(x1-x2)<=1:
              cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色直线

        # 显示图像
        cv2.imshow("Lines and Intersections", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
#工具函数，查找点的特征
    def features(self):
        '''
        更新所有点的特征
        :return:
        '''
        dl=[]
        for y in range(len(self.intersections_sorted)):
            lin = []
            for x in range(len(self.intersections_sorted[0])):
                # 使用 check_feature 函数捕捉交点的细节
                details = self.check_feature(x, y)
                lin.append(details)
            dl.append(lin)
        self.detail=dl.copy()
    def embellish(self):
        '''
        根据点的特征更新直线方程（舍弃一些直线）
        （对每条直线做严格检测，如果直线的交点特征在直线方向上全部为假，则舍弃掉）
        :return:
        '''
        lin = []
        for y in range(len(self.intersections_sorted)):
            for x in range(len(self.intersections_sorted[0])):
                try:
                    line1, line2 = self.point_to_lines[self.intersections_sorted[y][x]]
                    point1a, point1b = line1
                    point2a, point2b = line2
                    if True in [self.detail[y][x][0], self.detail[y][x][1]] and (point1a, point1b) not in lin:
                        lin.append((point1a, point1b))
                    if True in [self.detail[y][x][2], self.detail[y][x][3]] and (point2a, point2b) not in lin:
                        lin.append((point2a, point2b))

                except ValueError as e:
                    print(f"解包错误：{e}")
        self.lines_eq=lin.copy()