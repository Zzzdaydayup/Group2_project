import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_image_with_rounded_corners(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定义黄色的HSV阈值范围
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # 颜色过滤
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 形态学操作：膨胀后腐蚀，称为闭运算，用于填充内部小洞
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 假设最大的轮廓是边框
        largest_contour = max(contours, key=cv2.contourArea)
        # 用多边形近似轮廓，epsilon是近似程度
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 确保我们找到了四个角点
        if len(approx) == 4:
            # 进行透视变换
            # 透视变换的目标点
            pts1 = np.float32([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
            pts2 = np.float32([[0, 0], [270, 0], [270, 310], [0, 310]])  # 保持31:27的比例

            # 计算透视变换矩阵
            matrix = cv2.getPerspectiveTransform(pts1, pts2)

            # 应用透视变换
            result = cv2.warpPerspective(image, matrix, (270, 310))
            
            
            result_rotated = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # 左右镜像
            result2 = cv2.flip(result_rotated, 1)

            # 显示结果
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.show()
            
            # 如果需要，可以保存结果图像
            # cv2.imwrite('adjusted_image.jpg', result)
        else:
            print("未找到四个角点")
    else:
        print("没有找到轮廓")
    return result2


def show_map(map_matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(map_matrix, cmap='gray')  # 使用灰度图来展示
    plt.title('Map Matrix Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

def identify_regions(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 设置颜色阈值
    # 黄色
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 绿色
    lower_green = np.array([5, 40, 30])
    upper_green = np.array([60, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # # 斑马线（简化为白色检测）
    # lower_white = np.array([0, 0, 200])
    # upper_white = np.array([255, 55, 255])
    # mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # # 边缘检测增强斑马线检测
    # edges = cv2.Canny(mask_white, 50, 150)
    
    # 合并所有非行走区域
    mask = cv2.bitwise_or(mask_yellow, mask_green)
    # mask = cv2.bitwise_or(mask, edges)
    
    # 生成地图矩阵
    map_matrix = np.where(mask > 0, 0, 1)  # 不可行走区域为0，可行走区域为1
    
    return map_matrix

image = adjust_image_with_rounded_corners(r'.\mappic\map_origin1.jpg')

# 处理示例图片
map_matrix = identify_regions(image)

show_map(map_matrix)

