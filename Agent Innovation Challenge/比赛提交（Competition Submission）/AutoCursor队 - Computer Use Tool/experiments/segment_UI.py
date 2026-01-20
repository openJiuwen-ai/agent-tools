import cv2
import pyautogui
import numpy as np
import time


def screenshot(x, y, width, height):
    # 截图
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    # 转换为OpenCV格式并保存
    img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return img_cv


def sort_rectangle_vertices(vertices):
    """将矩形的四个顶点按顺时针排序（从左上开始）"""
    if len(vertices) != 4:
        return vertices

    # 计算中心点
    center_x = sum(v[0] for v in vertices) / 4
    center_y = sum(v[1] for v in vertices) / 4

    # 计算每个顶点相对于中心点的角度
    angles = []
    for (x, y) in vertices:
        angle = np.degrees(np.arctan2(y - center_y, x - center_x))
        # 调整角度范围，使左上角（角度在 -135 到 -45 度之间）排第一
        if angle < -135:
            angle += 360
        angles.append(angle)

    # 按角度排序
    sorted_vertices = [v for _, v in sorted(zip(angles, vertices), key=lambda pair: pair[0])]

    return sorted_vertices


def visualize_rectangle_on_image(img, vertices, title="Rectangle Detection"):
    """
    在图像上可视化矩形顶点和边界
    """
    result_img = img.copy()

    # 如果找到了4个顶点
    if len(vertices) == 4:
        # 绘制矩形边界（黄色）
        for i in range(4):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % 4]
            cv2.line(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        # 绘制顶点并标注坐标
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # 红、绿、蓝、青
        vertex_labels = ["TL", "TR", "BR", "BL"]  # 左上、右上、右下、左下

        for i, (x, y) in enumerate(vertices):
            color = colors[i % len(colors)]
            label = vertex_labels[i % len(vertex_labels)]

            # 绘制顶点（大圆点）
            cv2.circle(result_img, (int(x), int(y)), 8, color, -1)
            # 绘制顶点外圈
            cv2.circle(result_img, (int(x), int(y)), 10, (255, 255, 255), 2)

            # 标注坐标
            coord_text = f"{label}: ({int(x)}, {int(y)})"

            # 根据顶点位置调整文本位置，避免重叠
            if i == 0:  # 左上
                text_pos = (int(x) + 10, int(y) - 10)
            elif i == 1:  # 右上
                text_pos = (int(x) - 120, int(y) - 10)
            elif i == 2:  # 右下
                text_pos = (int(x) - 120, int(y) + 25)
            else:  # 左下
                text_pos = (int(x) + 10, int(y) + 25)

            cv2.putText(result_img, coord_text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 在图像中心添加总结信息
        height, width = result_img.shape[:2]
        summary_text = f"Rectangle Detected: {len(vertices)} vertices"
        cv2.putText(result_img, summary_text, (width // 2 - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_img, summary_text, (width // 2 - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    # 显示图像
    cv2.imshow(title, result_img)
    print(f"按任意键关闭 '{title}' 窗口...")
    cv2.waitKey(0)
    cv2.destroyWindow(title)

    return result_img


def find_rectangle_corners_simple(img, visualize=True):
    original_img = img.copy()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    # 查找黄色像素
    yellow_mask = ((r == 255) & (g == 255) & (b == 0)).astype(np.uint8)

    if np.sum(yellow_mask) == 0:
        print("未找到精确的黄色像素，尝试带容差查找...")
        yellow_mask = ((r >= 245) & (g >= 245) & (b <= 10)).astype(np.uint8)

    # 获取黄色像素数量
    yellow_pixel_count = np.sum(yellow_mask > 0)
    print(f"找到黄色像素数量: {yellow_pixel_count}")

    if yellow_pixel_count < 4:
        print(f"错误：黄色像素太少 ({yellow_pixel_count})，无法构成矩形")
        return []

    # 获取所有黄色像素坐标
    points = np.column_stack(np.where(yellow_mask > 0))

    # 反转坐标 (y, x) -> (x, y)
    points = points[:, [1, 0]]

    # 找到凸包（凸包上的点）
    hull = cv2.convexHull(points.astype(np.float32))

    # 近似多边形
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # 如果是四边形
    if len(approx) == 4:
        vertices = [point[0].tolist() for point in approx]
        sorted_vertices = sort_rectangle_vertices(vertices)

        print(f"检测到四边形，顶点已排序:")
        for i, (x, y) in enumerate(sorted_vertices):
            print(f"  顶点{i + 1}: ({x:.1f}, {y:.1f})")

        # 可视化结果
        if visualize:
            visualize_rectangle_on_image(original_img, sorted_vertices, "Rectangle Detection Result")

        return sorted_vertices
    else:
        print(f"未检测到四边形，检测到 {len(approx)} 个顶点")

        # 即使不是四边形，也可以显示检测到的点
        if visualize and len(approx) > 0:
            vertices = [point[0].tolist() for point in approx]
            print(f"检测到的顶点:")
            for i, (x, y) in enumerate(vertices):
                print(f"  顶点{i + 1}: ({x:.1f}, {y:.1f})")

            # 可视化多边形
            result_img = original_img.copy()

            # 绘制所有检测到的顶点
            for i, (x, y) in enumerate(vertices):
                cv2.circle(result_img, (int(x), int(y)), 8, (0, 0, 255), -1)
                cv2.putText(result_img, f"P{i + 1}", (int(x) + 5, int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 如果顶点数大于2，绘制多边形
            if len(vertices) > 2:
                pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
                cv2.polylines(result_img, [pts], True, (0, 255, 255), 2)

            window_name = f"Polygon with {len(vertices)} vertices"
            cv2.imshow(window_name, result_img)
            print(f"按任意键关闭 '{window_name}' 窗口...")
            cv2.waitKey(0)
            try:
                cv2.destroyWindow(window_name)
            except:
                pass

    return []


def merge_nearby_boxes(boxes, distance_threshold=30):
    """
    合并中心点距离小于 threshold 的框
    :param boxes: list of Box objects from locateAllOnScreen
    :param distance_threshold: 中心点最大允许距离（像素）
    :return: 合并后的 boxes 列表
    """
    if not boxes:
        return []

    # 转换为 (center_x, center_y, box) 列表
    centers = []
    for box in boxes:
        cx, cy = pyautogui.center(box)
        centers.append((cx, cy, box))

    merged = []
    used = [False] * len(centers)

    for i, (cx1, cy1, box1) in enumerate(centers):
        if used[i]:
            continue
        cluster = [box1]
        used[i] = True
        for j in range(i + 1, len(centers)):
            if used[j]:
                continue
            cx2, cy2, box2 = centers[j]
            dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            if dist <= distance_threshold:
                cluster.append(box2)
                used[j] = True

        # 可选：取 cluster 中第一个，或计算平均位置
        # 这里我们取第一个（也可以用平均）
        merged.append(cluster[0])

    return merged


if __name__ == "__main__":
    time.sleep(2)
    exam_pos = pyautogui.locateAllOnScreen('right_top.jpg', confidence=0.8)
    exam_pos = list(exam_pos)
    exam_pos = merge_nearby_boxes(exam_pos)
    boxes_x = []
    for i, loc in enumerate(exam_pos):
        x, y, w, h = loc
        x = x + w
        boxes_x.append(x)
    boxes_x.sort()
    print(len(boxes_x))
    locations = pyautogui.locateAllOnScreen('circle.jpg', confidence=0.55)
    locations_list = list(locations)
    locations_list = merge_nearby_boxes(locations_list)
    print(len(locations_list))
    locations_list.sort()
    for i, loc in enumerate(locations_list):
        x, y = pyautogui.center(loc)
        pyautogui.moveTo(x, y)
        time.sleep(4)

    # img = cv2.imread('Snipaste_2026-01-20_09-13-17.jpg')  # 替换为你的图片路径
    # rect_params = [[333, 115, 737, 528],
    #                [737, 115, 1142, 528],
    #                [1142, 115, 1547, 528]]
    # offset = 2
    # for rect in rect_params:
    #     cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
    #     width = rect[2] - rect[0]
    #     height = rect[3] - rect[1]
    #     scs = screenshot(rect[0], rect[1], width, height)
    #     # 检测矩形顶点
    #     vertices = find_rectangle_corners_simple(scs, visualize=False)
    #     x = vertices[2][0] + rect[0] - offset
    #     y = vertices[2][1] + rect[1] - offset
    #     print(x, y)
    #     pyautogui.moveTo(x, y, duration=0.5)
    #     time.sleep(3)

    # # 显示图片
    # cv2.imshow('Image with Rectangles', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
