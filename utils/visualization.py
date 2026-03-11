"""
可视化工具模块 - 用于在图片和视频上绘制检测框
支持中文路径和中文标签
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

# 尝试导入PIL用于中文绘制
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def get_chinese_font(font_size: int = 20):
    """获取支持中文的字体"""
    if not PIL_AVAILABLE:
        return None

    # Windows常用中文字体路径
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",      # 宋体
        "C:/Windows/Fonts/simkai.ttf",      # 楷体
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                continue

    # 使用默认字体
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_chinese_text_pil(
    image: np.ndarray,
    text: str,
    position: tuple,
    font_size: int = 20,
    color: tuple = (255, 255, 255),
    bg_color: tuple = None
) -> np.ndarray:
    """
    使用PIL在OpenCV图像上绘制中文文本

    Args:
        image: OpenCV图像 (BGR格式)
        text: 要绘制的文本
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 文本颜色 (BGR格式)
        bg_color: 背景颜色 (BGR格式)，None表示无背景

    Returns:
        绘制后的图像
    """
    if not PIL_AVAILABLE:
        # PIL不可用时回退到OpenCV（中文会显示为问号）
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)
        return image

    # 获取字体
    font = get_chinese_font(font_size)
    if font is None:
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)
        return image

    # 转换为PIL图像 (RGB格式)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    # 计算文本大小
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x, y = position

    # 绘制背景
    if bg_color is not None:
        # 转换BGR到RGB
        bg_rgb = (bg_color[2], bg_color[1], bg_color[0])
        draw.rectangle(
            [x, y - text_height - 2, x + text_width + 2, y + 2],
            fill=bg_rgb
        )

    # 转换BGR颜色到RGB
    text_rgb = (color[2], color[1], color[0])

    # 绘制文本
    draw.text((x, y - text_height), text, font=font, fill=text_rgb)

    # 转换回OpenCV格式 (BGR)
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return result


def imread_chinese(path: str) -> np.ndarray:
    """读取图片（支持中文路径）"""
    # 使用numpy读取文件，然后用cv2解码
    with open(path, 'rb') as f:
        data = f.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return image


def imwrite_chinese(path: str, image: np.ndarray) -> bool:
    """保存图片（支持中文路径）"""
    # 使用cv2编码，然后用Python写入文件
    ext = os.path.splitext(path)[1]
    success, encoded = cv2.imencode(ext, image)
    if success:
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())
        return True
    return False


# 颜色配置 (BGR格式)
COLORS = {
    'default': (0, 0, 255),      # 红色
    '抛洒物': (0, 165, 255),     # 橙色
    '裂缝': (255, 0, 0),         # 蓝色
    '坑洼': (0, 255, 255),       # 黄色
    '积水': (255, 255, 0),       # 青色
    '护栏破损': (255, 0, 255),   # 紫色
    '边坡滑坡': (0, 255, 0),     # 绿色
    '排水沟积水': (255, 255, 0), # 青色
    '排水沟破损': (0, 165, 255), # 橙色
    '违停': (0, 0, 255),         # 红色
}


def get_color_for_category(category: str) -> tuple:
    """根据类别获取颜色"""
    category_lower = category.lower()
    for key, color in COLORS.items():
        if key in category_lower:
            return color
    return COLORS['default']


def draw_bboxes_on_image(
    image_path: str,
    detections: List[Dict[str, Any]],
    output_path: str,
    line_thickness: int = 2,
    font_scale: float = 0.6
) -> str:
    """
    在图片上绘制检测框（支持中文标签）

    Args:
        image_path: 输入图片路径
        detections: 检测结果列表 [{"bbox": [x1,y1,x2,y2], "category": "类型"}]
        output_path: 输出图片路径
        line_thickness: 边界框线条粗细
        font_scale: 字体大小比例

    Returns:
        输出图片路径
    """
    # 读取图片（支持中文路径）
    image = imread_chinese(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    height, width = image.shape[:2]

    # 计算字体大小（基于图片尺寸）
    font_size = max(16, int(min(width, height) * 0.02 * font_scale / 0.6))

    # 绘制每个检测框
    for det in detections:
        bbox = det.get("bbox", [])
        if not bbox or len(bbox) != 4:
            continue

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # 边界检查
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # 跳过无效框
            if x1 >= x2 or y1 >= y2:
                continue

            # 获取类别和颜色
            category = det.get("category", "检测目标")
            color = get_color_for_category(category)

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

            # 绘制标签（使用PIL支持中文）
            label = category
            image = draw_chinese_text_pil(
                image,
                label,
                (x1, y1),
                font_size=font_size,
                color=(255, 255, 255),
                bg_color=color
            )

        except (ValueError, TypeError) as e:
            print(f"绘制检测框失败: {e}")
            continue

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 保存图片（支持中文路径）
    imwrite_chinese(output_path, image)

    return output_path


def draw_bboxes_on_video(
    video_path: str,
    detections: List[Dict[str, Any]],
    output_path: str,
    line_thickness: int = 2,
    font_scale: float = 0.6
) -> str:
    """
    在视频上绘制检测框（支持中文标签和精确帧级绘制）

    Args:
        video_path: 输入视频路径
        detections: 检测结果列表 [{"frame_id": int, "bbox": [x1,y1,x2,y2], "vehicle_type": "类型"}]
        output_path: 输出视频路径
        line_thickness: 边界框线条粗细
        font_scale: 字体大小比例

    Returns:
        输出视频路径
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Windows兼容的编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 构建帧到检测结果的映射（精确匹配）
    frame_detections = {}
    for det in detections:
        frame_id = det.get("frame_id", 0)
        if frame_id not in frame_detections:
            frame_detections[frame_id] = []
        frame_detections[frame_id].append(det)

    # 计算字体大小
    font_size = max(16, int(min(width, height) * 0.02 * font_scale / 0.6))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 精确匹配当前帧的检测结果
        if frame_count in frame_detections:
            current_detections = frame_detections[frame_count]
        else:
            # 如果没有精确匹配，找最近的检测帧
            all_frames = sorted(frame_detections.keys())
            if all_frames:
                nearest_frame = min(all_frames, key=lambda x: abs(x - frame_count))
                # 只在±fps/2范围内使用最近帧的检测结果
                if abs(frame_count - nearest_frame) <= fps // 2:
                    current_detections = frame_detections[nearest_frame]
                else:
                    current_detections = []
            else:
                current_detections = []

        # 绘制检测框
        for det in current_detections:
            try:
                bbox = det.get("bbox", [0, 0, 100, 100])
                x1, y1, x2, y2 = [
                    int(max(0, min(v, width if i % 2 == 0 else height)))
                    for i, v in enumerate(bbox)
                ]

                # 跳过无效框
                if x1 >= x2 or y1 >= y2:
                    continue

                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)

                # 添加标签（使用PIL支持中文）
                label = det.get("vehicle_type", det.get("category", "违停车辆"))
                frame = draw_chinese_text_pil(
                    frame,
                    label,
                    (x1, y1),
                    font_size=font_size,
                    color=(255, 255, 255),
                    bg_color=(0, 0, 255)
                )

            except (ValueError, TypeError, KeyError) as e:
                continue

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    return output_path


def draw_detections_on_image(
    image_path: str,
    l1_result: List[Dict[str, Any]],
    output_path: str
) -> str:
    """
    根据L1结果在图片上绘制检测框（便捷函数）

    Args:
        image_path: 输入图片路径
        l1_result: L1检测结果
        output_path: 输出图片路径

    Returns:
        输出图片路径
    """
    # 处理不同的l1_result格式
    detections = []
    if isinstance(l1_result, list):
        detections = l1_result
    elif isinstance(l1_result, dict):
        # Task 2格式
        if "violations" in l1_result:
            for v in l1_result.get("violations", []):
                detections.append({
                    "bbox": v.get("bbox", []),
                    "category": v.get("vehicle_type", "违停车辆")
                })
        else:
            detections = [l1_result]

    return draw_bboxes_on_image(image_path, detections, output_path)