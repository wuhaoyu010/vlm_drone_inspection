"""
Processor 工具函数
提取公共函数，避免代码重复
"""

from typing import Dict, List, Tuple, Any


def get_image_size(image_path: str) -> Tuple[int, int]:
    """获取图片尺寸

    Args:
        image_path: 图片路径

    Returns:
        (width, height) 元组，失败返回 (None, None)
    """
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None, None


def slice_image(
    image_path: str, tile_size: int, overlap: float
) -> List[Tuple[Any, int, int, int, int]]:
    """将图像切片，返回切片列表和位置信息

    Args:
        image_path: 图像路径
        tile_size: 切片尺寸
        overlap: 重叠比例

    Returns:
        List of (tile_image, x_offset, y_offset, tile_w, tile_h)
    """
    from PIL import Image

    img = Image.open(image_path)
    width, height = img.size

    stride = int(tile_size * (1 - overlap))
    tiles = []

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            x1, y1 = x, y
            x2, y2 = min(x + tile_size, width), min(y + tile_size, height)
            tile = img.crop((x1, y1, x2, y2))
            tiles.append((tile, x1, y1, x2 - x1, y2 - y1))

    return tiles


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """计算两个边界框的IOU

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IOU值 [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def merge_detections(
    detections: List[Dict], iou_threshold: float, group_by_category: bool = True
) -> List[Dict]:
    """合并重叠的检测结果

    Args:
        detections: 检测结果列表
        iou_threshold: IOU阈值
        group_by_category: 是否按类别分组后合并

    Returns:
        合并后的检测结果
    """
    if not detections:
        return []

    if group_by_category:
        # 按类别分组后合并
        category_groups = {}
        for det in detections:
            cat = det.get("category", "未知")
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(det)

        merged = []
        for category, group in category_groups.items():
            merged.extend(_merge_group(group, iou_threshold))
        return merged
    else:
        # 不分组，直接合并
        return _merge_group(detections, iou_threshold)


def _merge_group(group: List[Dict], iou_threshold: float) -> List[Dict]:
    """合并单组检测结果"""
    sorted_det = sorted(group, key=lambda x: x.get("confidence", 0), reverse=True)
    used = set()
    merged = []

    for i, det in enumerate(sorted_det):
        if i in used:
            continue

        merge_group = [det]
        for j, other in enumerate(sorted_det[i + 1 :], i + 1):
            if j in used:
                continue
            if compute_iou(det["bbox"], other["bbox"]) > iou_threshold:
                merge_group.append(other)
                used.add(j)

        used.add(i)

        if merge_group:
            x1 = min(d["bbox"][0] for d in merge_group)
            y1 = min(d["bbox"][1] for d in merge_group)
            x2 = max(d["bbox"][2] for d in merge_group)
            y2 = max(d["bbox"][3] for d in merge_group)

            # 保留所有字段
            result = {
                "bbox": [x1, y1, x2, y2],
                "category": det["category"]
            }
            # 如果有 scene_id，也保留
            if "scene_id" in det:
                result["scene_id"] = det["scene_id"]
            merged.append(result)

    return merged

