"""
Task 3: 路面与护栏病害检测处理器
Windows兼容版本
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base import BaseProcessor, denormalize_bbox
from prompts import TASK3_SYSTEM_PROMPT, TASK3_USER_PROMPT

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.visualization import draw_bboxes_on_image


def get_image_size(image_path: str):
    """获取图片尺寸"""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None, None


class Task3Processor(BaseProcessor):
    """路面与护栏病害检测处理器"""

    # 场景ID到类别的映射
    SCENE_CATEGORY_MAP = {2: "裂缝", 3: "坑洼", 4: "积水", 5: "护栏破损"}

    def process(
        self, input_data: str, scene_id: int = None, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理路面与护栏病害检测任务

        Args:
            input_data: 图片路径
            scene_id: 场景ID (2=裂缝, 3=坑洼, 4=积水, 5=护栏破损)
            output_dir: 输出目录（用于保存标注图片）

        Returns:
            检测结果
        """
        # 获取图片尺寸（原始尺寸，用于坐标转换）
        img_width, img_height = get_image_size(input_data)

        # 构建提示词，根据scene_id强调特定病害类型
        prompt = TASK3_USER_PROMPT
        if scene_id is not None:
            category = self.SCENE_CATEGORY_MAP.get(scene_id, "病害")
            prompt = prompt.replace(
                "检测路面病害和护栏损坏",
                f"重点检测{category}问题，同时也可以检测其他类型病害",
            )

        # 调用VLM
        response = self.vlm_client.chat_with_image(
            image_path=input_data, prompt=prompt, system_prompt=TASK3_SYSTEM_PROMPT
        )

        # 解析响应
        result = self.parse_json_response(response)

        # 验证和标准化
        result = self.normalize_response(result, scene_id)

        # 转换归一化坐标为实际坐标并验证边界框
        if img_width and img_height:
            valid_items = []
            for item in result.get("l1_result", []):
                bbox = item.get("bbox", [])
                if bbox:
                    # 将归一化坐标[0,1000]转换为实际像素坐标
                    item["bbox"] = denormalize_bbox(bbox, img_width, img_height)

                if self.validate_bbox(item.get("bbox", []), img_width, img_height):
                    # 根据类别确定scene_id
                    category = item.get("category", "").lower()
                    if "scene_id" not in item or item["scene_id"] is None:
                        item["scene_id"] = self._infer_scene_id(category)
                    valid_items.append(item)
            result["l1_result"] = valid_items

        # 生成标注图片
        if output_dir and result.get("l1_result"):
            try:
                input_path = Path(input_data)
                annotated_path = os.path.join(
                    output_dir, f"{input_path.stem}_annotated{input_path.suffix}"
                )
                draw_bboxes_on_image(input_data, result["l1_result"], annotated_path)
                result["annotated_image"] = annotated_path
            except Exception as e:
                result["annotated_image"] = f"标注图片生成失败: {str(e)}"
        else:
            result["annotated_image"] = ""

        return result

    def _infer_scene_id(self, category: str) -> int:
        """根据类别推断scene_id"""
        category_lower = category.lower()

        if "裂缝" in category_lower or "裂痕" in category_lower:
            return 2
        elif (
            "坑洼" in category_lower
            or "坑洞" in category_lower
            or "凹陷" in category_lower
        ):
            return 3
        elif "积水" in category_lower or "水" in category_lower:
            return 4
        elif "护栏" in category_lower:
            return 5
        else:
            return 2  # 默认返回裂缝
