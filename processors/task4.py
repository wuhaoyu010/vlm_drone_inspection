"""
Task 4: 路外病害检测处理器
Windows兼容版本
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .base import BaseProcessor, denormalize_bbox
from prompts import TASK4_SYSTEM_PROMPT, TASK4_USER_PROMPT

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


class Task4Processor(BaseProcessor):
    """路外病害检测处理器"""

    # 场景ID到类别的映射
    SCENE_CATEGORY_MAP = {
        6: "边坡滑坡",
        7: "排水沟积水",
        8: "排水沟破损"
    }

    def process(self, input_data: str, scene_id: int = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        处理路外病害检测任务

        Args:
            input_data: 图片路径
            scene_id: 场景ID (6=边坡滑坡, 7=排水沟积水, 8=排水沟破损)
            output_dir: 输出目录（用于保存标注图片）

        Returns:
            检测结果
        """
        # 获取图片尺寸（原始尺寸，用于坐标转换）
        img_width, img_height = get_image_size(input_data)

        # 构建提示词
        prompt = TASK4_USER_PROMPT
        if scene_id is not None:
            category = self.SCENE_CATEGORY_MAP.get(scene_id, "病害")
            prompt = prompt.replace("检测边坡和排水沟的问题",
                                   f"重点检测{category}问题，同时也可以检测其他类型问题")

        # 调用VLM
        response = self.vlm_client.chat_with_image(
            image_path=input_data,
            prompt=prompt,
            system_prompt=TASK4_SYSTEM_PROMPT
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
                annotated_path = os.path.join(output_dir, f"{input_path.stem}_annotated{input_path.suffix}")
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

        if "边坡" in category_lower or "滑坡" in category_lower or "坍塌" in category_lower:
            return 6
        elif "排水沟" in category_lower and "积水" in category_lower:
            return 7
        elif "排水沟" in category_lower:
            return 8
        else:
            return 6  # 默认返回边坡滑坡
