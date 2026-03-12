"""
Task 3: 路面与护栏病害检测处理器
Windows兼容版本
支持RAG知识库增强L2/L3输出
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

# RAG知识库导入
RAG_AVAILABLE = False
try:
    from rag_knowledge import (
        get_knowledge_base,
        retrieve_knowledge_for_task3_l2,
        retrieve_knowledge_for_task3_l3
    )
    RAG_AVAILABLE = True
except ImportError:
    pass


def get_image_size(image_path: str):
    """获取图片尺寸"""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None, None


class Task3Processor(BaseProcessor):
    """路面与护栏病害检测处理器 - 支持RAG知识库增强"""

    # 场景ID到类别的映射
    SCENE_CATEGORY_MAP = {2: "裂缝", 3: "坑洼", 4: "积水", 5: "护栏破损"}

    def __init__(self, vlm_client):
        super().__init__(vlm_client)
        self.knowledge_base = None
        self.use_rag = False
        self._init_rag()

    def _init_rag(self):
        """初始化RAG知识库"""
        self.use_rag = RAG_AVAILABLE

        if self.use_rag:
            try:
                self.knowledge_base = get_knowledge_base(
                    knowledge_dir="./knowledge_base",
                    embedding_model_path="./bge-small-zh-v1.5",
                    use_rag=True
                )
                if self.knowledge_base.is_available():
                    print("[Task3] RAG知识库已启用，L2/L3输出将基于专业知识增强")
                else:
                    print("[Task3] RAG知识库未就绪，使用默认知识模板")
                    self.use_rag = False
            except Exception as e:
                print(f"[Task3] RAG初始化失败: {e}，使用默认知识模板")
                self.use_rag = False

    def _enhance_l2_with_rag(self, l1_results: List[Dict], original_l2: str) -> str:
        """使用RAG知识库增强L2分析过程"""
        if not self.use_rag or not l1_results:
            return original_l2

        # 为每个检测结果检索相关知识
        enhanced_parts = []
        for item in l1_results:
            disease_type = item.get("category", "病害")
            severity = self._infer_severity(item.get("bbox", []))

            # 检索相关知识
            rag_knowledge = retrieve_knowledge_for_task3_l2(disease_type, severity)

            if rag_knowledge:
                enhanced_parts.append(f"【{disease_type}】{rag_knowledge}")

        if enhanced_parts:
            return f"{original_l2}\n\n【专业知识依据】\n" + "\n".join(enhanced_parts)

        return original_l2

    def _enhance_l3_with_rag(self, l1_results: List[Dict], original_l3: str) -> str:
        """使用RAG知识库增强L3风险评估"""
        if not self.use_rag or not l1_results:
            return original_l3

        # 提取风险等级
        risk_level = "P2"
        if "P0" in original_l3:
            risk_level = "P0"
        elif "P1" in original_l3:
            risk_level = "P1"

        # 为每个检测结果检索相关知识
        enhanced_parts = []
        for item in l1_results:
            disease_type = item.get("category", "病害")
            severity = self._infer_severity(item.get("bbox", []))

            # 检索相关知识
            rag_knowledge = retrieve_knowledge_for_task3_l3(disease_type, severity)

            if rag_knowledge:
                enhanced_parts.append(f"【{disease_type}养护建议】{rag_knowledge}")

        if enhanced_parts:
            return f"{original_l3}\n\n【专业处置依据】\n" + "\n".join(enhanced_parts)

        return original_l3

    def _infer_severity(self, bbox: List[int]) -> str:
        """根据bbox大小推断病害严重程度"""
        if not bbox or len(bbox) < 4:
            return "轻微"

        # 计算面积
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        # 简单推断：面积越大越严重
        if area > 50000:
            return "严重"
        elif area > 10000:
            return "中度"
        else:
            return "轻微"

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
        response_data = self.vlm_client.chat_with_image(
            image_path=input_data, prompt=prompt, system_prompt=TASK3_SYSTEM_PROMPT
        )

        # 获取处理后的图片尺寸（用于正确的坐标转换）
        processed_dimensions = response_data.get("processed_dimensions")
        processed_width, processed_height = processed_dimensions if processed_dimensions else (None, None)

        # 解析响应
        result = self.parse_json_response(response_data.get("content", ""))

        # 验证和标准化
        result = self.normalize_response(result, scene_id)

        # 转换归一化坐标为实际坐标并验证边界框
        if img_width and img_height:
            valid_items = []
            for item in result.get("l1_result", []):
                bbox = item.get("bbox", [])
                if bbox:
                    # 将归一化坐标[0,1000]转换为实际像素坐标
                    # 使用处理后的尺寸进行正确的坐标转换
                    item["bbox"] = denormalize_bbox(
                        bbox, img_width, img_height,
                        processed_width, processed_height
                    )

                if self.validate_bbox(item.get("bbox", []), img_width, img_height):
                    # 根据类别确定scene_id
                    category = item.get("category", "").lower()
                    if "scene_id" not in item or item["scene_id"] is None:
                        item["scene_id"] = self._infer_scene_id(category)
                    valid_items.append(item)
            result["l1_result"] = valid_items

        # RAG增强L2/L3
        if self.use_rag and result.get("l1_result"):
            if result.get("l2_result"):
                result["l2_result"] = self._enhance_l2_with_rag(
                    result["l1_result"],
                    result["l2_result"]
                )
            if result.get("l3_result"):
                result["l3_result"] = self._enhance_l3_with_rag(
                    result["l1_result"],
                    result["l3_result"]
                )

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
