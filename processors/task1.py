"""
Task 1: 路面抛洒物检测处理器
Windows兼容版本
支持RAG知识库增强L2/L3输出
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import BaseProcessor, denormalize_bbox
from prompts import TASK1_SYSTEM_PROMPT, TASK1_USER_PROMPT

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.visualization import draw_bboxes_on_image

# RAG知识库导入
RAG_AVAILABLE = False
try:
    from rag_knowledge import (
        get_knowledge_base,
        retrieve_knowledge_for_task1_l2,
        retrieve_knowledge_for_task1_l3
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


class Task1Processor(BaseProcessor):
    """抛洒物检测处理器 - 支持RAG知识库增强"""

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
                    print("[Task1] RAG知识库已启用，L2/L3输出将基于专业知识增强")
                else:
                    print("[Task1] RAG知识库未就绪，使用默认知识模板")
                    self.use_rag = False
            except Exception as e:
                print(f"[Task1] RAG初始化失败: {e}，使用默认知识模板")
                self.use_rag = False

    def _enhance_l2_with_rag(self, l1_results: List[Dict], original_l2: str) -> str:
        """使用RAG知识库增强L2分析过程"""
        if not self.use_rag or not l1_results:
            return original_l2

        # 为每个检测结果检索相关知识
        enhanced_parts = []
        for item in l1_results:
            debris_type = item.get("category", "抛洒物")
            location = self._infer_location(item.get("bbox", []))

            # 检索相关知识
            rag_knowledge = retrieve_knowledge_for_task1_l2(debris_type, location)

            if rag_knowledge:
                enhanced_parts.append(f"【{debris_type}】{rag_knowledge}")

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
            debris_type = item.get("category", "抛洒物")
            location = self._infer_location(item.get("bbox", []))

            # 检索相关知识
            rag_knowledge = retrieve_knowledge_for_task1_l3(debris_type, location, risk_level)

            if rag_knowledge:
                enhanced_parts.append(f"【{debris_type}处置建议】{rag_knowledge}")

        if enhanced_parts:
            return f"{original_l3}\n\n【专业处置依据】\n" + "\n".join(enhanced_parts)

        return original_l3

    def _infer_location(self, bbox: List[int]) -> str:
        """根据bbox位置推断抛洒物所在车道"""
        if not bbox or len(bbox) < 4:
            return "路面"

        # 简单推断：根据x坐标判断
        x_center = (bbox[0] + bbox[2]) / 2

        # 假设图片宽度为1000（归一化坐标）
        if x_center < 300:
            return "应急车道"
        elif x_center < 700:
            return "行车道"
        else:
            return "超车道"

    def process(self, input_data: str, scene_id: int = 0, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        处理抛洒物检测任务

        Args:
            input_data: 图片路径
            scene_id: 场景ID (默认为0-抛洒物)
            output_dir: 输出目录（用于保存标注图片）

        Returns:
            检测结果
        """
        # 获取图片尺寸（原始尺寸，用于坐标转换）
        img_width, img_height = get_image_size(input_data)

        # 调用VLM
        response_data = self.vlm_client.chat_with_image(
            image_path=input_data,
            prompt=TASK1_USER_PROMPT,
            system_prompt=TASK1_SYSTEM_PROMPT
        )

        # 获取处理后的图片尺寸（用于正确的坐标转换）
        processed_dimensions = response_data.get("processed_dimensions")
        processed_width, processed_height = processed_dimensions if processed_dimensions else (None, None)

        # 解析响应
        result = self.parse_json_response(response_data.get("content", ""))

        # 验证和标准化
        result = self.normalize_response(result, scene_id)

        # 转换归一化坐标为实际坐标
        if img_width and img_height:
            for item in result.get("l1_result", []):
                bbox = item.get("bbox", [])
                if bbox:
                    # 将归一化坐标[0,1000]转换为实际像素坐标
                    # 使用处理后的尺寸进行正确的坐标转换
                    item["bbox"] = denormalize_bbox(
                        bbox, img_width, img_height,
                        processed_width, processed_height
                    )

        # 验证边界框
        if img_width and img_height:
            valid_items = []
            for item in result.get("l1_result", []):
                bbox = item.get("bbox", [])
                if self.validate_bbox(bbox, img_width, img_height):
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
                annotated_path = os.path.join(output_dir, f"{input_path.stem}_annotated{input_path.suffix}")
                draw_bboxes_on_image(input_data, result["l1_result"], annotated_path)
                result["annotated_image"] = annotated_path
            except Exception as e:
                result["annotated_image"] = f"标注图片生成失败: {str(e)}"
        else:
            result["annotated_image"] = ""

        return result
