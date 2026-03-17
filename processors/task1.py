"""
Task 1: 路面抛洒物检测处理器
Windows兼容版本
支持RAG知识库增强L2/L3输出
支持图像切片推理（提升小目标检测）
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import BaseProcessor, RAG_MAX_ITEMS, RAG_MAX_KNOWLEDGE_LEN, RAG_MAX_ORIGINAL_LEN
from .utils import get_image_size
from prompts import TASK1_SYSTEM_PROMPT, TASK1_USER_PROMPT

# 添加项目根目录到路径
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.visualization import draw_bboxes_on_image

# RAG知识库导入
try:
    from rag_knowledge import (
        retrieve_knowledge_for_task1_l2,
        retrieve_knowledge_for_task1_l3,
    )
except ImportError:
    retrieve_knowledge_for_task1_l2 = None
    retrieve_knowledge_for_task1_l3 = None

# 配置导入
try:
    from config import config

    def get_task1_config() -> Dict[str, Any]:
        task_config = config.get_task_config("task1")
        return {
            "use_tiled_inference": task_config.get("use_tiled_inference", False),
            "tile_size": task_config.get("tile_size", 640),
            "tile_overlap": task_config.get("tile_overlap", 0.2),
            "merge_iou_threshold": task_config.get("merge_iou_threshold", 0.3),
        }
except ImportError:

    def get_task1_config() -> Dict[str, Any]:
        return {
            "use_tiled_inference": False,
            "tile_size": 640,
            "tile_overlap": 0.2,
            "merge_iou_threshold": 0.3,
        }


class Task1Processor(BaseProcessor):
    """抛洒物检测处理器 - 支持RAG知识库增强和切片推理"""

    def __init__(self, vlm_client):
        super().__init__(vlm_client)
        self.knowledge_base = None
        self.use_rag = False
        self.tile_config = get_task1_config()
        self._init_rag()

    def _init_rag(self):
        """初始化RAG知识库"""
        self._init_rag_common("Task1")

    def _enhance_l2_with_rag(self, l1_results: List[Dict], original_l2: str) -> str:
        """使用RAG知识库增强L2分析过程"""
        if not self.use_rag or not l1_results or not retrieve_knowledge_for_task1_l2:
            return original_l2

        # 按置信度排序，限制检索数量，避免token超限
        sorted_results = self._sort_by_confidence(l1_results)
        max_items = min(RAG_MAX_ITEMS, len(sorted_results))
        max_knowledge_len = RAG_MAX_KNOWLEDGE_LEN

        rag_knowledge_list = []
        for item in sorted_results[:max_items]:
            debris_type = item.get("category", "抛洒物")
            location = self._infer_location(item.get("bbox", []))
            rag_knowledge = retrieve_knowledge_for_task1_l2(debris_type, location)
            if rag_knowledge:
                rag_knowledge = self._smart_truncate(rag_knowledge, max_knowledge_len)
                rag_knowledge_list.append(f"【{debris_type}】{rag_knowledge}")

        if not rag_knowledge_list:
            return original_l2

        # 智能截断原始文本
        original_l2 = self._smart_truncate(original_l2, RAG_MAX_ORIGINAL_LEN)

        return self._enhance_with_rag_common(
            l1_results,
            original_l2,
            lambda _: "\n".join(rag_knowledge_list),
            "Task1",
            "L2",
        )

    def _enhance_l3_with_rag(self, l1_results: List[Dict], original_l3: str) -> str:
        """使用RAG知识库增强L3风险评估"""
        if not self.use_rag or not l1_results or not retrieve_knowledge_for_task1_l3:
            return original_l3

        # 提取风险等级
        risk_level = "P2"
        if "P0" in original_l3:
            risk_level = "P0"
        elif "P1" in original_l3:
            risk_level = "P1"

        # 按置信度排序，限制检索数量，避免token超限
        sorted_results = self._sort_by_confidence(l1_results)
        max_items = min(RAG_MAX_ITEMS, len(sorted_results))
        max_knowledge_len = RAG_MAX_KNOWLEDGE_LEN

        rag_knowledge_list = []
        for item in sorted_results[:max_items]:
            debris_type = item.get("category", "抛洒物")
            location = self._infer_location(item.get("bbox", []))
            rag_knowledge = retrieve_knowledge_for_task1_l3(
                debris_type, location, risk_level
            )
            if rag_knowledge:
                rag_knowledge = self._smart_truncate(rag_knowledge, max_knowledge_len)
                rag_knowledge_list.append(f"【{debris_type}处置建议】{rag_knowledge}")

        if not rag_knowledge_list:
            return original_l3

        # 智能截断原始文本
        original_l3 = self._smart_truncate(original_l3, RAG_MAX_ORIGINAL_LEN)

        return self._enhance_with_rag_common(
            l1_results,
            original_l3,
            lambda _: "\n".join(rag_knowledge_list),
            "Task1",
            "L3",
        )

    def _infer_location(self, bbox: List[int]) -> str:
        """根据bbox位置推断抛洒物所在车道"""
        if not bbox or len(bbox) < 4:
            return "路面"

        x_center = (bbox[0] + bbox[2]) / 2

        if x_center < 300:
            return "应急车道"
        elif x_center < 700:
            return "行车道"
        else:
            return "超车道"

    def _generate_l2_l3_for_detections(
        self, detections: List[Dict], scene_id: int
    ) -> tuple:
        """根据检测结果生成 L2/L3 文本"""
        _ = scene_id

        return self._generate_l2_l3_common(
            detections,
            empty_l2="经检测，影像中未发现抛洒物隐患。",
            empty_l3="隐患严重程度：无。风险说明：无检测到抛洒物。处理建议：无需处理。",
            default_l2=f"经检测，影像中存在{len(detections)}处抛洒物隐患（对应第二章隐患清单中「抛洒物」类别）。根据外观与体积特征判断，各抛洒物形态、材质各异，均位于道路车行区域，可能迫使后方车辆紧急避让；本任务未提供历史巡检对比信息，故本次仅基于单帧材质特性（硬/软）、尺寸观感及所在车道位置进行风险研判。",
            default_l3="隐患严重程度：P2。风险说明：抛洒物出现在车行区域，可能对高速车辆形成直接撞击与失控风险。处理建议：及时清理路面抛洒物，优先对行车道/超车道内抛洒物进行快速清障。",
            detection_type="抛洒物",
            task_name="Task1",
        )

    def process(
        self, input_data: str, scene_id: int = 0, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """处理抛洒物检测任务"""
        self._clear_log()
        self._log("[Task1] 开始处理抛洒物检测任务")
        self._log(f"[Task1] 输入文件: {input_data}")
        self._log(f"[Task1] 场景ID: {scene_id}")

        img_width, img_height = get_image_size(input_data)
        self._log(f"[Task1] 图像尺寸: {img_width}x{img_height}")

        use_tiled = self.tile_config.get("use_tiled_inference", False)
        tile_size = self.tile_config.get("tile_size", 640)

        # 判断推理模式
        if use_tiled and img_width and img_height:
            if img_width > tile_size * 1.5 or img_height > tile_size * 1.5:
                self._log(f"[Task1] 图像尺寸超过阈值({tile_size * 1.5})，启用切片推理")

                tiled_detections = self._process_with_tiled_inference_common(
                    input_data,
                    img_width,
                    img_height,
                    self.tile_config,
                    TASK1_USER_PROMPT,
                    TASK1_SYSTEM_PROMPT,
                    infer_scene_id_func=None,
                    default_category="抛洒物",
                    task_name="Task1",
                )

                self._log("[Task1] 调用 VLM 生成 L2/L3 分析结果...")
                l2_result, l3_result = self._generate_l2_l3_for_detections(
                    tiled_detections, scene_id
                )

                result = {
                    "l1_result": tiled_detections,
                    "l2_result": l2_result,
                    "l3_result": l3_result,
                }
            else:
                self._log("[Task1] 图像尺寸未超过阈值，使用标准推理")
                result = self._standard_inference(
                    input_data, img_width, img_height, scene_id
                )
        else:
            self._log("[Task1] 切片推理未启用或图像尺寸无效，使用标准推理")
            result = self._standard_inference(
                input_data, img_width, img_height, scene_id
            )

        # RAG增强
        if self.use_rag and result.get("l1_result"):
            if result.get("l2_result"):
                result["l2_result"] = self._enhance_l2_with_rag(
                    result["l1_result"], result["l2_result"]
                )
            if result.get("l3_result"):
                result["l3_result"] = self._enhance_l3_with_rag(
                    result["l1_result"], result["l3_result"]
                )

        # 置信度过滤
        if result.get("l1_result"):
            before = len(result["l1_result"])
            result["l1_result"] = self.filter_by_confidence(result["l1_result"])
            if before != len(result["l1_result"]):
                self._log(
                    f"[Task1] 置信度过滤: {before} -> {len(result['l1_result'])}个结果"
                )

        # 汇总结果
        final_count = len(result.get("l1_result", []))
        if final_count > 0:
            self._log(f"[Task1] 最终检测到 {final_count} 处抛洒物")
        else:
            self._log("[Task1] 未检测到抛洒物")

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

        self._log("[Task1] 任务处理完成")

        return result

    def _standard_inference(
        self, input_data: str, img_width: int, img_height: int, scene_id: int
    ) -> Dict[str, Any]:
        """标准推理"""
        return self._standard_inference_common(
            input_data,
            img_width,
            img_height,
            TASK1_USER_PROMPT,
            TASK1_SYSTEM_PROMPT,
            scene_id=scene_id,
            infer_scene_id_func=None,
            task_name="Task1",
        )

