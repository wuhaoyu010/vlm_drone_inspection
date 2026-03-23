"""
Task 4: 路外病害检测处理器
Windows兼容版本
支持RAG知识库增强L2/L3输出
支持图像切片推理（提升小目标检测）
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import BaseProcessor, RAG_MAX_ITEMS, RAG_MAX_KNOWLEDGE_LEN, RAG_MAX_ORIGINAL_LEN
from .utils import get_image_size
from prompts import TASK4_SYSTEM_PROMPT, TASK4_USER_PROMPT

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.visualization import draw_bboxes_on_image

# RAG知识库导入
try:
    from rag_knowledge import (
        retrieve_knowledge_for_task4_l2,
        retrieve_knowledge_for_task4_l3,
    )
except ImportError:
    retrieve_knowledge_for_task4_l2 = None
    retrieve_knowledge_for_task4_l3 = None

# 配置导入
try:
    from config import config

    def get_task4_config() -> Dict[str, Any]:
        task_config = config.get_task_config("task4")
        return {
            "use_tiled_inference": task_config.get("use_tiled_inference", False),
            "tile_size": task_config.get("tile_size", 1280),
            "tile_overlap": task_config.get("tile_overlap", 0.2),
            "merge_iou_threshold": task_config.get("merge_iou_threshold", 0.3),
            "max_workers": task_config.get("max_workers", 8),
            "use_rag": task_config.get("use_rag", False),
        }
except ImportError:

    def get_task4_config() -> Dict[str, Any]:
        return {
            "use_tiled_inference": False,
            "tile_size": 1280,
            "tile_overlap": 0.2,
            "merge_iou_threshold": 0.3,
            "max_workers": 8,
            "use_rag": False,
        }


class Task4Processor(BaseProcessor):
    """路外病害检测处理器 - 支持RAG知识库增强和切片推理"""

    # 场景ID到类别的映射
    SCENE_CATEGORY_MAP = {
        6: "边坡滑坡",
        7: "排水沟积水",
        8: "排水沟破损",
        9: "标志牌异常",
    }

    def __init__(self, vlm_client):
        super().__init__(vlm_client)
        self.knowledge_base = None
        self.tile_config = get_task4_config()
        self.use_rag = self.tile_config.get("use_rag", False)
        self._init_rag()

    def _init_rag(self):
        """初始化RAG知识库"""
        self._init_rag_common("Task4")

    def _enhance_l2_with_rag(self, l1_results: List[Dict], original_l2: str) -> str:
        """使用RAG知识库增强L2分析过程"""
        if not self.use_rag or not l1_results or not retrieve_knowledge_for_task4_l2:
            return original_l2

        # 按置信度排序，限制检索数量，避免token超限
        sorted_results = self._sort_by_confidence(l1_results)
        max_items = min(RAG_MAX_ITEMS, len(sorted_results))
        max_knowledge_len = RAG_MAX_KNOWLEDGE_LEN

        rag_knowledge_list = []
        for item in sorted_results[:max_items]:
            disease_type = item.get("category", "病害")
            rag_knowledge = retrieve_knowledge_for_task4_l2(disease_type)
            if rag_knowledge:
                rag_knowledge = self._smart_truncate(rag_knowledge, max_knowledge_len)
                rag_knowledge_list.append(f"【{disease_type}】{rag_knowledge}")

        if not rag_knowledge_list:
            return original_l2

        # 智能截断原始文本
        original_l2 = self._smart_truncate(original_l2, RAG_MAX_ORIGINAL_LEN)

        return self._enhance_with_rag_common(
            l1_results,
            original_l2,
            lambda _: "\n".join(rag_knowledge_list),
            "Task4",
            "L2",
        )

    def _enhance_l3_with_rag(self, l1_results: List[Dict], original_l3: str) -> str:
        """使用RAG知识库增强L3风险评估"""
        if not self.use_rag or not l1_results or not retrieve_knowledge_for_task4_l3:
            return original_l3

        # 按置信度排序，限制检索数量，避免token超限
        sorted_results = self._sort_by_confidence(l1_results)
        max_items = min(RAG_MAX_ITEMS, len(sorted_results))
        max_knowledge_len = RAG_MAX_KNOWLEDGE_LEN

        rag_knowledge_list = []
        for item in sorted_results[:max_items]:
            disease_type = item.get("category", "病害")
            severity = self._infer_severity(disease_type)
            rag_knowledge = retrieve_knowledge_for_task4_l3(disease_type, severity)
            if rag_knowledge:
                rag_knowledge = self._smart_truncate(rag_knowledge, max_knowledge_len)
                rag_knowledge_list.append(f"【{disease_type}处置建议】{rag_knowledge}")

        if not rag_knowledge_list:
            return original_l3

        # 智能截断原始文本
        original_l3 = self._smart_truncate(original_l3, RAG_MAX_ORIGINAL_LEN)

        return self._enhance_with_rag_common(
            l1_results,
            original_l3,
            lambda _: "\n".join(rag_knowledge_list),
            "Task4",
            "L3",
        )

    def _infer_severity(self, disease_type: str) -> str:
        """根据病害类型推断严重程度"""
        if "滑坡" in disease_type or "坍塌" in disease_type:
            return "严重"
        elif "积水" in disease_type:
            return "中度"
        else:
            return "轻微"

    def _infer_scene_id(self, category: str) -> int:
        """根据类别推断scene_id"""
        category_lower = category.lower()

        if (
            "边坡" in category_lower
            or "滑坡" in category_lower
            or "坍塌" in category_lower
        ):
            return 6
        elif "排水沟" in category_lower and "积水" in category_lower:
            return 7
        elif "排水沟" in category_lower:
            return 8
        elif "标志牌" in category_lower or "指示牌" in category_lower:
            return 9
        else:
            return 6

    def _generate_l2_l3_for_detections(
        self, detections: List[Dict], scene_id: int
    ) -> tuple:
        """根据检测结果生成 L2/L3 文本"""
        _ = scene_id

        return self._generate_l2_l3_common(
            detections,
            empty_l2="经检测，影像中未发现路外病害隐患。",
            empty_l3="隐患严重程度：无。风险说明：无检测到路外病害。处理建议：无需处理。",
            default_l2=f"经检测，影像中存在{len(detections)}处路外病害隐患（对应第二章隐患清单中「路外病害」类别）。根据病害形态与位置特征判断，各病害形态、范围各异，均位于路侧边缘区域，可能影响道路设施安全；本任务未提供历史巡检对比信息，故本次仅基于单帧病害形态、范围及所在位置进行风险研判。",
            default_l3="隐患严重程度：P2。风险说明：路外病害出现在路侧边缘区域，可能影响道路设施安全。处理建议：及时处置路外病害，优先对高风险区域进行加固或疏通。",
            detection_type="路外病害",
            task_name="Task4",
        )

    def process(
        self, input_data: str, scene_id: int = None, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """处理路外病害检测任务"""
        self._clear_log()
        self._log("[Task4] 开始处理路外病害检测任务")
        self._log(f"[Task4] 输入文件: {input_data}")
        self._log(
            f"[Task4] 场景ID: {scene_id} ({self.SCENE_CATEGORY_MAP.get(scene_id, '全部路外病害')})"
        )

        img_width, img_height = get_image_size(input_data)
        self._log(f"[Task4] 图像尺寸: {img_width}x{img_height}")

        use_tiled = self.tile_config.get("use_tiled_inference", False)
        tile_size = self.tile_config.get("tile_size", 1280)

        # 构建提示词（可能根据scene_id调整）
        prompt = TASK4_USER_PROMPT
        if scene_id is not None:
            category = self.SCENE_CATEGORY_MAP.get(scene_id, "病害")
            prompt = prompt.replace(
                "检测边坡和排水沟的问题",
                f"重点检测{category}问题，同时也可以检测其他类型问题",
            )

        # 判断推理模式
        if use_tiled and img_width and img_height:
            if img_width > tile_size * 1.5 or img_height > tile_size * 1.5:
                self._log(f"[Task4] 图像尺寸超过阈值({tile_size * 1.5})，启用切片推理")

                tiled_detections = self._process_with_tiled_inference_common(
                    input_data,
                    img_width,
                    img_height,
                    self.tile_config,
                    prompt,
                    TASK4_SYSTEM_PROMPT,
                    infer_scene_id_func=self._infer_scene_id,
                    default_category="病害",
                    task_name="Task4",
                )

                # 过滤scene_id
                if scene_id is not None:
                    before = len(tiled_detections)
                    tiled_detections = self._filter_by_scene_id(
                        tiled_detections, scene_id
                    )
                    self._log(
                        f"[Task4] scene_id过滤: {before} -> {len(tiled_detections)}个结果"
                    )

                self._log("[Task4] 调用 VLM 生成 L2/L3 分析结果...")
                l2_result, l3_result = self._generate_l2_l3_for_detections(
                    tiled_detections, scene_id
                )

                result = {
                    "l1_result": tiled_detections,
                    "l2_result": l2_result,
                    "l3_result": l3_result,
                }
            else:
                self._log("[Task4] 图像尺寸未超过阈值，使用标准推理")
                result = self._standard_inference(
                    input_data, img_width, img_height, scene_id, prompt
                )
                if scene_id is not None and result.get("l1_result"):
                    before = len(result["l1_result"])
                    result["l1_result"] = self._filter_by_scene_id(
                        result["l1_result"], scene_id
                    )
                    self._log(
                        f"[Task4] scene_id过滤: {before} -> {len(result['l1_result'])}个结果"
                    )
        else:
            self._log("[Task4] 切片推理未启用或图像尺寸无效，使用标准推理")
            result = self._standard_inference(
                input_data, img_width, img_height, scene_id, prompt
            )
            if scene_id is not None and result.get("l1_result"):
                before = len(result["l1_result"])
                result["l1_result"] = self._filter_by_scene_id(
                    result["l1_result"], scene_id
                )
                self._log(
                    f"[Task4] scene_id过滤: {before} -> {len(result['l1_result'])}个结果"
                )

        # RAG增强
        if self.use_rag and result.get("l1_result"):
            self._log("[Task4] 执行RAG增强...")
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
                    f"[Task4] 置信度过滤: {before} -> {len(result['l1_result'])}个结果"
                )

        # 汇总结果
        final_count = len(result.get("l1_result", []))
        if final_count > 0:
            self._log(f"[Task4] 最终检测到 {final_count} 处路外病害")
        else:
            self._log("[Task4] 未检测到路外病害")

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

        self._log("[Task4] 任务处理完成")

        return result

    def _standard_inference(
        self,
        input_data: str,
        img_width: int,
        img_height: int,
        scene_id: int,
        prompt: str = None,
    ) -> Dict[str, Any]:
        """标准推理"""
        if prompt is None:
            prompt = TASK4_USER_PROMPT
            if scene_id is not None:
                category = self.SCENE_CATEGORY_MAP.get(scene_id, "病害")
                prompt = prompt.replace(
                    "检测边坡和排水沟的问题",
                    f"重点检测{category}问题，同时也可以检测其他类型问题",
                )
                self._log(f"[Task4] 提示词已调整: 重点检测{category}")

        return self._standard_inference_common(
            input_data,
            img_width,
            img_height,
            prompt,
            TASK4_SYSTEM_PROMPT,
            scene_id=scene_id,
            infer_scene_id_func=self._infer_scene_id,
            task_name="Task4",
        )

    def _filter_by_scene_id(
        self, l1_result: List[Dict], target_scene_id: int
    ) -> List[Dict]:
        """过滤检测结果，只保留匹配目标scene_id的项目"""
        if target_scene_id is None:
            return l1_result

        filtered = []
        for item in l1_result:
            item_scene_id = item.get("scene_id")
            if item_scene_id is not None:
                if item_scene_id == target_scene_id:
                    filtered.append(item)
            else:
                category = item.get("category", "").lower()
                inferred_id = self._infer_scene_id(category)
                if inferred_id == target_scene_id:
                    item["scene_id"] = inferred_id
                    filtered.append(item)

        return filtered