"""
任务处理器基类
Windows兼容版本
包含公共方法，避免代码重复
"""

import os
import re
import json
import sys
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import get_image_size, slice_image, merge_detections

# 尝试导入配置
try:
    from config import VLM_CONFIDENCE_THRESHOLD
except ImportError:
    VLM_CONFIDENCE_THRESHOLD = 0.5

# RAG知识库导入
RAG_AVAILABLE = False
try:
    from rag_knowledge import get_knowledge_base

    RAG_AVAILABLE = True
except ImportError:
    pass

# RAG Token限制常量（避免token超限）
RAG_MAX_ITEMS = 3  # 最多检索3个类别的知识
RAG_MAX_KNOWLEDGE_LEN = 500  # 每条知识最多500字符
RAG_MAX_ORIGINAL_LEN = 1500  # 原始文本最多1500字符


# Qwen3-VL归一化坐标范围
QWEN_NORMALIZED_RANGE = 1000


def denormalize_bbox(
    bbox: List,
    img_width: int,
    img_height: int,
    processed_width: int = None,
    processed_height: int = None,
    force_normalized: bool = None,
) -> List[int]:
    """
    将归一化坐标[0,1000]转换为实际图片坐标

    Qwen3-VL模型输出的bbox坐标可能是归一化的[0,1000]格式或实际像素坐标，
    需要根据情况自动检测并转换。

    重要：如果图片在发送到模型前被缩放过，需要提供processed_width/processed_height
    以确保坐标转换正确。

    Args:
        bbox: 边界框 [x1, y1, x2, y2]
        img_width: 原始图片宽度
        img_height: 原始图片高度
        processed_width: 发送到模型时的图片宽度（如果有缩放）
        processed_height: 发送到模型时的图片高度（如果有缩放）
        force_normalized: 强制指定是否为归一化坐标（None则自动检测）

    Returns:
        实际像素坐标 [x1, y1, x2, y2]
    """
    if not bbox or len(bbox) != 4:
        return bbox

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]

        # 判断是否为归一化坐标
        if force_normalized is not None:
            is_normalized = force_normalized
        else:
            # 简化检测逻辑：
            # Qwen3-VL 模型输出的归一化坐标范围是 [0, 1000]
            # 如果所有坐标值都 <= 1100（允许10%误差），则认为是归一化坐标
            # 如果有任何坐标值 > 1100，则认为是像素坐标
            max_coord = max(x1, y1, x2, y2)
            is_normalized = max_coord <= QWEN_NORMALIZED_RANGE * 1.1

        if is_normalized:
            # 如果提供了处理后的尺寸，先转换到处理后的像素坐标
            if processed_width and processed_height:
                # 模型看到的是缩放后的图片，先转换到缩放后的像素坐标
                x1_px = x1 * processed_width / QWEN_NORMALIZED_RANGE
                y1_px = y1 * processed_height / QWEN_NORMALIZED_RANGE
                x2_px = x2 * processed_width / QWEN_NORMALIZED_RANGE
                y2_px = y2 * processed_height / QWEN_NORMALIZED_RANGE

                # 然后转换回原始图片坐标
                scale_x = img_width / processed_width
                scale_y = img_height / processed_height
                x1 = int(x1_px * scale_x)
                y1 = int(y1_px * scale_y)
                x2 = int(x2_px * scale_x)
                y2 = int(y2_px * scale_y)
            else:
                # 没有缩放，直接转换
                x1 = int(x1 * img_width / QWEN_NORMALIZED_RANGE)
                y1 = int(y1 * img_height / QWEN_NORMALIZED_RANGE)
                x2 = int(x2 * img_width / QWEN_NORMALIZED_RANGE)
                y2 = int(y2 * img_height / QWEN_NORMALIZED_RANGE)
        else:
            # 已经是实际像素坐标，直接取整
            x1, y1, x2, y2 = [int(v) for v in [x1, y1, x2, y2]]

        return [x1, y1, x2, y2]
    except (ValueError, TypeError):
        return bbox


def is_normalized_bbox(
    bbox: List, img_width: int = None, img_height: int = None
) -> bool:
    """
    判断bbox是否为归一化坐标

    Args:
        bbox: 边界框 [x1, y1, x2, y2]
        img_width: 图片宽度（可选，用于更精确判断）
        img_height: 图片高度（可选）

    Returns:
        是否为归一化坐标
    """
    if not bbox or len(bbox) != 4:
        return False

    try:
        values = [float(v) for v in bbox]

        # 如果所有值都在0-1000范围内，很可能是归一化坐标
        if all(0 <= v <= QWEN_NORMALIZED_RANGE * 1.1 for v in values):
            return True

        # 如果有值超过1000，说明是实际像素坐标
        return False
    except (ValueError, TypeError):
        return False


class BaseProcessor(ABC):
    """任务处理器基类"""

    def __init__(self, vlm_client):
        self.vlm_client = vlm_client
        self._inference_log = []  # 推理日志收集

    def _log(self, message: str):
        """添加推理日志"""
        self._inference_log.append(message)
        print(message)  # 同时打印到控制台

    def _clear_log(self):
        """清空推理日志"""
        self._inference_log = []

    def _get_log_message(self) -> str:
        """获取推理日志作为message字段"""
        return "\n".join(self._inference_log)

    @abstractmethod
    def process(self, input_data: Any, scene_id: int = None) -> Dict[str, Any]:
        """
        处理输入数据

        Args:
            input_data: 输入数据（图片路径或视频路径）
            scene_id: 场景ID

        Returns:
            处理结果字典
        """
        pass

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        从模型响应中解析JSON

        Args:
            response: 模型响应文本

        Returns:
            解析后的字典
        """
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试提取JSON代码块
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_pattern, response)

        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass

        # 尝试提取花括号内容
        brace_pattern = r"\{[\s\S]*\}"
        match = re.search(brace_pattern, response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # 返回空结构
        return {"l1_result": [], "l2_result": response, "l3_result": ""}

    def validate_bbox(
        self, bbox: List, img_width: int = None, img_height: int = None
    ) -> bool:
        """
        验证边界框格式

        Args:
            bbox: 边界框 [x1, y1, x2, y2]
            img_width: 图片宽度（可选）
            img_height: 图片高度（可选）

        Returns:
            是否有效
        """
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]

            # 检查坐标顺序
            if x1 > x2 or y1 > y2:
                return False

            # 检查是否为负数
            if any(v < 0 for v in [x1, y1, x2, y2]):
                return False

            # 检查是否全为0（无效框）
            if all(v == 0 for v in [x1, y1, x2, y2]):
                return False

            # 如果有图片尺寸，检查是否越界
            if img_width is not None and x2 > img_width * 1.1:  # 允许10%误差
                pass  # 不强制要求，只记录
            if img_height is not None and y2 > img_height * 1.1:
                pass

            return True
        except (ValueError, TypeError):
            return False

    def normalize_response(
        self, result: Dict[str, Any], scene_id: int = None
    ) -> Dict[str, Any]:
        """
        标准化响应格式

        Args:
            result: 原始结果
            scene_id: 场景ID

        Returns:
            标准化后的结果
        """
        # 确保l1_result是列表
        if "l1_result" not in result:
            result["l1_result"] = []
        elif not isinstance(result["l1_result"], list):
            # 尝试转换
            if isinstance(result["l1_result"], dict):
                # 可能是单个检测结果
                result["l1_result"] = [result["l1_result"]]
            else:
                result["l1_result"] = []

        # 确保每个bbox项有正确格式
        for item in result["l1_result"]:
            if not isinstance(item, dict):
                continue
            if "bbox" not in item:
                item["bbox"] = [0, 0, 0, 0]
            if "category" not in item:
                item["category"] = ""
            # 添加scene_id
            if scene_id is not None and "scene_id" not in item:
                item["scene_id"] = scene_id

        # 确保l2_result和l3_result是字符串
        if "l2_result" not in result:
            result["l2_result"] = ""
        elif not isinstance(result["l2_result"], str):
            result["l2_result"] = str(result["l2_result"])

        if "l3_result" not in result:
            result["l3_result"] = ""
        elif not isinstance(result["l3_result"], str):
            result["l3_result"] = str(result["l3_result"])

        return result

    def filter_by_confidence(
        self, l1_result: List[Dict], threshold: float = None
    ) -> List[Dict]:
        """
        过滤低置信度的检测结果

        Args:
            l1_result: 检测结果列表
            threshold: 置信度阈值（默认使用配置中的值）

        Returns:
            过滤后的结果列表
        """
        if threshold is None:
            threshold = VLM_CONFIDENCE_THRESHOLD

        if not l1_result:
            return l1_result

        filtered = []
        for item in l1_result:
            confidence = item.get("confidence", 1.0)
            # 如果没有confidence字段，默认保留（兼容旧格式）
            if confidence is None:
                filtered.append(item)
            elif confidence >= threshold:
                filtered.append(item)
            else:
                print(
                    f"[置信度过滤] 过滤低置信度结果: {item.get('category', 'unknown')} (confidence={confidence:.2f} < {threshold})"
                )

        if len(filtered) < len(l1_result):
            print(
                f"[置信度过滤] 过滤完成: {len(l1_result)} -> {len(filtered)}个结果 (阈值={threshold})"
            )

        return filtered

    # ==================== RAG 公共方法 ====================

    def _init_rag_common(self, task_name: str) -> bool:
        """初始化RAG知识库（公共方法）

        Args:
            task_name: 任务名称，用于日志输出

        Returns:
            是否成功启用RAG
        """
        # 如果配置中已禁用RAG，直接返回
        if not self.use_rag:
            print(f"[{task_name}] RAG已在配置中禁用")
            return False

        # 否则检查RAG是否可用
        self.use_rag = RAG_AVAILABLE

        if self.use_rag:
            try:
                self.knowledge_base = get_knowledge_base(
                    knowledge_dir="./knowledge_base",
                    embedding_model_path="./bge-small-zh-v1.5",
                    use_rag=True,
                )
                if self.knowledge_base.is_available():
                    print(f"[{task_name}] RAG知识库已启用，L2/L3输出将基于专业知识增强")
                    return True
                else:
                    print(f"[{task_name}] RAG知识库未就绪，使用默认知识模板")
                    self.use_rag = False
            except Exception as e:
                print(f"[{task_name}] RAG初始化失败: {e}，使用默认知识模板")
                self.use_rag = False

        return False

    def _smart_truncate(self, text: str, max_len: int) -> str:
        """智能截断：在句子边界截断，保持语义完整

        Args:
            text: 原始文本
            max_len: 最大长度

        Returns:
            截断后的文本
        """
        if len(text) <= max_len:
            return text

        # 按优先级查找句子结束符（中文优先）
        for sep in ['。', '！', '？', '；', '\n', '.', '!', '?', ';']:
            pos = text[:max_len].rfind(sep)
            # 确保截断位置不会太短（至少保留50%）
            if pos > max_len * 0.5:
                return text[:pos + 1]

        # 没找到合适的句子边界，直接截断
        return text[:max_len] + "..."

    def _sort_by_confidence(self, items: List[Dict]) -> List[Dict]:
        """按置信度排序检测结果（高置信度优先）

        Args:
            items: 检测结果列表

        Returns:
            排序后的列表
        """
        return sorted(items, key=lambda x: x.get("confidence", 0), reverse=True)

    def _enhance_with_rag_common(
        self,
        l1_results: List[Dict],
        original_text: str,
        retrieve_func: Callable,
        task_name: str,
        result_type: str = "L2",
    ) -> str:
        """RAG增强公共方法

        Args:
            l1_results: 检测结果列表
            original_text: 原始文本
            retrieve_func: 知识检索函数
            task_name: 任务名称
            result_type: 结果类型（L2或L3）

        Returns:
            增强后的文本
        """
        if not self.use_rag or not l1_results:
            return original_text

        # 检索知识（按置信度排序，限制最多3个检测结果的知识，避免token超限）
        sorted_results = self._sort_by_confidence(l1_results)
        rag_knowledge_list = []
        max_items = min(RAG_MAX_ITEMS, len(sorted_results))
        max_knowledge_len = RAG_MAX_KNOWLEDGE_LEN

        for item in sorted_results[:max_items]:
            category = item.get("category", "未知")

            # 调用检索函数
            try:
                rag_knowledge = retrieve_func(category)
                if rag_knowledge:
                    # 智能截断过长的知识文本
                    rag_knowledge = self._smart_truncate(rag_knowledge, max_knowledge_len)
                    rag_knowledge_list.append(f"【{category}】{rag_knowledge}")
            except Exception:
                pass

        if not rag_knowledge_list:
            return original_text

        # 调用VLM整合
        rag_context = "\n".join(rag_knowledge_list)

        # 智能截断原始文本（如果太长）
        original_text = self._smart_truncate(original_text, RAG_MAX_ORIGINAL_LEN)

        if result_type == "L2":
            prompt = f"""请根据以下专业知识，对原有的分析结果进行整合和完善，输出更专业、更准确的L2分析结果。

原有分析结果：
{original_text}

专业知识依据：
{rag_context}

要求：
1. 保持原有分析的核心内容
2. 将专业知识自然融入，不要简单堆砌
3. 输出格式与原有格式一致
4. 不要添加多余的前缀或说明，直接输出整合后的结果

请直接输出整合后的L2分析结果："""
        else:
            prompt = f"""请根据以下专业知识，对原有的风险评估进行整合和完善，输出更专业、更准确的L3风险评估结果。

原有风险评估：
{original_text}

专业处置依据：
{rag_context}

要求：
1. 保持原有风险评估的核心内容（风险等级、风险说明、处理建议）
2. 将专业知识自然融入处理建议部分
3. 输出格式与原有格式一致
4. 不要添加多余的前缀或说明，直接输出整合后的结果

请直接输出整合后的L3风险评估结果："""

        try:
            response = self.vlm_client.chat(
                prompt=prompt,
                system_prompt=f"你是一位专业的公路养护专家，负责整合{result_type}分析结果。",
            )
            enhanced = (
                response.get("content", original_text)
                if isinstance(response, dict)
                else str(response)
            )
            print(f"[{task_name}] RAG增强{result_type}完成")
            return enhanced
        except Exception as e:
            print(f"[{task_name}] RAG增强{result_type}失败: {e}，返回原始结果")
            return original_text

    # ==================== 切片推理公共方法 ====================

    def _process_tiled_slice(
        self,
        tile_img,
        x_off: int,
        y_off: int,
        tile_w: int,
        tile_h: int,
        img_width: int,
        img_height: int,
        prompt: str,
        system_prompt: str,
        infer_scene_id_func: Callable = None,
        default_category: str = "未知",
    ) -> List[Dict]:
        """处理单个切片的检测

        Args:
            tile_img: 切片图像
            x_off, y_off: 切片偏移
            tile_w, tile_h: 切片尺寸
            img_width, img_height: 原图尺寸
            prompt: 提示词
            system_prompt: 系统提示
            infer_scene_id_func: 推断scene_id的函数
            default_category: 默认类别

        Returns:
            检测结果列表
        """
        detections = []

        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tile_path = tmp.name
            tile_img.save(tile_path)

        try:
            response_data = self.vlm_client.chat_with_image(
                image_path=tile_path, prompt=prompt, system_prompt=system_prompt
            )

            processed_dimensions = response_data.get("processed_dimensions")
            processed_w, processed_h = (
                processed_dimensions if processed_dimensions else (None, None)
            )

            result = self.parse_json_response(response_data.get("content", ""))

            for item in result.get("l1_result", []):
                bbox = item.get("bbox", [])
                if not bbox:
                    continue

                # 坐标转换
                tile_bbox = denormalize_bbox(
                    bbox, tile_w, tile_h, processed_w, processed_h
                )
                original_bbox = [
                    tile_bbox[0] + x_off,
                    tile_bbox[1] + y_off,
                    tile_bbox[2] + x_off,
                    tile_bbox[3] + y_off,
                ]

                if self.validate_bbox(original_bbox, img_width, img_height):
                    detection = {
                        "bbox": original_bbox,
                        "category": item.get("category", default_category),
                        "confidence": item.get("confidence", 1.0),
                    }

                    # 如果有推断函数，添加scene_id
                    if infer_scene_id_func:
                        category = item.get("category", "").lower()
                        detection["scene_id"] = infer_scene_id_func(category)

                    detections.append(detection)

        except Exception as e:
            print(f"切片检测失败: {e}")
        finally:
            try:
                os.unlink(tile_path)
            except:
                pass

        return detections

    def _process_with_tiled_inference_common(
        self,
        image_path: str,
        img_width: int,
        img_height: int,
        tile_config: Dict,
        prompt: str,
        system_prompt: str,
        infer_scene_id_func: Callable = None,
        default_category: str = "未知",
        task_name: str = "Task",
    ) -> List[Dict]:
        """切片推理公共方法 - 并发版本

        Args:
            image_path: 图像路径
            img_width, img_height: 图像尺寸
            tile_config: 切片配置
            prompt: 提示词
            system_prompt: 系统提示
            infer_scene_id_func: 推断scene_id的函数
            default_category: 默认类别
            task_name: 任务名称

        Returns:
            合并后的检测结果
        """
        tile_size = tile_config.get("tile_size", 640)
        overlap = tile_config.get("tile_overlap", 0.2)
        merge_iou = tile_config.get("merge_iou_threshold", 0.3)
        max_workers = tile_config.get("max_workers", 8)  # 并发线程数

        print(f"[{task_name}] 切片推理: tile_size={tile_size}, overlap={overlap:.0%}")
        tiles = slice_image(image_path, tile_size, overlap)
        print(f"[{task_name}] 切片数量: {len(tiles)}, 并发线程: {max_workers}")

        all_detections = []

        # 并发处理所有切片
        def process_single_tile(tile_info):
            i, (tile_img, x_off, y_off, tile_w, tile_h) = tile_info
            return self._process_tiled_slice(
                tile_img,
                x_off,
                y_off,
                tile_w,
                tile_h,
                img_width,
                img_height,
                prompt,
                system_prompt,
                infer_scene_id_func,
                default_category,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_tile, (i, tile))
                       for i, tile in enumerate(tiles)]

            for future in as_completed(futures):
                try:
                    detections = future.result()
                    all_detections.extend(detections)
                except Exception as e:
                    print(f"[{task_name}] 切片处理异常: {e}")

        # 合并重叠检测结果
        merged = merge_detections(all_detections, merge_iou, group_by_category=True)
        print(
            f"[{task_name}] 切片推理完成: 原始{len(all_detections)}个, 合并后{len(merged)}个"
        )

        return merged

    # ==================== 标准推理公共方法 ====================

    def _standard_inference_common(
        self,
        input_data: str,
        img_width: int,
        img_height: int,
        prompt: str,
        system_prompt: str,
        scene_id: int = None,
        infer_scene_id_func: Callable = None,
        task_name: str = "Task",
    ) -> Dict[str, Any]:
        """标准推理公共方法

        Args:
            input_data: 输入数据路径
            img_width, img_height: 图像尺寸
            prompt: 提示词
            system_prompt: 系统提示
            scene_id: 场景ID
            infer_scene_id_func: 推断scene_id的函数
            task_name: 任务名称

        Returns:
            推理结果
        """
        self._log(f"[{task_name}] 调用VLM进行标准推理...")

        response_data = self.vlm_client.chat_with_image(
            image_path=input_data, prompt=prompt, system_prompt=system_prompt
        )

        processed_dimensions = response_data.get("processed_dimensions")
        processed_width, processed_height = (
            processed_dimensions if processed_dimensions else (None, None)
        )
        self._log(f"[{task_name}] VLM处理尺寸: {processed_width}x{processed_height}")

        raw_response = response_data.get("content", "")
        self._log(f"[{task_name}] VLM原始响应(前500字符): {raw_response[:500]}...")

        result = self.parse_json_response(raw_response)
        l1_count = len(result.get("l1_result", []))
        self._log(f"[{task_name}] JSON解析结果: l1_result={l1_count}项")

        result = self.normalize_response(result, scene_id)

        # 坐标转换和验证
        if img_width and img_height:
            valid_items = []
            invalid_count = 0
            for item in result.get("l1_result", []):
                bbox = item.get("bbox", [])
                if bbox:
                    item["bbox"] = denormalize_bbox(
                        bbox, img_width, img_height, processed_width, processed_height
                    )

                if self.validate_bbox(item.get("bbox", []), img_width, img_height):
                    # 推断scene_id
                    if infer_scene_id_func and (
                        "scene_id" not in item or item["scene_id"] is None
                    ):
                        category = item.get("category", "").lower()
                        item["scene_id"] = infer_scene_id_func(category)
                    valid_items.append(item)
                else:
                    invalid_count += 1
                    self._log(f"[{task_name}] 无效边界框被过滤: {item.get('bbox', [])}")

            result["l1_result"] = valid_items
            if invalid_count > 0:
                self._log(f"[{task_name}] 过滤了{invalid_count}个无效边界框")

        return result

    # ==================== L2/L3 生成公共方法 ====================

    def _generate_l2_l3_common(
        self,
        detections: List[Dict],
        empty_l2: str,
        empty_l3: str,
        default_l2: str,
        default_l3: str,
        detection_type: str,
        task_name: str,
    ) -> Tuple[str, str]:
        """生成L2/L3公共方法

        Args:
            detections: 检测结果列表
            empty_l2: 无检测结果时的L2
            empty_l3: 无检测结果时的L3
            default_l2: 默认L2模板
            default_l3: 默认L3模板
            detection_type: 检测类型描述
            task_name: 任务名称

        Returns:
            (l2_result, l3_result)
        """
        if not detections:
            return empty_l2, empty_l3

        # 构建检测摘要
        detection_summary = []
        for i, item in enumerate(detections):
            category = item.get("category", detection_type)
            detection_summary.append(f"  {i + 1}. {category}")

        summary_text = "\n".join(detection_summary)

        # 调用VLM生成
        prompt = f"""请根据以下{detection_type}检测结果，生成专业的 L2 分析过程和 L3 风险评估。

检测到的{detection_type}：
{summary_text}

请按以下JSON格式输出：
```json
{{
"l2_result": "经检测，影像中存在X处{detection_type}隐患...根据特征判断：...。",
"l3_result": "隐患严重程度：致命风险/ 严重风险/一般风险/无风险。风险说明：抛洒物出现在车行区域，...。处理建议：立即下发处置指令，..."
}}
```

要求：
1. L2 要分析每个检测目标的特征、位置、潜在影响
2. L3 要给出合理的风险等级（致命风险/ 严重风险/一般风险/无风险）和具体的处置建议
3. 直接输出JSON，不要有其他内容"""

        try:
            response = self.vlm_client.chat(
                prompt=prompt,
                system_prompt=f"你是一位专业的公路养护专家，负责生成{detection_type}检测的分析报告。",
            )
            content = (
                response.get("content", "")
                if isinstance(response, dict)
                else str(response)
            )

            result = self.parse_json_response(content)
            l2 = result.get("l2_result", default_l2)
            l3 = result.get("l3_result", default_l3)

            return l2, l3
        except Exception as e:
            self._log(f"[{task_name}] L2/L3 生成失败: {e}，使用默认模板")
            return default_l2, default_l3