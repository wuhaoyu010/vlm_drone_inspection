"""
任务处理器基类
Windows兼容版本
"""
import re
import json
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple


# Qwen3-VL归一化坐标范围
QWEN_NORMALIZED_RANGE = 1000


def denormalize_bbox(bbox: List, img_width: int, img_height: int) -> List[int]:
    """
    将归一化坐标[0,1000]转换为实际图片坐标

    Qwen3-VL模型输出的bbox坐标是归一化的[0,1000]格式，
    需要转换为实际图片像素坐标。

    Args:
        bbox: 归一化边界框 [x1, y1, x2, y2]，范围[0, 1000]
        img_width: 原始图片宽度
        img_height: 原始图片高度

    Returns:
        实际像素坐标 [x1, y1, x2, y2]
    """
    if not bbox or len(bbox) != 4:
        return bbox

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]

        # 检测是否为归一化坐标（值在0-1000范围内）
        # 如果坐标值大于图片尺寸，说明不是归一化坐标
        is_normalized = all(v <= QWEN_NORMALIZED_RANGE * 1.1 for v in [x1, y1, x2, y2])

        if is_normalized:
            # 归一化坐标转实际坐标
            x1 = int(x1 * img_width / QWEN_NORMALIZED_RANGE)
            y1 = int(y1 * img_height / QWEN_NORMALIZED_RANGE)
            x2 = int(x2 * img_width / QWEN_NORMALIZED_RANGE)
            y2 = int(y2 * img_height / QWEN_NORMALIZED_RANGE)
        else:
            # 已经是实际坐标，直接取整
            x1, y1, x2, y2 = [int(v) for v in [x1, y1, x2, y2]]

        return [x1, y1, x2, y2]
    except (ValueError, TypeError):
        return bbox


def is_normalized_bbox(bbox: List, img_width: int = None, img_height: int = None) -> bool:
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
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, response)
        
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # 尝试提取花括号内容
        brace_pattern = r'\{[\s\S]*\}'
        match = re.search(brace_pattern, response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        # 返回空结构
        return {
            "l1_result": [],
            "l2_result": response,
            "l3_result": ""
        }
    
    def validate_bbox(self, bbox: List, img_width: int = None, img_height: int = None) -> bool:
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
    
    def normalize_response(self, result: Dict[str, Any], scene_id: int = None) -> Dict[str, Any]:
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
