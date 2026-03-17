"""
VLM调用模块 - 支持多种模型服务商（OpenAI兼容接口）
"""

import base64
import math
import os
import sys
import re
from typing import Optional, List, Dict, Any, Tuple
from openai import OpenAI
from pathlib import Path

from config import (
    VLM_API_KEY,
    VLM_BASE_URL,
    VLM_MODEL,
    VLM_MAX_TOKENS,
    VLM_TEMPERATURE,
    VLM_TIMEOUT,
    VLM_PROVIDER,
)


def calculate_qwen_processed_size(
    width: int, height: int, max_pixels: int = None, vl_high_resolution: bool = False
) -> Tuple[int, int]:
    """
    计算Qwen3-VL模型实际处理的图像尺寸

    Qwen3-VL会将图像调整为32的整数倍，并根据像素上限进行缩放。
    这个尺寸是坐标转换的关键基准。

    Args:
        width: 原始图像宽度
        height: 原始图像高度
        max_pixels: 最大像素数，默认使用vLLM默认值
        vl_high_resolution: 是否启用高分辨率模式

    Returns:
        (processed_width, processed_height): 模型实际处理的图像尺寸
    """
    # Qwen3-VL: 每个Token对应32x32像素
    TOKEN_PIXELS = 32

    # 像素下限: 4个Token
    min_pixels = 4 * TOKEN_PIXELS * TOKEN_PIXELS  # 4096

    # 像素上限
    # 注意：vLLM默认使用的max_pixels可能与HuggingFace模型默认值不同
    # 经过测试验证，服务端实际使用的值约为 576 * 32 * 32 = 589824
    if vl_high_resolution:
        max_pixels = 16384 * TOKEN_PIXELS * TOKEN_PIXELS  # 16777216
    elif max_pixels is None:
        # 使用与服务端一致的默认值
        # 这个值对应 vLLM 未指定 mm-processor-kwargs 时的默认行为
        max_pixels = 576 * TOKEN_PIXELS * TOKEN_PIXELS  # 589824

    # 初始对齐到32的整数倍
    h_bar = round(height / TOKEN_PIXELS) * TOKEN_PIXELS
    w_bar = round(width / TOKEN_PIXELS) * TOKEN_PIXELS

    # 缩放处理
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / TOKEN_PIXELS) * TOKEN_PIXELS
        w_bar = math.floor(width / beta / TOKEN_PIXELS) * TOKEN_PIXELS
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / TOKEN_PIXELS) * TOKEN_PIXELS
        w_bar = math.ceil(width * beta / TOKEN_PIXELS) * TOKEN_PIXELS

    return w_bar, h_bar


class VLMClient:
    """视觉语言模型客户端 - OpenAI兼容接口"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        provider: str = None,
    ):
        self.api_key = api_key or VLM_API_KEY
        self.base_url = base_url or VLM_BASE_URL
        self.model = model or VLM_MODEL
        self.provider = provider or VLM_PROVIDER

        if not self.api_key:
            raise ValueError(
                "请设置VLM API Key:\n"
                "1. 在config.yaml中设置 vlm.api_key\n"
                "2. 或设置环境变量 VLM_API_KEY\n"
                f"当前服务商: {self.provider}"
            )

        # 初始化OpenAI客户端（兼容各服务商）
        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=VLM_TIMEOUT
        )

        print(f"VLM客户端初始化成功: {self.provider} / {self.model}")

    def _resize_image(self, image_path: str, max_size: int = 1920) -> tuple:
        """
        调整图片大小以减少token数量，并返回Qwen3-VL实际处理的尺寸

        Args:
            image_path: 图片路径
            max_size: 最大边长 (默认1920)

        Returns:
            (base64_data, mime_type, processed_width, processed_height)
            processed_width/processed_height 是Qwen3-VL实际处理的图像尺寸
        """
        try:
            import cv2
            import numpy as np

            # 支持中文路径
            with open(image_path, "rb") as f:
                data = f.read()
            img_array = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError(f"无法读取图片: {image_path}")

            # 获取原始尺寸
            h, w = img.shape[:2]

            # 如果图片太大，进行缩放（减少传输数据量）
            scale_factor = 1.0
            if max(h, w) > max_size:
                scale_factor = max_size / max(h, w)
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                img = cv2.resize(
                    img, (new_w, new_h), interpolation=cv2.INTER_AREA
                )

            # 编码为JPEG
            _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            base64_data = base64.b64encode(buffer).decode("utf-8")

            # 计算Qwen3-VL实际处理的尺寸（用于坐标转换）
            # 注意：即使我们缩放了图像，Qwen内部还会进一步调整
            qwen_w, qwen_h = calculate_qwen_processed_size(
                int(w * scale_factor), int(h * scale_factor)
            )

            return base64_data, "image/jpeg", qwen_w, qwen_h

        except ImportError:
            # 如果没有opencv，直接读取原图
            with open(image_path, "rb") as f:
                # 尝试用PIL获取尺寸并计算Qwen处理尺寸
                try:
                    from PIL import Image
                    img = Image.open(image_path)
                    qwen_w, qwen_h = calculate_qwen_processed_size(img.width, img.height)
                    return (
                        base64.b64encode(f.read()).decode("utf-8"),
                        self._get_image_mime_type(image_path),
                        qwen_w,
                        qwen_h,
                    )
                except Exception:
                    return (
                        base64.b64encode(f.read()).decode("utf-8"),
                        self._get_image_mime_type(image_path),
                        None,
                        None,
                    )

    def _encode_image(self, image_path: str) -> str:
        """将图片编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_image_mime_type(self, image_path: str) -> str:
        """获取图片MIME类型"""
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_map.get(ext, "image/jpeg")

    def _get_image_content(
        self, image_path: str, resize: bool = True, max_size: int = 1024
    ) -> dict:
        """获取图片内容（支持URL和本地文件）

        Args:
            image_path: 图片路径或URL
            resize: 是否压缩图片以减少token数量
            max_size: 压缩后的最大边长

        Returns:
            dict with image content and optionally processed dimensions
        """
        if image_path.startswith(("http://", "https://")):
            return {"type": "image_url", "image_url": {"url": image_path}}
        else:
            # 本地文件
            if resize:
                # 压缩图片以减少token数量
                base64_data, mime_type, processed_w, processed_h = self._resize_image(
                    image_path, max_size
                )
                result = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
                }
                # 附加处理后的尺寸信息（用于坐标转换）
                if processed_w and processed_h:
                    result["processed_dimensions"] = (processed_w, processed_h)
                return result
            else:
                # 不压缩图片，但仍需计算Qwen处理尺寸（用于坐标转换）
                mime_type = self._get_image_mime_type(image_path)
                base64_data = self._encode_image(image_path)

                # 获取原始图像尺寸并计算Qwen处理尺寸
                try:
                    from PIL import Image
                    with Image.open(image_path) as img:
                        qwen_w, qwen_h = calculate_qwen_processed_size(img.width, img.height)
                        return {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
                            "processed_dimensions": (qwen_w, qwen_h),
                        }
                except Exception:
                    # 无法获取尺寸，返回不带processed_dimensions的结果
                    return {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
                    }

    def _get_video_content(self, video_path: str) -> dict:
        """
        获取视频内容
        注意：大多数VLM API不直接支持视频，需要提取关键帧
        """
        # 检查是否支持视频
        if self.provider in ["openai", "zhipu", "deepseek"]:
            # 这些服务商支持部分视频处理
            if video_path.startswith(("http://", "https://")):
                return {"type": "video_url", "video_url": {"url": video_path}}
            else:
                # 本地视频需要特殊处理
                # 方案1: 使用file://协议（部分服务商支持）
                # 方案2: 提取关键帧作为图片
                return self._extract_video_frames(video_path)
        else:
            # 其他服务商：提取关键帧
            return self._extract_video_frames(video_path)

    def _extract_video_frames(
        self,
        video_path: str,
        max_frames: int = 3,
        max_size: int = 512,
        quality: int = 70,
    ) -> List[dict]:
        """
        从视频中提取关键帧并压缩

        针对token限制优化：
        - 减少帧数（默认3帧）
        - 压缩分辨率（默认512px）
        - 降低JPEG质量（默认70%）

        Args:
            video_path: 视频路径
            max_frames: 最大提取帧数（建议2-4帧）
            max_size: 帧图片最大边长（像素）
            quality: JPEG编码质量（1-100）

        Returns:
            base64编码的帧图片列表
        """
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 计算缩放比例
            scale = 1.0
            if max(width, height) > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
            else:
                new_width, new_height = width, height

            # 计算帧间隔（均匀采样）
            interval = max(1, total_frames // max_frames)

            frames_content = []
            frame_count = 0

            while len(frames_content) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % interval == 0:
                    # 缩放帧
                    if scale < 1.0:
                        frame = cv2.resize(
                            frame, (new_width, new_height), interpolation=cv2.INTER_AREA
                        )

                    # 编码为JPEG（指定质量）
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                    _, buffer = cv2.imencode(".jpg", frame, encode_params)
                    base64_data = base64.b64encode(buffer).decode("utf-8")

                    frames_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_data}"
                            },
                        }
                    )

                frame_count += 1

            cap.release()

            if not frames_content:
                raise ValueError("无法从视频提取帧")

            # 打印调试信息
            print(
                f"视频处理: {total_frames}帧, 提取{len(frames_content)}帧, 分辨率{new_width}x{new_height}"
            )

            return frames_content

        except ImportError:
            raise ImportError(
                "视频处理需要opencv-python，请安装: pip install opencv-python"
            )

    def chat_with_image(
        self, image_path: str, prompt: str, system_prompt: str = None
    ) -> dict:
        """
        单图片对话

        Args:
            image_path: 图片路径或URL
            prompt: 用户提示
            system_prompt: 系统提示

        Returns:
            dict with:
                - content: 模型回复文本
                - processed_dimensions: 处理后的图片尺寸 (width, height)，用于坐标转换
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 构建用户消息
        image_content = self._get_image_content(image_path)
        processed_dimensions = image_content.pop("processed_dimensions", None)

        user_content = [image_content, {"type": "text", "text": prompt}]

        messages.append({"role": "user", "content": user_content})

        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=VLM_MAX_TOKENS,
            temperature=VLM_TEMPERATURE,
        )

        return {
            "content": response.choices[0].message.content,
            "processed_dimensions": processed_dimensions,
        }

    def chat_with_video(
        self, video_path: str, prompt: str, system_prompt: str = None
    ) -> str:
        """
        视频对话（通过提取关键帧实现）

        Args:
            video_path: 视频路径或URL
            prompt: 用户提示
            system_prompt: 系统提示

        Returns:
            模型回复文本
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 提取视频帧
        frames_content = self._extract_video_frames(video_path)

        # 构建用户消息（多图片）
        user_content = frames_content + [
            {
                "type": "text",
                "text": f"以下是视频的关键帧，共{len(frames_content)}帧。{prompt}",
            }
        ]

        messages.append({"role": "user", "content": user_content})

        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=VLM_MAX_TOKENS,
            temperature=VLM_TEMPERATURE,
        )

        return response.choices[0].message.content

    def chat(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        纯文本对话（无图片）

        Args:
            prompt: 用户提示
            system_prompt: 系统提示

        Returns:
            包含content的字典
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=VLM_MAX_TOKENS,
            temperature=VLM_TEMPERATURE,
        )

        return {
            "content": response.choices[0].message.content,
            "processed_dimensions": None,
        }

    def chat_with_images(
        self, image_paths: List[str], prompt: str, system_prompt: str = None
    ) -> str:
        """
        多图片对话

        Args:
            image_paths: 图片路径列表
            prompt: 用户提示
            system_prompt: 系统提示

        Returns:
            模型回复文本
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 构建用户消息
        user_content = []
        for img_path in image_paths:
            user_content.append(self._get_image_content(img_path))
        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content})

        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=VLM_MAX_TOKENS,
            temperature=VLM_TEMPERATURE,
        )

        return response.choices[0].message.content

    def diagnose_coordinate_accuracy(self, test_size: int = 512) -> dict:
        """
        诊断坐标准确性，检测服务端处理参数

        通过发送带有已知位置的测试图片，验证坐标转换是否正确

        Args:
            test_size: 测试图片尺寸

        Returns:
            诊断结果字典
        """
        try:
            from PIL import Image, ImageDraw
            import tempfile
            import os

            # 创建带标记的测试图片
            img = Image.new('RGB', (test_size, test_size), color='white')
            draw = ImageDraw.Draw(img)

            # 在中心绘制一个红色方块
            center = test_size // 2
            marker_size = test_size // 10
            draw.rectangle(
                [center - marker_size, center - marker_size,
                 center + marker_size, center + marker_size],
                fill='red'
            )

            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                test_path = tmp.name
                img.save(test_path)

            try:
                # 询问模型红色方块的位置
                prompt = f"""请识别这张{test_size}x{test_size}图片中心的红色方块位置。
输出JSON格式：{{"center": [x, y]}}
center是方块的中心坐标（像素）。"""

                response = self.chat_with_image(
                    image_path=test_path,
                    prompt=prompt,
                    system_prompt="你是精确的图像分析助手，输出JSON格式坐标。"
                )

                import json
                import re

                content = response.get('content', '')
                processed_dims = response.get('processed_dimensions')

                # 解析结果
                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    result = json.loads(json_match.group())
                    detected_center = result.get('center', [0, 0])

                    # 计算误差
                    expected_center = (center, center)
                    error_x = abs(detected_center[0] - expected_center[0])
                    error_y = abs(detected_center[1] - expected_center[1])

                    return {
                        "status": "success",
                        "test_image_size": test_size,
                        "processed_dimensions": processed_dims,
                        "expected_center": expected_center,
                        "detected_center": tuple(detected_center),
                        "error_pixels": (error_x, error_y),
                        "error_percent": (error_x / test_size * 100, error_y / test_size * 100),
                        "is_accurate": error_x < test_size * 0.05 and error_y < test_size * 0.05
                    }
                else:
                    return {
                        "status": "parse_error",
                        "raw_response": content,
                        "processed_dimensions": processed_dims
                    }

            finally:
                os.unlink(test_path)

        except Exception as e:
            return {"status": "error", "message": str(e)}


# 创建全局客户端实例
_vlm_client: Optional[VLMClient] = None


def get_vlm_client() -> VLMClient:
    """获取VLM客户端实例"""
    global _vlm_client
    if _vlm_client is None:
        _vlm_client = VLMClient()
    return _vlm_client


def reset_vlm_client():
    """重置VLM客户端（用于切换配置）"""
    global _vlm_client
    _vlm_client = None
