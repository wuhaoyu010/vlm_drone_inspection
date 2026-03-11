"""
VLM调用模块 - 支持多种模型服务商（OpenAI兼容接口）
"""
import base64
import os
import sys
import re
from typing import Optional, List, Dict, Any
from openai import OpenAI
from pathlib import Path

from config import (
    VLM_API_KEY, VLM_BASE_URL, VLM_MODEL,
    VLM_MAX_TOKENS, VLM_TEMPERATURE, VLM_TIMEOUT, VLM_PROVIDER
)


class VLMClient:
    """视觉语言模型客户端 - OpenAI兼容接口"""
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        provider: str = None
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
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=VLM_TIMEOUT
        )
        
        print(f"VLM客户端初始化成功: {self.provider} / {self.model}")

    def _resize_image(self, image_path: str, max_size: int = 1920) -> tuple:
        """
        调整图片大小以减少token数量

        Args:
            image_path: 图片路径
            max_size: 最大边长 (默认1920)

        Returns:
            (base64_data, mime_type)
        """
        try:
            import cv2
            import numpy as np

            # 支持中文路径
            with open(image_path, 'rb') as f:
                data = f.read()
            img_array = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError(f"无法读取图片: {image_path}")

            # 获取原始尺寸
            h, w = img.shape[:2]

            # 如果图片太大，进行缩放
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 编码为JPEG
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            base64_data = base64.b64encode(buffer).decode('utf-8')

            return base64_data, "image/jpeg"

        except ImportError:
            # 如果没有opencv，直接读取原图
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8"), self._get_image_mime_type(image_path)

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
            ".bmp": "image/bmp"
        }
        return mime_map.get(ext, "image/jpeg")
    
    def _get_image_content(self, image_path: str, resize: bool = True, max_size: int = 1024) -> dict:
        """获取图片内容（支持URL和本地文件）

        Args:
            image_path: 图片路径或URL
            resize: 是否压缩图片以减少token数量
            max_size: 压缩后的最大边长
        """
        if image_path.startswith(("http://", "https://")):
            return {"type": "image_url", "image_url": {"url": image_path}}
        else:
            # 本地文件
            if resize:
                # 压缩图片以减少token数量
                base64_data, mime_type = self._resize_image(image_path, max_size)
            else:
                mime_type = self._get_image_mime_type(image_path)
                base64_data = self._encode_image(image_path)

            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_data}"
                }
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
        quality: int = 70
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
                        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

                    # 编码为JPEG（指定质量）
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                    _, buffer = cv2.imencode('.jpg', frame, encode_params)
                    base64_data = base64.b64encode(buffer).decode('utf-8')

                    frames_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_data}"
                        }
                    })

                frame_count += 1

            cap.release()

            if not frames_content:
                raise ValueError("无法从视频提取帧")

            # 打印调试信息
            print(f"视频处理: {total_frames}帧, 提取{len(frames_content)}帧, 分辨率{new_width}x{new_height}")

            return frames_content

        except ImportError:
            raise ImportError(
                "视频处理需要opencv-python，请安装: pip install opencv-python"
            )
    
    def chat_with_image(
        self, 
        image_path: str, 
        prompt: str,
        system_prompt: str = None
    ) -> str:
        """
        单图片对话
        
        Args:
            image_path: 图片路径或URL
            prompt: 用户提示
            system_prompt: 系统提示
        
        Returns:
            模型回复文本
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 构建用户消息
        user_content = [
            self._get_image_content(image_path),
            {"type": "text", "text": prompt}
        ]
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=VLM_MAX_TOKENS,
            temperature=VLM_TEMPERATURE
        )
        
        return response.choices[0].message.content
    
    def chat_with_video(
        self,
        video_path: str,
        prompt: str,
        system_prompt: str = None
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
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 提取视频帧
        frames_content = self._extract_video_frames(video_path)
        
        # 构建用户消息（多图片）
        user_content = frames_content + [
            {"type": "text", "text": f"以下是视频的关键帧，共{len(frames_content)}帧。{prompt}"}
        ]
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=VLM_MAX_TOKENS,
            temperature=VLM_TEMPERATURE
        )
        
        return response.choices[0].message.content
    
    def chat_with_images(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: str = None
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
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 构建用户消息
        user_content = []
        for img_path in image_paths:
            user_content.append(self._get_image_content(img_path))
        user_content.append({"type": "text", "text": prompt})
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=VLM_MAX_TOKENS,
            temperature=VLM_TEMPERATURE
        )
        
        return response.choices[0].message.content


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
