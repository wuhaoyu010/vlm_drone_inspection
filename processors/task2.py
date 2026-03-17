"""
Task 2: 车辆违停检测处理器
YOLO + ByteTrack方案（使用ultralytics官方track接口）
支持镜头运动补偿、轨迹绘制、静止车辆检测

核心功能：
1. 光流法镜头运动补偿（适用于无人机/航拍视角）
2. 每帧检测+追踪（与test_2.py一致）
3. 基于补偿轨迹的静止判断
4. RAG知识库增强（提升L2/L3专业性）
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import cv2

from .base import BaseProcessor, denormalize_bbox
from prompts import TASK2_SYSTEM_PROMPT, TASK2_USER_PROMPT

# YOLO导入
from ultralytics import YOLO

# RAG知识库导入
RAG_AVAILABLE = False
try:
    from rag_knowledge import (
        get_knowledge_base,
        retrieve_knowledge_for_l2,
        retrieve_knowledge_for_l3,
        check_rag_availability,
    )

    RAG_AVAILABLE = True
except ImportError:
    pass

# 配置导入
from config import config


def get_task2_config() -> Dict[str, Any]:
    """获取Task2配置，带默认值"""
    task_config = config.get_task_config("task2")
    # 默认车辆类别
    default_vehicle_classes = {
        0: 'wrecker',
        1: 'police_car',
        2: 'maintenance_vehicle',
        3: 'Machineshop',
        4: 'Truck',
        5: 'car',
        6: 'bus'
    }
    return {
        "model_path": task_config.get("model_path", "yolov8n.pt"),
        "conf_threshold": task_config.get("conf_threshold", 0.25),
        "iou_threshold": task_config.get("iou_threshold", 0.7),
        "vehicle_classes": task_config.get("vehicle_classes", default_vehicle_classes),
        "min_track_frames": task_config.get("min_track_frames", 10),
        "trajectory_length": task_config.get("trajectory_length", 50),
        "tracker": task_config.get("tracker", "botsort.yaml"),
        "device": task_config.get("device", "auto"),
        # 静止判断参数（与test_2.py一致）
        "static_time_window": task_config.get("static_time_window", 20),
        "static_dist_threshold": task_config.get("static_dist_threshold", 20.0),
        "use_optical_flow": task_config.get("use_optical_flow", True),
        # RAG知识库参数
        "use_rag": task_config.get("use_rag", True),
        "knowledge_dir": task_config.get("knowledge_dir", "./knowledge_base"),
        "embedding_model_path": task_config.get(
            "embedding_model_path", "./bge-small-zh-v1.5"
        ),
        # 视频输出配置
        "generate_annotated_video": task_config.get("generate_annotated_video", False),
    }


# 车辆类别（从配置文件读取）
_task2_config = get_task2_config()
VEHICLE_CLASSES: Dict[int, str] = _task2_config["vehicle_classes"]


def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """获取视频信息"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return width, height, total_frames, fps
    except ImportError:
        raise ImportError(
            "视频处理需要opencv-python，请安装: pip install opencv-python"
        )


class CameraMotionCompensator:
    """轻量化镜头运动补偿器 - 使用光流法补偿无人机/航拍镜头移动"""

    def __init__(self):
        # 特征点检测参数（优化适配无人机场景，与test_2.py一致）
        self.feature_params = dict(
            maxCorners=200, qualityLevel=0.1, minDistance=5, blockSize=11
        )
        # 光流法参数（与test_2.py一致）
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01),
        )
        self.old_gray = None
        self.p0 = None
        self.prev_transform = np.eye(3)

    def compute_motion(self, frame):
        """计算镜头运动，返回运动偏移量（与test_2.py一致）"""
        # 直接使用原始帧计算光流
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_x, motion_y = 0, 0

        if self.old_gray is None or self.p0 is None or len(self.p0) < 10:
            self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.old_gray = gray.copy()
            return motion_x, motion_y

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_gray, gray, self.p0, None, **self.lk_params
        )

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            if len(good_new) > 10:
                transform, _ = cv2.estimateAffine2D(good_old, good_new)
                if transform is not None:
                    motion_x = transform[0, 2]
                    motion_y = transform[1, 2]
                    affine_to_hom = np.vstack([transform, [0, 0, 1]])
                    self.prev_transform = affine_to_hom @ self.prev_transform

        # 每帧更新特征点（与test_2.py一致）
        self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        self.old_gray = gray.copy()

        return motion_x, motion_y

    def compensate_point(self, x, y):
        """补偿坐标点，返回相对于初始帧的绝对坐标"""
        try:
            point = np.array([[x, y, 1.0]], dtype=np.float32).T
            inv_transform = np.linalg.inv(self.prev_transform)
            compensated_point = inv_transform @ point
            return int(compensated_point[0, 0]), int(compensated_point[1, 0])
        except:
            return x, y

    def reset(self):
        """重置补偿器状态"""
        self.old_gray = None
        self.p0 = None
        self.prev_transform = np.eye(3)


def calculate_displacement(points: List[Tuple[float, float]]) -> float:
    """计算轨迹点的最大位移（包围盒对角线，与test_2.py一致）"""
    if len(points) < 2:
        return 0.0
    points = np.array(points)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    return np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)


class VehicleTracker:
    """车辆追踪器 - YOLO + ByteTrack（官方接口）

    支持镜头运动补偿，适用于无人机/航拍视角
    """

    def __init__(self, model_path: str = None, device: str = None):
        """初始化追踪器"""
        self.config = get_task2_config()

        # 基础配置
        self.model_path = model_path or self.config["model_path"]
        self.conf_threshold = self.config["conf_threshold"]
        self.iou_threshold = self.config["iou_threshold"]
        self.min_track_frames = self.config["min_track_frames"]
        self.trajectory_length = self.config["trajectory_length"]
        self.tracker = self.config["tracker"]

        # 静止判断配置
        self.static_time_window = self.config["static_time_window"]
        self.static_dist_threshold = self.config["static_dist_threshold"]
        self.use_optical_flow = self.config["use_optical_flow"]

        # 视频输出配置
        self.generate_annotated_video = self.config["generate_annotated_video"]

        print(f"[Task2] 加载YOLO模型: {self.model_path}")
        self.model = YOLO(self.model_path)

        # 确定运行设备 - 优先使用传入参数，否则使用配置文件
        device = device or self.config["device"]
        self.device = self._get_device(device)
        if self.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA不可用")
                if not torch.backends.cudnn.is_available():
                    raise RuntimeError("cuDNN不可用")
                self.model.to("cuda")
                print(f"[Task2] 使用GPU: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"[Task2] GPU初始化失败({e})，回退到CPU")
                self.device = "cpu"
        else:
            print(f"[Task2] 使用设备: {self.device}")

        # 打印配置参数
        print(f"[Task2] 光流补偿: {'启用' if self.use_optical_flow else '禁用'}")

    def _get_device(self, device: str) -> str:
        """确定运行设备"""
        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except:
                pass
            return "cpu"
        return device

    def track_video(
        self, video_path: str, output_video: str = None, min_track_frames: int = None
    ) -> Tuple[List[Dict], Dict]:
        """
        追踪视频中的车辆并检测违停

        Args:
            video_path: 视频路径
            output_video: 输出标注视频路径
            min_track_frames: 最小静止帧数阈值

        Returns:
            violations: 违停检测结果
            video_info: 视频信息
        """
        min_track_frames = min_track_frames or self.min_track_frames

        # 获取视频信息
        raw_width, raw_height, total_frames, fps = get_video_info(video_path)

        video_info = {
            "width": raw_width,
            "height": raw_height,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration": round(total_frames / fps, 2) if fps > 0 else 0,
        }

        print(
            f"[Task2] 视频信息: {total_frames}帧, {fps:.1f}fps, {video_info['duration']:.1f}秒"
        )

        # 使用原始帧尺寸处理（与test_2.py一致）
        width, height = raw_width, raw_height

        # 初始化
        cap = cv2.VideoCapture(video_path)
        motion_compensator = (
            CameraMotionCompensator() if self.use_optical_flow else None
        )

        # 追踪状态
        track_history = defaultdict(list)  # 补偿后轨迹
        raw_track_history = defaultdict(list)  # 原始轨迹
        static_status = defaultdict(lambda: False)
        all_tracks = []

        # 轨迹画布（原始帧尺寸）
        scene_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 步骤1：镜头运动补偿（使用原始帧）
            if motion_compensator:
                motion_compensator.compute_motion(frame)

            # 步骤2：执行YOLO追踪（每帧检测，与test_2.py一致）
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
                tracker=self.tracker,
                device=self.device,
            )

            # 步骤4：处理追踪结果
            if (
                results is not None
                and len(results) > 0
                and results[0].boxes.id is not None
            ):
                boxes = results[0].boxes
                track_ids = boxes.id.int().cpu().tolist()
                xyxy = boxes.xyxy.cpu().numpy()
                class_ids = (
                    boxes.cls.int().cpu().tolist()
                    if boxes.cls is not None
                    else [0] * len(track_ids)
                )
                confs = (
                    boxes.conf.cpu().tolist()
                    if boxes.conf is not None
                    else [0.0] * len(track_ids)
                )

                frame_tracks = []
                for i, track_id in enumerate(track_ids):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # 补偿坐标
                    comp_x, comp_y = cx, cy
                    if motion_compensator:
                        comp_x, comp_y = motion_compensator.compensate_point(cx, cy)

                    # 更新轨迹
                    raw_track = raw_track_history[track_id]
                    raw_track.append((cx, cy))
                    if len(raw_track) > self.trajectory_length:
                        raw_track.pop(0)

                    track = track_history[track_id]
                    track.append((comp_x, comp_y))
                    if len(track) > self.trajectory_length:
                        track.pop(0)

                    # 判断静止（基于补偿后轨迹）
                    if len(track) >= self.static_time_window:
                        recent_points = track[-self.static_time_window :]
                        displacement = calculate_displacement(recent_points)
                        static_status[track_id] = (
                            displacement < self.static_dist_threshold
                        )

                    frame_tracks.append(
                        {
                            "track_id": track_id,
                            "bbox": [x1, y1, x2, y2],
                            "center": (cx, cy),
                            "class_id": class_ids[i],
                            "conf": confs[i],
                            "is_static": static_status[track_id],
                        }
                    )

                all_tracks.append({"frame_id": frame_id, "tracks": frame_tracks})

            frame_id += 1
            if frame_id % 100 == 0:
                print(f"[Task2] 已处理 {frame_id}/{total_frames} 帧")

        cap.release()
        print(f"[Task2] 追踪完成: 共追踪 {len(track_history)} 个目标")

        # 分析违停（静止车辆）
        violations = []

        # 调试信息：追踪统计
        total_tracks = len(track_history)
        static_count = len([s for s in static_status.values() if s])
        print(f"[Task2] 追踪统计: 共{total_tracks}个目标, 其中{static_count}个静止")

        static_track_ids = {tid for tid, status in static_status.items() if status}

        for track_id in static_track_ids:
            track = track_history[track_id]
            track_len = len(track)
            print(
                f"[Task2] 静止目标 {track_id}: 轨迹长度={track_len}, 最小要求={min_track_frames}"
            )

            if track_len >= min_track_frames:
                # 获取该track的首次出现位置
                for ft in all_tracks:
                    for t in ft["tracks"]:
                        if t["track_id"] == track_id:
                            violations.append(
                                {
                                    "frame_id": ft["frame_id"],
                                    "bbox": t["bbox"],
                                    "track_id": int(track_id),
                                    "category": "违停车辆",
                                    "vehicle_type": VEHICLE_CLASSES.get(
                                        t.get("class_id", 2), "车辆"
                                    ),
                                    "parking_location": "应急车道",
                                    "track_duration": len(track),
                                    "is_static": True,
                                    "displacement": calculate_displacement(
                                        track[-self.static_time_window :]
                                    )
                                    if len(track) >= self.static_time_window
                                    else 0,
                                    "confidence": t.get(
                                        "conf", 0.9
                                    ),  # 添加YOLO检测置信度
                                }
                            )
                            break
                    else:
                        continue
                    break

        # 生成标注视频（根据配置决定是否生成）
        if output_video and self.generate_annotated_video:
            print(f"[Task2] 生成标注视频...")
            self._generate_annotated_video(
                video_path,
                output_video,
                all_tracks,
                track_history,
                raw_track_history,
                static_status,
                fps,
                width,
                height,
                scene_canvas,
                motion_compensator,
            )
        elif output_video and not self.generate_annotated_video:
            print("[Task2] 标注视频生成已禁用（配置: generate_annotated_video=false）")

        unique_tracks = len(set(v["track_id"] for v in violations))
        print(f"[Task2] 检测到 {unique_tracks} 辆静止车辆（违停）")

        return violations, video_info

    def _generate_annotated_video(
        self,
        video_path: str,
        output_video: str,
        all_tracks: List[Dict],
        track_history: Dict,
        raw_track_history: Dict,
        static_status: Dict,
        fps: float,
        width: int,
        height: int,
        scene_canvas: np.ndarray,
        motion_compensator,
    ):
        """生成带标注和轨迹的视频"""
        os.makedirs(
            os.path.dirname(output_video) if os.path.dirname(output_video) else ".",
            exist_ok=True,
        )

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            output_video, fourcc, int(fps), (width, height)
        )

        frame_tracks_map = {ft["frame_id"]: ft["tracks"] for ft in all_tracks}
        frame_id = 0

        # 重置画布和补偿器
        scene_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if motion_compensator:
            motion_compensator.reset()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 光流补偿（使用原始帧）
            if motion_compensator:
                motion_compensator.compute_motion(frame)

            # 绘制追踪结果
            if frame_id in frame_tracks_map:
                for t in frame_tracks_map[frame_id]:
                    x1, y1, x2, y2 = t["bbox"]
                    track_id = t["track_id"]
                    is_static = static_status.get(track_id, False)

                    # 颜色：静止-红色，移动-绿色
                    color = (0, 0, 255) if is_static else (0, 255, 0)

                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # 绘制标签
                    status_text = "STATIC" if is_static else "MOVING"
                    label = f"ID:{track_id} {status_text}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                    # 绘制原始轨迹（蓝色）
                    raw_track = raw_track_history.get(track_id, [])
                    if len(raw_track) > 1:
                        raw_points = np.array(raw_track, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [raw_points], False, (255, 0, 0), 1)

                    # 绘制补偿后轨迹到场景画布
                    track = track_history.get(track_id, [])
                    if len(track) > 1:
                        track_clipped = [
                            (
                                max(0, min(width - 1, int(px))),
                                max(0, min(height - 1, int(py))),
                            )
                            for px, py in track
                        ]
                        absolute_points = np.array(track_clipped, np.int32).reshape(
                            (-1, 1, 2)
                        )
                        cv2.polylines(scene_canvas, [absolute_points], False, color, 2)

            # 叠加轨迹画布
            frame = cv2.addWeighted(frame, 0.8, scene_canvas, 0.4, 0)

            # 显示运动信息
            if motion_compensator:
                cum_x = motion_compensator.prev_transform[0, 2]
                cum_y = motion_compensator.prev_transform[1, 2]
                motion_text = f"Cam Motion: X={cum_x:.1f}, Y={cum_y:.1f}"
                cv2.putText(
                    frame,
                    motion_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

            # 写入帧
            video_writer.write(frame)

            frame_id += 1

        cap.release()
        video_writer.release()
        print(f"[Task2] 标注视频已保存: {output_video}")


class Task2Processor(BaseProcessor):
    """车辆违停检测处理器 - YOLO + ByteTrack + 光流补偿 + VLM增强 + RAG知识库

    输出格式符合官方标准：
    - L1/L2/L3 均为列表形式，一一对应
    - 每个违停车辆包含 frame_id 和 bbox
    - L2/L3 基于VLM和RAG知识库增强专业性
    """

    def __init__(self, vlm_client, preload_model: bool = True):
        super().__init__(vlm_client)
        self.tracker = None
        self.knowledge_base = None
        self.use_rag = False

        # 初始化RAG知识库
        self._init_rag()

        # 预加载 YOLO 模型（默认启用）
        if preload_model:
            self._get_tracker()

    def _init_rag(self):
        """初始化RAG知识库 - 使用基类公共方法"""
        self._init_rag_common("Task2")

    def _get_tracker(self):
        if self.tracker is None:
            self.tracker = VehicleTracker()
        return self.tracker

    def _normalize_bbox(self, bbox: List[int], width: int, height: int) -> List[int]:
        """将像素坐标转换为归一化[0,1000]格式"""
        x1, y1, x2, y2 = bbox
        return [
            int(x1 * 1000 / width),
            int(y1 * 1000 / height),
            int(x2 * 1000 / width),
            int(y2 * 1000 / height),
        ]

    def format_output_new(
        self,
        violations: List[Dict],
        reasoning: List[str],
        suggestion: List[str],
        scene_id: int = 1,
    ) -> Dict[str, Any]:
        """格式化输出为主办方新格式

        新格式要求：
        - 只输出一个帧（取最早检测到违停的帧）
        - bboxs: 该帧中所有违停车辆的框（实际像素坐标，非归一化）
        - reasoning: 每个违停车辆的分析过程列表
        - suggestion: 每个违停车辆的风险建议列表

        Args:
            violations: 违停检测结果列表
            reasoning: L2分析结果列表（reasoning）
            suggestion: L3风险建议列表（suggestion）
            scene_id: 场景ID

        Returns:
            标准化输出格式
        """
        if not violations:
            return {
                "success": True,
                "message": "",
                "scene_id": scene_id,
                "data": [],
            }

        # 找到最早的帧ID（主办方要求只输出一个帧）
        min_frame_id = min(v["frame_id"] for v in violations)

        # 收集该帧中所有违停车辆的信息
        bboxs = []
        reasoning_list = []
        suggestion_list = []

        for i, v in enumerate(violations):
            # bbox 使用实际像素坐标（不归一化）
            bboxs.append(v["bbox"])

            # reasoning 和 suggestion 与 bbox 一一对应
            if i < len(reasoning):
                reasoning_list.append(reasoning[i])
            else:
                reasoning_list.append(self._generate_l2_fallback(v))

            if i < len(suggestion):
                suggestion_list.append(suggestion[i])
            else:
                suggestion_list.append(self._generate_l3_fallback(v))

        return {
            "success": True,
            "message": "",
            "scene_id": scene_id,
            "data": [
                {
                    "frame_id": min_frame_id,
                    "bboxs": bboxs,
                    "reasoning": reasoning_list,
                    "suggestion": suggestion_list,
                }
            ],
        }

    def _generate_l2_l3_with_vlm_image(
        self, annotated_frame_path: str, violations: List[Dict], video_info: Dict
    ) -> Tuple[List[str], List[str]]:
        """使用VLM分析标注图像，生成L2/L3结果

        Args:
            annotated_frame_path: 标注后的帧图像路径
            violations: 违停检测结果列表
            video_info: 视频信息

        Returns:
            (l2_result_list, l3_result_list)
        """
        l2_result = []
        l3_result = []

        if not violations:
            return l2_result, l3_result

        try:
            # 使用TASK2_SYSTEM_PROMPT和TASK2_USER_PROMPT调用VLM
            response = self.vlm_client.chat_with_image(
                image_path=annotated_frame_path,
                prompt=TASK2_USER_PROMPT,
                system_prompt=TASK2_SYSTEM_PROMPT,
            )

            if response and response.get("content"):
                content = response["content"]
                # 使用基类的parse_json_response方法
                parsed = self.parse_json_response(content)

                if parsed:
                    # 从解析结果中提取L2和L3
                    l2_result = parsed.get("l2_result", [])
                    l3_result = parsed.get("l3_result", [])

                    # 确保返回的列表长度与violations一致
                    while len(l2_result) < len(violations):
                        l2_result.append(
                            self._generate_l2_fallback(violations[len(l2_result)])
                        )
                    while len(l3_result) < len(violations):
                        l3_result.append(
                            self._generate_l3_fallback(violations[len(l3_result)])
                        )

                    # 截断多余的结果
                    l2_result = l2_result[: len(violations)]
                    l3_result = l3_result[: len(violations)]

        except Exception as e:
            print(f"[Task2] VLM图像分析失败: {e}，使用模板生成")

        # 如果VLM未能生成结果，使用回退方法
        if not l2_result:
            for v in violations:
                l2_result.append(self._generate_l2_fallback(v))
        if not l3_result:
            for v in violations:
                l3_result.append(self._generate_l3_fallback(v))

        return l2_result, l3_result

    def _generate_l2_fallback(self, v: Dict) -> str:
        """L2生成回退方法"""
        vehicle_type = v.get("vehicle_type", "车辆")
        parking_loc = v.get("parking_location", "应急车道")
        track_duration = v.get("track_duration", 0)
        displacement = v.get("displacement", 0)

        is_large = vehicle_type in ["bus", "truck"]
        vehicle_size = "大型" if is_large else "小型"
        impact_level = "严重影响" if parking_loc == "行车道" else "影响"
        static_confirm = "确认为静止状态" if displacement < 15 else "状态待确认"

        return (
            f"分析过程：检测到{vehicle_type}在{parking_loc}区域静止停放。"
            f"该车辆属于{vehicle_size}车辆，停车时长约{track_duration}帧，位移{displacement:.1f}像素，{static_confirm}。"
            f"{impact_level}其他车辆通行，{'属于高危违停行为' if parking_loc == '行车道' else '存在安全隐患'}。"
            f"车辆位于{parking_loc}，经视觉检测未发现双闪灯亮起，未检测到三角警示牌，非执行任务的特种车辆，符合违停判定条件。"
        )

    def _generate_l3_fallback(self, v: Dict) -> str:
        """L3生成回退方法"""
        vehicle_type = v.get("vehicle_type", "车辆")
        parking_loc = v.get("parking_location", "应急车道")
        track_duration = v.get("track_duration", 0)

        if parking_loc == "行车道":
            risk_level = "P0"
            risk_desc = "高危"
            urgency = "立即"
        elif parking_loc == "应急车道":
            risk_level = "P1"
            risk_desc = "中危"
            urgency = "尽快"
        else:
            risk_level = "P2"
            risk_desc = "低危"
            urgency = "适时"

        return (
            f"风险等级：{risk_level}。"
            f"风险说明：{parking_loc}违停车辆属于{risk_desc}风险级别，违停时长约{track_duration}帧。"
            f"处理建议：{urgency}通知路政巡逻人员前往现场核实，发布预警信息提醒后方车辆注意避让。"
        )

    def _enhance_l2_list_with_rag(
        self, l2_list: List[str], violations: List[Dict]
    ) -> List[str]:
        """使用RAG知识库增强L2分析过程列表

        Args:
            l2_list: L2分析结果列表
            violations: 违停检测结果列表（用于获取车辆类型和位置信息）

        Returns:
            增强后的L2列表
        """
        if not self.use_rag or not l2_list:
            return l2_list

        enhanced_list = []
        for i, (l2_text, v) in enumerate(zip(l2_list, violations)):
            vehicle_type = v.get("vehicle_type", "车辆")
            parking_loc = v.get("parking_location", "应急车道")

            # 检索RAG知识
            rag_knowledge = ""
            if retrieve_knowledge_for_l2:
                try:
                    rag_knowledge = retrieve_knowledge_for_l2(vehicle_type, parking_loc)
                except Exception:
                    pass

            if not rag_knowledge:
                enhanced_list.append(l2_text)
                continue

            # 使用基类公共方法增强（调用VLM整合）
            rag_context = f"【{vehicle_type}在{parking_loc}】{rag_knowledge}"

            prompt = f"""请根据以下专业知识，对原有的分析结果进行整合和完善，输出更专业、更准确的L2分析结果。

原有分析结果：
{l2_text}

专业知识依据：
{rag_context}

要求：
1. 保持原有分析的核心内容（风险属性判断、场景解释能力、粗粒度归因）
2. 将专业知识自然融入，不要简单堆砌
3. 输出格式与原有格式一致
4. 不要添加多余的前缀或说明，直接输出整合后的结果

请直接输出整合后的L2分析结果："""

            try:
                response = self.vlm_client.chat(
                    prompt=prompt,
                    system_prompt="你是一位专业的公路养护专家，负责整合L2分析结果。",
                )
                enhanced = (
                    response.get("content", l2_text)
                    if isinstance(response, dict)
                    else str(response)
                )
                enhanced_list.append(enhanced)
                print(f"[Task2] RAG增强L2[{i}]完成")
            except Exception as e:
                print(f"[Task2] RAG增强L2[{i}]失败: {e}")
                enhanced_list.append(l2_text)

        return enhanced_list

    def _enhance_l3_list_with_rag(
        self, l3_list: List[str], violations: List[Dict]
    ) -> List[str]:
        """使用RAG知识库增强L3风险评估列表

        Args:
            l3_list: L3风险评估列表
            violations: 违停检测结果列表

        Returns:
            增强后的L3列表
        """
        if not self.use_rag or not l3_list:
            return l3_list

        enhanced_list = []
        for i, (l3_text, v) in enumerate(zip(l3_list, violations)):
            vehicle_type = v.get("vehicle_type", "车辆")
            parking_loc = v.get("parking_location", "应急车道")

            # 检索RAG知识
            rag_knowledge = ""
            if retrieve_knowledge_for_l3:
                try:
                    rag_knowledge = retrieve_knowledge_for_l3(
                        vehicle_type, parking_loc, ""
                    )
                except Exception:
                    pass

            if not rag_knowledge:
                enhanced_list.append(l3_text)
                continue

            # 使用VLM整合
            rag_context = f"【{vehicle_type}在{parking_loc}处置建议】{rag_knowledge}"

            prompt = f"""请根据以下专业知识，对原有的风险评估进行整合和完善，输出更专业、更准确的L3风险评估结果。

原有风险评估：
{l3_text}

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
                    system_prompt="你是一位专业的公路养护专家，负责整合L3风险评估。",
                )
                enhanced = (
                    response.get("content", l3_text)
                    if isinstance(response, dict)
                    else str(response)
                )
                enhanced_list.append(enhanced)
                print(f"[Task2] RAG增强L3[{i}]完成")
            except Exception as e:
                print(f"[Task2] RAG增强L3[{i}]失败: {e}")
                enhanced_list.append(l3_text)

        return enhanced_list

    def process(
        self, input_data: str, scene_id: int = 1, output_dir: str = None
    ) -> Dict[str, Any]:
        """处理车辆违停任务

        返回格式符合官方标准：
        - l1_result: 列表，每个元素包含 frame_id, bbox, vehicle_type, parking_location
        - l2_result: 列表，每个元素是对应违停车辆的分析过程（使用TASK2_USER_PROMPT增强）
        - l3_result: 列表，每个元素是对应违停车辆的风险评估（使用TASK2_USER_PROMPT增强）
        """
        try:
            tracker = self._get_tracker()

            output_video = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                video_name = Path(input_data).stem
                output_video = os.path.join(output_dir, f"{video_name}_annotated.mp4")

            violations, video_info = tracker.track_video(
                input_data, output_video, min_track_frames=None
            )

            width = video_info.get("width", 1920)
            height = video_info.get("height", 1080)

            # 按官方标准格式化输出：L1/L2/L3 为列表，一一对应
            l1_result = []
            l2_result = []
            l3_result = []

            # 无违停时的默认输出
            if len(violations) == 0:
                return {
                    "success": True,
                    "message": "",
                    "scene_id": scene_id,
                    "data": [],
                    "video_info": video_info,
                    "annotated_video": output_video
                    if output_video and os.path.exists(output_video)
                    else "",
                }

            # 提取关键帧并绘制标注框，用于VLM分析
            annotated_frame_path = None
            if output_dir:
                try:
                    annotated_frame_path = self._extract_and_annotate_frame(
                        input_data, violations, width, height, output_dir
                    )
                except Exception as e:
                    print(f"[Task2] 提取标注帧失败: {e}")

            for v in violations:
                # L1: 检测结果
                normalized_bbox = self._normalize_bbox(v["bbox"], width, height)
                l1_result.append(
                    {
                        "frame_id": v["frame_id"],
                        "bbox": normalized_bbox,
                        "vehicle_type": v.get("vehicle_type", "车辆"),
                        "parking_location": v.get("parking_location", "应急车道")
                    }
                )

            # 使用VLM分析标注图像生成L2/L3
            if annotated_frame_path and os.path.exists(annotated_frame_path):
                print(f"[Task2] 使用VLM分析标注图像，生成L2/L3结果...")
                l2_result, l3_result = self._generate_l2_l3_with_vlm_image(
                    annotated_frame_path, violations, video_info
                )
            else:
                # 回退：使用模板生成L2/L3
                print("[Task2] 使用模板生成L2/L3结果...")
                for v in violations:
                    l2_result.append(self._generate_l2_fallback(v))
                    l3_result.append(self._generate_l3_fallback(v))

            # RAG增强（对所有L2/L3结果进行专业知识增强）
            if self.use_rag and l2_result:
                print("[Task2] 执行RAG增强...")
                l2_result = self._enhance_l2_list_with_rag(l2_result, violations)
            if self.use_rag and l3_result:
                l3_result = self._enhance_l3_list_with_rag(l3_result, violations)

            unique_tracks = len(set(v["track_id"] for v in violations))
            print(
                f"[Task2] 检测到 {unique_tracks} 辆静止车辆（违停），共 {len(violations)} 条检测记录"
            )

            # 使用新格式输出（主办方要求格式）
            result = self.format_output_new(
                violations=violations,
                reasoning=l2_result,
                suggestion=l3_result,
                scene_id=scene_id,
            )
            # 保留 video_info 和 annotated_video 用于调试
            result["video_info"] = video_info
            result["annotated_video"] = (
                output_video if output_video and os.path.exists(output_video) else ""
            )

            return result

        except Exception as e:
            import traceback

            traceback.print_exc()
            return {
                "success": False,
                "message": str(e),
                "scene_id": scene_id,
                "data": [],
                "video_info": {"error": str(e)},
                "annotated_video": "",
            }

    def _extract_and_annotate_frame(
        self,
        video_path: str,
        violations: List[Dict],
        width: int,
        height: int,
        output_dir: str,
    ) -> Optional[str]:
        """从视频中提取关键帧并绘制标注框

        Args:
            video_path: 视频路径
            violations: 违停检测结果
            width: 视频宽度
            height: 视频高度
            output_dir: 输出目录

        Returns:
            标注后的帧图像路径
        """
        try:
            # 找到最早出现的违停帧
            first_violation = min(violations, key=lambda v: v["frame_id"])
            target_frame_id = first_violation["frame_id"]

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_id)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return None

            # 为所有在该帧或之后出现的违停车辆绘制红框
            for v in violations:
                x1, y1, x2, y2 = v["bbox"]
                # 绘制红色边框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # 添加标签
                label = f"ID:{v.get('track_id', '?')} {v.get('vehicle_type', '车辆')}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # 保存标注帧
            video_name = Path(video_path).stem
            annotated_frame_path = os.path.join(
                output_dir, f"{video_name}_annotated_frame.jpg"
            )
            cv2.imwrite(annotated_frame_path, frame)

            print(f"[Task2] 已提取标注帧: {annotated_frame_path}")
            return annotated_frame_path

        except Exception as e:
            print(f"[Task2] 提取标注帧失败: {e}")
            return None

    def normalize_response_for_task2(
        self, result: Dict[str, Any], scene_id: int
    ) -> Dict[str, Any]:
        """标准化Task2响应，确保符合官方列表格式"""
        if "l1_result" not in result:
            result["l1_result"] = []
        if "l2_result" not in result:
            result["l2_result"] = []
        if "l3_result" not in result:
            result["l3_result"] = []

        # 确保都是列表类型
        if not isinstance(result["l1_result"], list):
            result["l1_result"] = []
        if not isinstance(result["l2_result"], list):
            result["l2_result"] = []
        if not isinstance(result["l3_result"], list):
            result["l3_result"] = []

        return result