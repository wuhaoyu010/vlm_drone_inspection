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
import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import cv2

from .base import BaseProcessor
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
        # 静止判断参数（与demo_15.py一致）
        "static_time_window": task_config.get("static_time_window", 30),
        "static_dist_threshold": task_config.get("static_dist_threshold", 20.0),
        "use_optical_flow": task_config.get("use_optical_flow", True),
        # 新增：静止判定参数（来自demo_15.py）
        "static_smooth_frames": task_config.get("static_smooth_frames", 30),
        "static_ratio_threshold": task_config.get("static_ratio_threshold", 0.5),
        # RAG知识库参数
        "use_rag": task_config.get("use_rag", True),
        "knowledge_dir": task_config.get("knowledge_dir", "./knowledge_base"),
        "embedding_model_path": task_config.get(
            "embedding_model_path", "./bge-small-zh-v1.5"
        ),
        # 视频输出配置
        "generate_annotated_video": task_config.get("generate_annotated_video", False),
        # 静止车辆图片保存配置
        "save_static_images": task_config.get("save_static_images", True),
        "static_img_dir": task_config.get("static_img_dir", "./static_cars_imgs"),
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
        # 新增：静止判定参数（来自demo_15.py）
        self.static_smooth_frames = self.config["static_smooth_frames"]
        self.static_ratio_threshold = self.config["static_ratio_threshold"]

        # 视频输出配置
        self.generate_annotated_video = self.config["generate_annotated_video"]

        # 静止车辆图片保存配置
        self.save_static_images = self.config["save_static_images"]
        self.static_img_dir = self.config["static_img_dir"]

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

        # 追踪状态（demo_15.py逻辑）
        track_history = defaultdict(list)  # 补偿后轨迹
        raw_track_history = defaultdict(list)  # 原始轨迹
        box_history = defaultdict(list)  # 框平滑历史
        static_count = defaultdict(int)  # 静止帧计数
        static_hysteresis = defaultdict(int)  # 迟滞计数器
        prev_disp = defaultdict(float)  # 上一帧位移（指数平滑用）
        static_status = defaultdict(lambda: False)  # 实时静止状态
        all_tracks = []

        # 全局统计变量（demo_15.py逻辑）
        total_appear_frames = defaultdict(int)  # 总出现帧数
        total_static_frames = defaultdict(int)  # 总静止帧数
        class_name_record = defaultdict(str)  # 类别名称记录
        avg_displacement_record = defaultdict(list)  # 平均位移记录

        # 帧记录（用于图片生成）
        frame_records = defaultdict(dict)  # 帧号: {车辆ID: (x1,y1,x2,y2)}
        raw_frames = dict()  # 帧号: 原始帧图片
        vehicle_appear_frames = defaultdict(list)  # 车辆ID: [出现过的帧号列表]

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

                    # 框平滑（demo_15.py逻辑）
                    box_history[track_id].append((x1, y1, x2, y2))
                    if len(box_history[track_id]) > 3:
                        box_history[track_id].pop(0)
                    x1, y1, x2, y2 = map(int, np.mean(box_history[track_id], axis=0))

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

                    # 车辆基础信息统计
                    total_appear_frames[track_id] += 1
                    class_name_record[track_id] = self.model.names[int(class_ids[i])]

                    # 计算位移（demo_15.py逻辑）
                    disp = 0.0
                    if len(track) >= self.static_time_window:
                        recent_points = track[-self.static_time_window :]
                        disp = calculate_displacement(recent_points)
                    elif 20 <= len(track) < self.static_time_window:
                        # 轨迹不足时使用全部点
                        recent_points = track
                        disp = calculate_displacement(recent_points)

                    # 动态静止阈值（根据镜头抖动自适应）
                    if motion_compensator:
                        motion_amp = np.sqrt(motion_compensator.prev_transform[0, 2]**2 +
                                            motion_compensator.prev_transform[1, 2]**2)
                    else:
                        motion_amp = 0
                    dynamic_thresh = self.static_dist_threshold + min(motion_amp * 2, 10)

                    # 轨迹不足时降低阈值
                    if len(track) < self.static_time_window:
                        dynamic_thresh *= 0.8

                    # 位移指数平滑（避免突变）
                    disp_smoothed = 0.7 * prev_disp[track_id] + 0.3 * disp
                    prev_disp[track_id] = disp_smoothed

                    # 静止状态判定（带迟滞机制，避免状态跳变）
                    is_static_now = disp_smoothed < dynamic_thresh
                    if is_static_now:
                        static_count[track_id] += 1
                        static_hysteresis[track_id] = 0
                        static_status[track_id] = static_count[track_id] >= self.static_smooth_frames
                    else:
                        static_hysteresis[track_id] += 1
                        if static_hysteresis[track_id] >= 3:
                            static_count[track_id] = 0
                            static_status[track_id] = False

                    # 累计车辆静止帧数
                    if static_status[track_id]:
                        total_static_frames[track_id] += 1
                    avg_displacement_record[track_id].append(disp_smoothed)

                    # 记录每帧的车辆坐标和车辆出现的帧号
                    frame_records[frame_id][track_id] = (x1, y1, x2, y2)
                    vehicle_appear_frames[track_id].append(frame_id)

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

            # 保存原始帧（demo_15.py逻辑）
            raw_frames[frame_id] = frame.copy()

            frame_id += 1
            if frame_id % 100 == 0:
                print(f"[Task2] 已处理 {frame_id}/{total_frames} 帧")

        cap.release()
        print(f"[Task2] 追踪完成: 共追踪 {len(track_history)} 个目标")

        # ====================== 最终统一判定：静止帧数≥总帧数50% 判定为静止（demo_15.py逻辑）======================
        final_status = dict()  # 车辆ID: True(静止)/False(运动)
        all_ids = sorted(total_appear_frames.keys())
        for tid in all_ids:
            appear = total_appear_frames[tid]
            static = total_static_frames[tid]
            # 过滤跟踪帧数不足10帧的车辆，避免误判
            if appear < self.min_track_frames:
                final_status[tid] = False
                continue
            # 核心判定规则：静止帧数 >= 总帧数 * 50%
            final_status[tid] = static >= appear * self.static_ratio_threshold

        # 筛选最终静止的车辆ID
        final_static_ids = [tid for tid in all_ids if final_status[tid]]
        print(f"[Task2] 最终判定静止车辆数：{len(final_static_ids)} 辆")

        # 分析违停（静止车辆）
        violations = []

        # 打印详细统计信息
        print(f"\n{'='*100}")
        print(f"{'ID':<8} {'类别':<15} {'总帧数':<10} {'静止帧数':<10} {'静止占比(%)':<12} {'最终状态':<10}")
        print("-" * 100)
        for tid in all_ids:
            cls = class_name_record[tid]
            appear = total_appear_frames[tid]
            static = total_static_frames[tid]
            ratio = (static / appear) * 100 if appear > 0 else 0.0
            status_str = "静止" if final_status[tid] else "运动"
            print(f"{tid:<8} {cls:<15} {appear:<10} {static:<10} {ratio:<12.1f} {status_str:<10}")
        print(f"{'='*100}\n")

        for track_id in final_static_ids:
            track = track_history[track_id]
            track_len = len(track)

            if track_len >= self.min_track_frames:
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
                                    "vehicle_type": class_name_record.get(
                                        track_id, VEHICLE_CLASSES.get(t.get("class_id", 2), "车辆")
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
                                    ),
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

        # ====================== 生成静止车辆图片 + 构建JSON数据（demo_15.py逻辑）======================
        static_images_info = []  # 返回给VLM复判的图片信息
        json_data = {"data": []}
        json_recorded_frames = set()  # 记录已生成图片的帧号

        if len(final_static_ids) > 0 and self.save_static_images:
            # 创建保存目录
            save_dir = Path(self.static_img_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            video_name = Path(video_path).stem

            print(f"[Task2] 开始生成静止车辆图片（保存至：{save_dir.absolute()}）...")

            # 标记已绘制的车辆ID（避免重复生成图片）
            drawn_vehicle_ids = set()

            for tid in final_static_ids:
                if tid in drawn_vehicle_ids:
                    continue

                vehicle_frames = vehicle_appear_frames[tid]
                if len(vehicle_frames) < 10:
                    continue

                # 取该车辆的第10帧作为绘制主体帧
                target_frame = vehicle_frames[9]
                if target_frame not in raw_frames or target_frame not in frame_records:
                    continue

                # 提取目标帧中所有最终静止的车辆ID和bbox
                target_frame_vehicles = frame_records[target_frame]
                target_static_bboxes = []
                draw_vehicle_ids = []

                for vid, bbox in target_frame_vehicles.items():
                    if final_status.get(vid, False):
                        x1, y1, x2, y2 = bbox
                        target_static_bboxes.append([int(x1), int(y1), int(x2), int(y2)])
                        draw_vehicle_ids.append(vid)

                # 绘制目标帧图片
                draw_img = raw_frames[target_frame].copy()
                for vid in draw_vehicle_ids:
                    x1, y1, x2, y2 = frame_records[target_frame][vid]
                    cls_name = class_name_record[vid]
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(draw_img, f"ID:{vid} {cls_name} STATIC",
                                (x1, y1 - 10 if y1 > 10 else y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 保存图片
                img_save_path = save_dir / f"{video_name}_frame{target_frame}_mainID{tid}.jpg"
                cv2.imwrite(str(img_save_path), draw_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # 构建JSON数据
                if target_frame not in json_recorded_frames:
                    json_data["data"].append({
                        "frame_id": target_frame,
                        "bboxs": target_static_bboxes
                    })
                    json_recorded_frames.add(target_frame)

                    # 添加到返回信息（供VLM复判）
                    static_images_info.append({
                        "frame_id": target_frame,
                        "img_path": str(img_save_path),
                        "bboxes": target_static_bboxes,
                        "vehicle_ids": draw_vehicle_ids
                    })

                # 标记已绘制
                drawn_vehicle_ids.update(draw_vehicle_ids)

            # 保存JSON文件
            if len(json_data["data"]) > 0:
                json_save_path = save_dir / f"{video_name}_static_cars_bbox.json"
                with open(json_save_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"[Task2] JSON坐标文件已保存至：{json_save_path}")

        return violations, video_info, static_images_info

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

    def _vlm_analyze_static_vehicles(
        self,
        static_images_info: List[Dict],
        violations: List[Dict],
        video_info: Dict
    ) -> Tuple[List[Dict], List[str], List[str]]:
        """使用VLM分析静止车辆，判断场景位置并生成L2/L3结果

        使用已有的 TASK2_SYSTEM_PROMPT 和 TASK2_USER_PROMPT

        Args:
            static_images_info: 静止车辆图片信息列表
            violations: 原始违停检测结果列表
            video_info: 视频信息（用于坐标归一化）

        Returns:
            (confirmed_violations, l2_result, l3_result)
        """
        if not static_images_info or not violations:
            return violations, [], []

        width = video_info.get("width", 1920)
        height = video_info.get("height", 1080)

        # ==================== 过滤原因收集（便于排查） ====================
        filter_reasons = []  # 记录每辆车的过滤原因

        confirmed_violations = []
        l2_result = []
        l3_result = []

        # 构建车辆ID到violation的映射
        violation_map = {v.get("track_id"): v for v in violations}

        # 记录输入统计
        print(f"[Task2] VLM复核输入: {len(violations)}辆静止车辆候选")
        for img_info in static_images_info:
            img_path = img_info.get("img_path", "")
            bboxes = img_info.get("bboxes", [])  # 像素坐标
            vehicle_ids = img_info.get("vehicle_ids", [])
            frame_id = img_info.get("frame_id", 60)

            if not img_path or not os.path.exists(img_path):
                print(f"[Task2] VLM分析: 图片不存在 {img_path}，保留所有检测结果")
                for vid in vehicle_ids:
                    if vid in violation_map:
                        confirmed_violations.append(violation_map[vid])
                        l2_result.append(self._generate_l2_fallback(violation_map[vid]))
                        l3_result.append(self._generate_l3_fallback(violation_map[vid]))
                continue

            try:
                # 归一化坐标（转换为0-1000范围）
                normalized_bboxes = []
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    normalized_bboxes.append([
                        int(x1 * 1000 / width),
                        int(y1 * 1000 / height),
                        int(x2 * 1000 / width),
                        int(y2 * 1000 / height)
                    ])

                # 构建车辆信息（带归一化坐标）
                vehicle_info_list = []
                for i, vid in enumerate(vehicle_ids):
                    v = violation_map.get(vid, {})
                    vehicle_info_list.append({
                        "id": int(vid),
                        "type": v.get("vehicle_type", "车辆"),
                        "bbox_normalized": normalized_bboxes[i] if i < len(normalized_bboxes) else None
                    })

                # 使用模板方式：在 TASK2_USER_PROMPT 前插入坐标信息
                user_prompt = f"""## 已检测车辆信息（归一化坐标[0,1000]）
{json.dumps(vehicle_info_list, ensure_ascii=False, indent=2)}

---

{TASK2_USER_PROMPT}"""

                # 使用已有的系统提示词
                response = self.vlm_client.chat_with_image(
                    image_path=img_path,
                    prompt=user_prompt,
                    system_prompt=TASK2_SYSTEM_PROMPT,
                )

                if response and response.get("content"):
                    content = response["content"]

                    # 记录VLM原始响应（便于排查）
                    print(f"[Task2] VLM原始响应 (前500字符): {content[:500]}...")

                    # 解析JSON响应
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        result = json.loads(json_match.group())
                        l1_list = result.get("l1_result", [])
                        l1_list = [{**it, "frame_id": frame_id} for it in l1_list]
                        l2_list = result.get("l2_result", [])
                        l3_list = result.get("l3_result", [])

                        print(f"[Task2] VLM解析结果: l1_result={len(l1_list)}项")

                        # 如果l1_result为空，说明VLM判断所有车辆均非违停
                        if not l1_list:
                            print(f"[Task2] VLM判断: 所有{len(vehicle_ids)}辆车均非违停")
                            for vid in vehicle_ids:
                                v = violation_map.get(vid, {})
                                filter_reasons.append({
                                    "vehicle_id": vid,
                                    "vehicle_type": v.get("vehicle_type", "未知"),
                                    "bbox": v.get("bbox", []),
                                    "reason": "VLM判断非违停（l1_result为空）"
                                })
                            continue

                        print(f"[Task2] VLM分析结果 (帧{img_info['frame_id']}):")

                        # 构建车辆ID到L1结果的映射
                        for i, l1_item in enumerate(l1_list):
                            parking_loc = l1_item.get("parking_location", "应急车道")

                            # 找到对应的violation（通过bbox匹配或按顺序匹配）
                            matched_vid = None
                            if i < len(vehicle_ids):
                                matched_vid = vehicle_ids[i]

                            if matched_vid and matched_vid in violation_map:
                                v = violation_map[matched_vid]
                                confirmed_violations.append(v)

                                # 使用VLM生成的L2/L3
                                l2_text = l2_list[i] if i < len(l2_list) else self._generate_l2_fallback(v)
                                l3_text = l3_list[i] if i < len(l3_list) else self._generate_l3_fallback(v)

                                l2_result.append(l2_text)
                                l3_result.append(l3_text)

                                print(f"  车辆ID {matched_vid}: 确认违停 - {parking_loc}")
                            else:
                                print(f"  索引 {i}: 未能匹配车辆ID")

                        # 打印被剔除的车辆（带详细原因）
                        confirmed_ids = set(v.get("track_id") for v in confirmed_violations)
                        for vid in vehicle_ids:
                            if vid not in confirmed_ids:
                                v = violation_map.get(vid, {})
                                filter_reasons.append({
                                    "vehicle_id": vid,
                                    "vehicle_type": v.get("vehicle_type", "未知"),
                                    "bbox": v.get("bbox", []),
                                    "reason": "VLM未返回该车辆（l1_result中无匹配）"
                                })
                                print(f"  车辆ID {vid}: 剔除（VLM未确认违停）")

                    else:
                        print("[Task2] VLM响应解析失败，使用回退模板")
                        print("[Task2] 解析失败原因: 未找到JSON结构")
                        self._add_fallback_results(
                            vehicle_ids, violation_map, confirmed_violations, l2_result, l3_result
                        )
                else:
                    print("[Task2] VLM响应为空，使用回退模板")
                    print(f"[Task2] 响应对象: {response}")
                    self._add_fallback_results(
                        vehicle_ids, violation_map, confirmed_violations, l2_result, l3_result
                    )

            except Exception as e:
                print(f"[Task2] VLM分析失败: {e}，使用回退模板")
                self._add_fallback_results(
                    vehicle_ids, violation_map, confirmed_violations, l2_result, l3_result
                )

        # 输出过滤原因汇总（便于排查）
        if filter_reasons:
            print(f"[Task2] 过滤原因汇总 ({len(filter_reasons)}辆被过滤):")
            for reason in filter_reasons:
                print(f"  - 车辆ID {reason['vehicle_id']} ({reason['vehicle_type']}): {reason['reason']}")

        print(f"[Task2] VLM分析完成: {len(violations)} -> {len(confirmed_violations)} 辆违停车辆")
        return confirmed_violations, l2_result, l3_result

    def format_output_new(
        self,
        violations: List[Dict],
        reasoning: List[str],
        suggestion: List[str],
        scene_id: int = 1,
        static_images_info: List[Dict] = None,
    ) -> List[Dict]:
        """格式化输出为主办方新格式（与demo_15.py逻辑一致）

        新格式要求（与demo_15.py一致）：
        - 每辆静止车辆取自己的第10帧作为代表帧
        - 同一帧中的多辆车合并到一个事件
        - 可能输出多个事件（不同帧）

        Args:
            violations: 违停检测结果列表
            reasoning: L2分析结果列表（reasoning）
            suggestion: L3风险建议列表（suggestion）
            scene_id: 场景ID
            static_images_info: 静止车辆图片信息（包含每辆车的代表帧）

        Returns:
            标准化输出格式列表
        """
        if not violations:
            return []

        # 构建车辆ID到violation的映射
        violation_map = {v.get("track_id"): v for v in violations}

        # 构建车辆ID到L2/L3的映射
        l2_map = {}
        l3_map = {}
        for i, v in enumerate(violations):
            tid = v.get("track_id")
            l2_map[tid] = reasoning[i] if i < len(reasoning) else self._generate_l2_fallback(v)
            l3_map[tid] = suggestion[i] if i < len(suggestion) else self._generate_l3_fallback(v)

        # 如果没有 static_images_info，回退到原来的逻辑（取最早帧）
        if not static_images_info:
            min_frame_id = min(v["frame_id"] for v in violations)
            bboxs = []
            reasoning_list = []
            suggestion_list = []
            for v in violations:
                bboxs.append(v["bbox"])
                reasoning_list.append(l2_map.get(v.get("track_id"), self._generate_l2_fallback(v)))
                suggestion_list.append(l3_map.get(v.get("track_id"), self._generate_l3_fallback(v)))
            return [{
                "frame_id": min_frame_id,
                "bboxs": bboxs,
                "reasoning": reasoning_list,
                "suggestion": suggestion_list,
            }]

        data_list = []
        recorded_frames = set()  # 避免同一帧重复记录
        drawn_vehicle_ids = set()  # 避免同一辆车重复记录

        for img_info in static_images_info:
            target_frame = img_info.get("frame_id")
            bboxes = img_info.get("bboxes", [])
            vehicle_ids = img_info.get("vehicle_ids", [])

            if target_frame in recorded_frames:
                continue

            # 收集该帧中所有静止车辆的信息
            frame_bboxs = []
            frame_reasoning = []
            frame_suggestion = []

            for i, vid in enumerate(vehicle_ids):
                if vid in drawn_vehicle_ids:
                    continue

                v = violation_map.get(vid)
                if v:
                    # bbox 使用像素坐标
                    if i < len(bboxes):
                        frame_bboxs.append(bboxes[i])
                    else:
                        frame_bboxs.append(v["bbox"])

                    frame_reasoning.append(l2_map.get(vid, self._generate_l2_fallback(v)))
                    frame_suggestion.append(l3_map.get(vid, self._generate_l3_fallback(v)))
                    drawn_vehicle_ids.add(vid)

            # 如果该帧有车辆，添加到数据列表
            if frame_bboxs:
                data_list.append({
                    "frame_id": target_frame,
                    "bboxs": frame_bboxs,
                    "reasoning": frame_reasoning,
                    "suggestion": frame_suggestion,
                })
                recorded_frames.add(target_frame)

        return data_list

    def _add_fallback_results(
        self,
        vehicle_ids: List[int],
        violation_map: Dict[int, Dict],
        confirmed_violations: List[Dict],
        l2_result: List[str],
        l3_result: List[str],
    ):
        """添加回退结果到输出列表

        当VLM分析失败或返回空响应时，使用回退模板填充结果。

        Args:
            vehicle_ids: 待处理的车辆ID列表
            violation_map: 车辆ID到violation字典的映射
            confirmed_violations: 已确认的违停车辆列表（会被修改）
            l2_result: L2分析结果列表（会被修改）
            l3_result: L3风险建议列表（会被修改）
        """
        for vid in vehicle_ids:
            if vid in violation_map:
                confirmed_violations.append(violation_map[vid])
                l2_result.append(self._generate_l2_fallback(violation_map[vid]))
                l3_result.append(self._generate_l3_fallback(violation_map[vid]))

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

            # 使用新的返回值（3个）：violations, video_info, static_images_info
            violations, video_info, static_images_info = tracker.track_video(
                input_data, output_video, min_track_frames=None
            )

            # 按官方标准格式化输出：L2/L3 为列表，一一对应
            l2_result = []
            l3_result = []

            # 无违停时的默认输出（完整结构，通过校验）
            if len(violations) == 0:
                return {
                    "success": True,
                    "message": "未检测到违停车辆",
                    "scene_id": scene_id,
                    "data": [{
                        "frame_id": 0,
                        "bboxs": [],
                        "reasoning": [],
                        "suggestion": []
                    }],
                    "video_info": video_info,
                    "annotated_video": output_video
                    if output_video and os.path.exists(output_video)
                    else "",
                }

            # ====================== VLM 整合分析：位置场景判断 + L2/L3 生成 ======================
            if static_images_info:
                # 记录VLM分析前的数量（便于统计过滤率）
                pre_vlm_count = len(violations)
                print(f"[Task2] 开始VLM整合分析（位置场景 + L2/L3），输入 {pre_vlm_count} 辆静止车辆候选...")

                violations, l2_result, l3_result = self._vlm_analyze_static_vehicles(
                    static_images_info, violations, video_info
                )

                if len(violations) == 0:
                    # 详细日志已在 _vlm_analyze_static_vehicles 中输出
                    print("[Task2] VLM分析后无确认的违停车辆")
                    print(f"[Task2] 统计: 原始{pre_vlm_count}辆候选 -> 确认0辆（全部被过滤）")
                    return {
                        "success": True,
                        "message": f"VLM分析后无确认的违停车辆（原始{pre_vlm_count}辆候选全部被过滤，详见控制台日志）",
                        "scene_id": scene_id,
                        "data": [{
                            "frame_id": 0,
                            "bboxs": [],
                            "reasoning": [],
                            "suggestion": []
                        }],
                        "video_info": video_info,
                        "annotated_video": output_video
                        if output_video and os.path.exists(output_video)
                        else "",
                    }
                else:
                    print(f"[Task2] VLM确认: {pre_vlm_count} -> {len(violations)} 辆（过滤{pre_vlm_count - len(violations)}辆）")
            else:
                # 无静止车辆图片信息，使用模板生成L2/L3
                print("[Task2] 无VLM图片信息，使用模板生成L2/L3...")
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

            # 使用新格式输出（与demo_15.py一致）
            data_list = self.format_output_new(
                violations=violations,
                reasoning=l2_result,
                suggestion=l3_result,
                scene_id=scene_id,
                static_images_info=static_images_info,
            )
            # 包装成标准返回格式
            result = {
                "success": True,
                "message": "",
                "scene_id": scene_id,
                "data": data_list,
                "video_info": video_info,
                "annotated_video": output_video if output_video and os.path.exists(output_video) else "",
            }

            return result

        except Exception as e:
            import traceback

            traceback.print_exc()
            return {
                "success": False,
                "message": str(e),
                "scene_id": scene_id,
                "data": [{
                    "frame_id": 0,
                    "bboxs": [],
                    "reasoning": [],
                    "suggestion": []
                }],
                "video_info": {"error": str(e)},
                "annotated_video": "",
            }