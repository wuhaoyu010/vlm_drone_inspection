"""
Task 2: 车辆违停检测处理器
YOLO + ByteTrack方案（使用ultralytics官方track接口）
支持轨迹绘制，可视化追踪路径
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


# 配置参数
MODEL_PATH = "yolov8n.pt"     # YOLO模型路径（yolov8n最快，s/m/l/x更准但更慢）
CONF_THRESH = 0.15            # 检测置信度阈值（降低以提高召回率）
IOU_THRESH = 0.7              # IOU阈值
MIN_TRACK_FRAMES = 10         # 最小追踪帧数（用于判断违停）
TRAJECTORY_LENGTH = 30        # 轨迹显示长度（帧数）

# 车辆类别（COCO数据集）
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


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
        raise ImportError("视频处理需要opencv-python，请安装: pip install opencv-python")


class VehicleTracker:
    """车辆追踪器 - YOLO + ByteTrack（官方接口）"""

    def __init__(self, model_path: str = MODEL_PATH):
        """初始化追踪器"""
        print(f"[Task2] 加载YOLO模型: {model_path}")
        self.model = YOLO(model_path)
        # 自动检测并使用GPU
        import torch
        if torch.cuda.is_available():
            self.model.to('cuda')
            print(f"[Task2] 使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[Task2] 使用CPU")
        self.model_name = model_path

    def track_video(
        self,
        video_path: str,
        output_video: str = None,
        min_track_frames: int = MIN_TRACK_FRAMES
    ) -> Tuple[List[Dict], Dict]:
        """
        使用YOLO track方法追踪视频中的车辆并检测违停

        Args:
            video_path: 视频路径
            output_video: 输出标注视频路径
            min_track_frames: 最小追踪帧数阈值（超过此帧数认为违停）

        Returns:
            violations: 违停检测结果
            video_info: 视频信息
        """
        # 获取视频信息
        width, height, total_frames, fps = get_video_info(video_path)

        video_info = {
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration": round(total_frames / fps, 2) if fps > 0 else 0
        }

        print(f"[Task2] 视频信息: {total_frames}帧, {fps:.1f}fps, {video_info['duration']:.1f}秒")

        # 追踪记录 {track_id: [frame_ids]}
        track_history = {}
        # 轨迹记录 {track_id: [(x, y), ...]} 用于绘制轨迹线
        trajectory_history = defaultdict(list)
        all_tracks = []

        # 使用YOLO的track方法进行视频追踪
        # persist=True 保持追踪器状态，tracker="bytetrack.yaml" 使用ByteTrack
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        results = self.model.track(
            source=video_path,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            classes=list(VEHICLE_CLASSES.keys()),  # 只检测车辆类别
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            save=False,
            stream=True,  # 流式处理，减少内存
            device=device  # 使用GPU加速
        )

        frame_id = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # 获取追踪ID
                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                    xyxy = boxes.xyxy.cpu().numpy()

                    frame_tracks = []
                    for i, track_id in enumerate(track_ids):
                        x1, y1, x2, y2 = xyxy[i]

                        # 边界检查
                        x1 = max(0, min(int(x1), width))
                        y1 = max(0, min(int(y1), height))
                        x2 = max(0, min(int(x2), width))
                        y2 = max(0, min(int(y2), height))

                        # 计算中心点用于轨迹绘制
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        # 记录追踪历史
                        if track_id not in track_history:
                            track_history[track_id] = []
                        track_history[track_id].append(frame_id)

                        # 记录轨迹点
                        trajectory_history[track_id].append((float(cx), float(cy)))
                        if len(trajectory_history[track_id]) > TRAJECTORY_LENGTH:
                            trajectory_history[track_id].pop(0)

                        frame_tracks.append({
                            "track_id": track_id,
                            "bbox": [x1, y1, x2, y2],
                            "center": (cx, cy)
                        })

                    all_tracks.append({
                        "frame_id": frame_id,
                        "tracks": frame_tracks
                    })

            frame_id += 1
            if frame_id % 100 == 0:
                print(f"[Task2] 已处理 {frame_id}/{total_frames} 帧")

        print(f"[Task2] 追踪完成: 共追踪 {len(track_history)} 个目标")

        # 分析违停
        violations = []
        for track_id, frames in track_history.items():
            track_duration = len(frames)
            # 如果追踪帧数超过阈值，认为是违停
            if track_duration >= min_track_frames:
                # 获取该track的所有bbox
                for ft in all_tracks:
                    for t in ft["tracks"]:
                        if t["track_id"] == track_id:
                            violations.append({
                                "frame_id": ft["frame_id"],
                                "bbox": t["bbox"],
                                "track_id": int(track_id),
                                "category": "违停车辆",
                                "vehicle_type": "车辆",
                                "parking_location": "应急车道",
                                "track_duration": track_duration,
                                "is_tracked": True
                            })
                            break

        # 生成标注视频
        if output_video and violations:
            print(f"[Task2] 生成标注视频...")
            self._generate_annotated_video(video_path, output_video, all_tracks, trajectory_history, fps, width, height)

        unique_tracks = len(set(v['track_id'] for v in violations))
        print(f"[Task2] 检测到 {unique_tracks} 个违停车辆")

        return violations, video_info

    def _generate_annotated_video(
        self,
        video_path: str,
        output_video: str,
        all_tracks: List[Dict],
        trajectory_history: Dict[int, List[Tuple[float, float]]],
        fps: float,
        width: int,
        height: int
    ):
        """生成带标注和轨迹的视频"""
        os.makedirs(os.path.dirname(output_video) if os.path.dirname(output_video) else '.', exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, int(fps), (width, height))

        # 构建帧索引
        frame_tracks_map = {ft["frame_id"]: ft["tracks"] for ft in all_tracks}

        # 计算违停ID集合
        violation_ids = set()
        for track_id, traj in trajectory_history.items():
            if len(traj) >= MIN_TRACK_FRAMES:
                violation_ids.add(track_id)

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 绘制该帧的追踪框和轨迹
            if frame_id in frame_tracks_map:
                for t in frame_tracks_map[frame_id]:
                    x1, y1, x2, y2 = t["bbox"]
                    track_id = t["track_id"]

                    # 判断是否违停（红色）或普通追踪（绿色）
                    is_violation = track_id in violation_ids
                    box_color = (0, 0, 255) if is_violation else (0, 255, 0)
                    text_color = (0, 0, 255) if is_violation else (0, 255, 0)

                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # 绘制ID标签
                    label = f"ID:{track_id}" + (" [违停]" if is_violation else "")
                    cv2.putText(frame, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                    # 绘制轨迹线（参考官方示例）
                    track = trajectory_history.get(track_id, [])
                    if len(track) > 1:
                        # 轨迹颜色：违停用红色，普通用青色
                        traj_color = (0, 0, 255) if is_violation else (255, 255, 0)
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=traj_color, thickness=2)

            video_writer.write(frame)
            frame_id += 1

        cap.release()
        video_writer.release()
        print(f"[Task2] 标注视频已保存: {output_video}")


class Task2Processor(BaseProcessor):
    """车辆违停检测处理器 - YOLO + ByteTrack（官方接口）"""

    def __init__(self, vlm_client):
        super().__init__(vlm_client)
        self.tracker = None  # 延迟初始化

    def _get_tracker(self):
        """延迟初始化追踪器"""
        if self.tracker is None:
            self.tracker = VehicleTracker()
        return self.tracker

    def process(self, input_data: str, scene_id: int = 1, output_dir: str = None) -> Dict[str, Any]:
        """
        处理车辆违停任务

        流程：
        1. YOLO检测车辆
        2. ByteTrack追踪车辆
        3. 分析追踪数据判断违停
        4. 生成标注视频
        """
        try:
            tracker = self._get_tracker()

            # 设置输出视频路径
            output_video = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                video_name = Path(input_data).stem
                output_video = os.path.join(output_dir, f"{video_name}_annotated.mp4")

            # 追踪视频
            violations, video_info = tracker.track_video(
                input_data,
                output_video,
                min_track_frames=MIN_TRACK_FRAMES
            )

            # 判断是否有违停
            has_violation = len(violations) > 0

            # 生成L2和L3结果
            if has_violation:
                unique_tracks = len(set(v['track_id'] for v in violations))
                l2_result = f"检测到{unique_tracks}辆违停车辆，已追踪标注"
                l3_result = "风险等级：P1。风险说明：检测到违停车辆，可能影响其他车辆通行。处理建议：通知相关部门进行处理。"
            else:
                l2_result = "未检测到违停车辆"
                l3_result = "风险等级：P2。风险说明：未检测到违停情况。处理建议：继续保持监控。"

            result = {
                "l1_result": {
                    "has_violation": has_violation,
                    "violations": violations,
                    "video_info": video_info,
                    "annotated_video": output_video if output_video and os.path.exists(output_video) else ""
                },
                "l2_result": l2_result,
                "l3_result": l3_result
            }

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "l1_result": {
                    "has_violation": False,
                    "violations": [],
                    "video_info": {"error": str(e)},
                    "annotated_video": ""
                },
                "l2_result": f"视频处理失败: {str(e)}",
                "l3_result": ""
            }

    def normalize_response_for_task2(self, result: Dict[str, Any], scene_id: int) -> Dict[str, Any]:
        """标准化Task2响应"""
        if "l1_result" not in result:
            result["l1_result"] = {
                "has_violation": False,
                "violations": [],
                "annotated_video": "",
                "video_info": {}
            }
        if "l2_result" not in result:
            result["l2_result"] = ""
        if "l3_result" not in result:
            result["l3_result"] = ""
        return result