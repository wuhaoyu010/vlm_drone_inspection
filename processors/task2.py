"""
Task 2: 车辆违停检测处理器
使用 YOLO + ByteTrack 实现实时车辆追踪
VLM 用于违停判定
Windows兼容版本
"""
import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

from .base import BaseProcessor, denormalize_bbox
from prompts import TASK2_SYSTEM_PROMPT, TASK2_USER_PROMPT


def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """获取视频信息"""
    try:
        import cv2
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


class ByteTracker:
    """
    简化版 ByteTrack 追踪器
    基于 IoU 匹配实现多目标追踪
    """

    def __init__(self, iou_threshold: float = 0.3, min_hits: int = 3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_age = max_age
        self.tracks = {}  # track_id -> track_info
        self.next_id = 1
        self.frame_count = 0

    def iou(self, box1, box2):
        """计算两个框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        更新追踪状态

        Args:
            detections: 当前帧检测结果 [{"bbox": [x1,y1,x2,y2], "conf": 0.9, "cls": 0}]

        Returns:
            追踪结果 [{"track_id": int, "bbox": [...], "hits": int}]
        """
        self.frame_count += 1

        # 匹配阶段
        matched = {}
        unmatched_dets = list(range(len(detections)))

        for track_id, track in list(self.tracks.items()):
            if track["age"] > self.max_age:
                del self.tracks[track_id]
                continue

            best_iou = 0
            best_det_idx = None

            for det_idx in unmatched_dets:
                det = detections[det_idx]
                iou_val = self.iou(track["bbox"], det["bbox"])
                if iou_val > best_iou and iou_val > self.iou_threshold:
                    best_iou = iou_val
                    best_det_idx = det_idx

            if best_det_idx is not None:
                # 更新轨迹
                det = detections[best_det_idx]
                self.tracks[track_id]["bbox"] = det["bbox"]
                self.tracks[track_id]["hits"] += 1
                self.tracks[track_id]["age"] = 0
                self.tracks[track_id]["cls"] = det.get("cls", 0)
                matched[track_id] = best_det_idx
                unmatched_dets.remove(best_det_idx)

        # 更新未匹配轨迹的age
        for track_id in self.tracks:
            if track_id not in matched:
                self.tracks[track_id]["age"] += 1

        # 创建新轨迹
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self.tracks[self.next_id] = {
                "bbox": det["bbox"],
                "hits": 1,
                "age": 0,
                "cls": det.get("cls", 0),
                "start_frame": self.frame_count
            }
            self.next_id += 1

        # 返回确认的轨迹
        results = []
        for track_id, track in self.tracks.items():
            if track["hits"] >= self.min_hits and track["age"] <= self.max_age:
                results.append({
                    "track_id": track_id,
                    "bbox": track["bbox"],
                    "hits": track["hits"],
                    "age": track["age"],
                    "cls": track.get("cls", 0)
                })

        return results


class Task2Processor(BaseProcessor):
    """车辆违停检测处理器 - YOLO + ByteTrack + VLM"""

    def __init__(self, vlm_client):
        super().__init__(vlm_client)
        self.yolo_model = None
        self._init_yolo()

    def _init_yolo(self):
        """初始化YOLO模型"""
        try:
            from ultralytics import YOLO
            # 使用yolov8n模型（轻量级，自动下载）
            model_path = "yolov8n.pt"
            self.yolo_model = YOLO(model_path)
            print("[Task2] YOLO模型加载成功")
        except Exception as e:
            print(f"[Task2] YOLO模型加载失败: {e}")
            self.yolo_model = None

    def detect_vehicles(self, frame) -> List[Dict]:
        """
        使用YOLO检测车辆

        Returns:
            车辆检测结果列表
        """
        if self.yolo_model is None:
            return []

        # YOLO类别: 2=car, 3=motorcycle, 5=bus, 7=truck
        vehicle_classes = [2, 3, 5, 7]

        results = self.yolo_model(frame, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                if cls in vehicle_classes:
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i])
                    detections.append({
                        "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        "conf": conf,
                        "cls": cls
                    })

        return detections

    def get_vehicle_type(self, cls: int) -> str:
        """根据YOLO类别获取车辆类型"""
        vehicle_types = {
            2: "小型轿车",
            3: "摩托车",
            5: "客车",
            7: "货车"
        }
        return vehicle_types.get(cls, "车辆")

    def is_stationary(self, track_history: List[Tuple[int, List[int]]], fps: float, threshold_seconds: float = 2.0) -> bool:
        """
        判断车辆是否静止（违停）

        Args:
            track_history: [(frame_id, bbox), ...]
            fps: 帧率
            threshold_seconds: 静止阈值（秒）

        Returns:
            是否静止
        """
        if len(track_history) < fps * threshold_seconds:
            return False

        # 检查最近N帧的位移
        recent = track_history[-int(fps * threshold_seconds):]

        # 计算中心点
        centers = []
        for _, bbox in recent:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append((cx, cy))

        # 计算位移
        if len(centers) < 2:
            return False

        total_displacement = 0
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            total_displacement += (dx**2 + dy**2) ** 0.5

        avg_displacement = total_displacement / (len(centers) - 1)

        # 位移小于阈值则认为静止
        return avg_displacement < 5.0  # 像素

    def process(self, input_data: str, scene_id: int = 1, output_dir: str = None) -> Dict[str, Any]:
        """
        处理车辆违停任务

        流程：
        1. YOLO检测视频中的所有车辆
        2. ByteTrack追踪车辆
        3. 识别静止车辆（潜在违停）
        4. VLM验证是否为违停

        Args:
            input_data: 视频路径
            scene_id: 场景ID
            output_dir: 输出目录

        Returns:
            检测结果
        """
        # 获取视频信息
        try:
            width, height, total_frames, fps = get_video_info(input_data)
            video_info = {
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration": round(total_frames / fps, 2) if fps > 0 else 0
            }
        except Exception as e:
            video_info = {"error": str(e)}
            return {
                "l1_result": {"has_violation": False, "violations": [], "video_info": video_info, "annotated_video": ""},
                "l2_result": f"视频处理失败: {str(e)}",
                "l3_result": ""
            }

        print(f"[Task2] 视频信息: {total_frames}帧, {fps:.1f}fps, {video_info['duration']:.1f}秒")

        all_violations = []
        frame_results = []

        try:
            import cv2

            cap = cv2.VideoCapture(input_data)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {input_data}")

            # 初始化追踪器
            tracker = ByteTracker(iou_threshold=0.3, min_hits=3, max_age=int(fps * 2))

            # 存储轨迹
            track_trajectories = defaultdict(list)  # track_id -> [(frame_id, bbox)]
            stationary_tracks = set()  # 静止车辆的track_id

            frame_count = 0
            all_detections_for_video = []

            print("[Task2] 开始追踪...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO检测车辆
                detections = self.detect_vehicles(frame)

                # ByteTrack追踪
                tracks = tracker.update(detections)

                # 记录轨迹
                for track in tracks:
                    track_id = track["track_id"]
                    bbox = track["bbox"]
                    track_trajectories[track_id].append((frame_count, bbox.copy()))

                    # 判断是否静止
                    if len(track_trajectories[track_id]) > fps * 2:  # 超过2秒的轨迹
                        if self.is_stationary(track_trajectories[track_id], fps):
                            stationary_tracks.add(track_id)

                    # 记录检测
                    all_detections_for_video.append({
                        "frame_id": frame_count,
                        "track_id": track_id,
                        "bbox": bbox,
                        "cls": track.get("cls", 2),
                        "is_stationary": track_id in stationary_tracks
                    })

                frame_count += 1

                # 进度显示
                if frame_count % 100 == 0:
                    print(f"[Task2] 已处理 {frame_count}/{total_frames} 帧, 追踪{len(tracker.tracks)}个目标, 静止{len(stationary_tracks)}个")

            cap.release()

            print(f"[Task2] 追踪完成: {len(tracker.tracks)}个目标, {len(stationary_tracks)}个静止")

            # Step 2: 对静止车辆进行VLM验证
            if stationary_tracks:
                print(f"[Task2] 对 {len(stationary_tracks)} 个静止车辆进行VLM验证...")

                # 采样关键帧用于VLM验证
                sample_interval = max(1, total_frames // 5)  # 采样5帧
                sample_frames = list(range(0, total_frames, sample_interval))[:5]

                for sample_frame in sample_frames:
                    # 提取该帧的静止车辆
                    frame_stationary = [d for d in all_detections_for_video
                                       if d["frame_id"] == sample_frame and d["track_id"] in stationary_tracks]

                    if not frame_stationary:
                        continue

                    # 读取帧
                    cap = cv2.VideoCapture(input_data)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
                    ret, frame = cap.read()
                    cap.release()

                    if not ret:
                        continue

                    # 保存临时文件
                    temp_dir = tempfile.mkdtemp()
                    temp_frame_path = os.path.join(temp_dir, f"frame_{sample_frame}.jpg")
                    cv2.imwrite(temp_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

                    try:
                        response = self.vlm_client.chat_with_image(
                            image_path=temp_frame_path,
                            prompt=TASK2_USER_PROMPT,
                            system_prompt=TASK2_SYSTEM_PROMPT
                        )

                        result = self.parse_json_response(response)
                        l1_result = result.get("l1_result", {})

                        if isinstance(l1_result, dict) and l1_result.get("has_violation"):
                            violations = l1_result.get("violations", [])
                            for v in violations:
                                if width and height:
                                    v["bbox"] = denormalize_bbox(v.get("bbox", []), width, height)
                                v["frame_id"] = sample_frame
                                v["category"] = v.get("vehicle_type", "违停车辆")
                                all_violations.append(v)

                            frame_results.append({
                                "frame_id": sample_frame,
                                "has_violation": True,
                                "l2_result": result.get("l2_result", "")
                            })
                        else:
                            frame_results.append({
                                "frame_id": sample_frame,
                                "has_violation": False
                            })

                    except Exception as e:
                        frame_results.append({"frame_id": sample_frame, "error": str(e)})

                    finally:
                        try:
                            os.remove(temp_frame_path)
                            os.rmdir(temp_dir)
                        except:
                            pass

            else:
                print("[Task2] 未检测到静止车辆")

            # Step 3: 如果VLM确认有违停，为所有静止帧生成标注
            if all_violations:
                # 找到确认违停的track
                violation_tracks = set()
                for v in all_violations:
                    # 根据bbox找到对应的track
                    for d in all_detections_for_video:
                        if d["frame_id"] == v.get("frame_id"):
                            # 检查bbox是否匹配
                            vb = v.get("bbox", [])
                            db = d.get("bbox", [])
                            if len(vb) == 4 and len(db) == 4:
                                iou = self._calc_iou(vb, db)
                                if iou > 0.5:
                                    violation_tracks.add(d["track_id"])

                # 为这些track的所有帧生成violation
                final_violations = []
                for d in all_detections_for_video:
                    if d["track_id"] in violation_tracks or d["is_stationary"]:
                        final_violations.append({
                            "frame_id": d["frame_id"],
                            "bbox": d["bbox"],
                            "category": self.get_vehicle_type(d.get("cls", 2)),
                            "track_id": d["track_id"]
                        })

                all_violations = final_violations

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "l1_result": {"has_violation": False, "violations": [], "video_info": video_info, "annotated_video": "", "error": str(e)},
                "l2_result": f"视频处理失败: {str(e)}",
                "l3_result": ""
            }

        # 汇总结果
        has_violation = len(all_violations) > 0

        l2_result = "未检测到违停车辆"
        if has_violation:
            unique_tracks = len(set(v.get("track_id", 0) for v in all_violations if "track_id" in v))
            l2_result = f"通过YOLO+ByteTrack追踪，检测到{unique_tracks}个违停目标，共{len(all_violations)}帧"

            l2_parts = [fr.get("l2_result", "") for fr in frame_results if fr.get("l2_result")]
            if l2_parts:
                l2_result = l2_parts[0]

        if has_violation:
            l3_result = "风险等级：P1。风险说明：检测到违停车辆，可能影响其他车辆通行。处理建议：通知相关部门进行处理。"
        else:
            l3_result = "风险等级：P2。风险说明：未检测到违停情况。处理建议：继续保持监控。"

        result = {
            "l1_result": {
                "has_violation": has_violation,
                "violations": all_violations,
                "video_info": video_info,
                "frame_analysis": frame_results
            },
            "l2_result": l2_result,
            "l3_result": l3_result
        }

        # 创建标注视频
        if has_violation and all_violations:
            try:
                video_name = Path(input_data).stem
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_video = os.path.join(output_dir, f"{video_name}_annotated.mp4")
                else:
                    video_dir = os.path.dirname(input_data)
                    output_video = os.path.join(video_dir, f"{video_name}_annotated.mp4")

                from utils.visualization import draw_bboxes_on_video
                draw_bboxes_on_video(input_data, all_violations, output_video)
                result["l1_result"]["annotated_video"] = output_video
                print(f"[Task2] 标注视频已保存: {output_video}")
            except Exception as e:
                result["l1_result"]["annotated_video"] = f"视频标注失败: {str(e)}"
        else:
            result["l1_result"]["annotated_video"] = ""

        return result

    def _calc_iou(self, box1, box2):
        """计算IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

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