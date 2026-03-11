"""
目标跟踪模块 - 实现视频中的目标跟踪
支持基于IoU匹配的简单跟踪和OpenCV内置追踪器
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TrackedObject:
    """跟踪目标"""
    track_id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    category: str
    last_frame: int
    lost_frames: int = 0
    trajectory: List[Tuple[int, List[int]]] = None  # [(frame_id, bbox), ...]

    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = [(self.last_frame, self.bbox.copy())]


class SimpleTracker:
    """
    简单的目标跟踪器
    基于IoU匹配实现帧间目标关联
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30):
        """
        Args:
            iou_threshold: IoU匹配阈值
            max_lost_frames: 最大丢失帧数
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1

    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """计算两个框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, frame_id: int, detections: List[Dict]) -> List[Dict]:
        """
        更新跟踪状态

        Args:
            frame_id: 当前帧ID
            detections: 当前帧的检测结果 [{"bbox": [x1,y1,x2,y2], "category": "类型"}]

        Returns:
            更新后的跟踪结果（包含track_id）
        """
        # 标记所有现有轨迹为未匹配
        matched_tracks = set()
        matched_detections = set()

        # 为每个检测找最佳匹配的轨迹
        for det_idx, det in enumerate(detections):
            det_bbox = det.get("bbox", [0, 0, 100, 100])
            det_category = det.get("category", "未知")

            best_iou = 0
            best_track_id = None

            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue

                # 类别必须匹配
                if track.category != det_category:
                    continue

                iou = self.calculate_iou(det_bbox, track.bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # 更新匹配的轨迹
                track = self.tracks[best_track_id]
                track.bbox = det_bbox
                track.last_frame = frame_id
                track.lost_frames = 0
                track.trajectory.append((frame_id, det_bbox.copy()))
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)

        # 处理未匹配的轨迹（增加丢失计数）
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track.lost_frames += 1
                if track.lost_frames > self.max_lost_frames:
                    tracks_to_remove.append(track_id)

        # 移除长时间丢失的轨迹
        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        # 为未匹配的检测创建新轨迹
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                det_bbox = det.get("bbox", [0, 0, 100, 100])
                det_category = det.get("category", "未知")

                new_track = TrackedObject(
                    track_id=self.next_id,
                    bbox=det_bbox,
                    category=det_category,
                    last_frame=frame_id
                )
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        # 返回所有活跃轨迹
        results = []
        for track_id, track in self.tracks.items():
            results.append({
                "track_id": track.track_id,
                "bbox": track.bbox,
                "category": track.category,
                "last_frame": track.last_frame,
                "lost_frames": track.lost_frames
            })

        return results

    def get_all_trajectories(self) -> Dict[int, List[Tuple[int, List[int]]]]:
        """获取所有轨迹"""
        return {track_id: track.trajectory for track_id, track in self.tracks.items()}


class OpenCVTracker:
    """
    基于OpenCV的目标追踪器
    使用CSRT或KCF追踪器实现实时跟踪
    """

    def __init__(self, tracker_type: str = "CSRT"):
        """
        Args:
            tracker_type: 追踪器类型 (CSRT, KCF, MIL, MOSSE)
        """
        self.tracker_type = tracker_type
        self.trackers: Dict[int, cv2.Tracker] = {}
        self.track_info: Dict[int, Dict] = {}
        self.next_id = 1

    def _create_tracker(self) -> cv2.Tracker:
        """创建追踪器 - 兼容不同OpenCV版本"""
        try:
            # OpenCV 4.5.1+ 新API
            if self.tracker_type == "CSRT":
                return cv2.TrackerCSRT()
            elif self.tracker_type == "KCF":
                return cv2.TrackerKCF()
            elif self.tracker_type == "MIL":
                return cv2.TrackerMIL()
            elif self.tracker_type == "MOSSE":
                return cv2.TrackerMOSSE()
            else:
                return cv2.TrackerCSRT()
        except TypeError:
            # OpenCV 4.2- 旧API (cv2.TrackerXXX_create)
            if self.tracker_type == "CSRT":
                return cv2.TrackerCSRT_create()
            elif self.tracker_type == "KCF":
                return cv2.TrackerKCF_create()
            elif self.tracker_type == "MIL":
                return cv2.TrackerMIL_create()
            elif self.tracker_type == "MOSSE":
                return cv2.TrackerMOSSE_create()
            else:
                return cv2.TrackerCSRT_create()

    def init(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """
        初始化追踪器

        Args:
            frame: 视频帧
            detections: 初始检测结果
        """
        self.trackers.clear()
        self.track_info.clear()

        for det in detections:
            bbox = det.get("bbox", [0, 0, 100, 100])
            category = det.get("category", "未知")

            # 转换为OpenCV格式 (x, y, w, h)
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y

            tracker = self._create_tracker()
            tracker.init(frame, (x, y, w, h))

            self.trackers[self.next_id] = tracker
            self.track_info[self.next_id] = {
                "category": category,
                "bbox": bbox
            }
            self.next_id += 1

    def update(self, frame: np.ndarray) -> List[Dict]:
        """
        更新追踪状态

        Args:
            frame: 当前帧

        Returns:
            跟踪结果
        """
        results = []
        tracks_to_remove = []

        for track_id, tracker in self.trackers.items():
            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = bbox
                # 转换回 [x1, y1, x2, y2] 格式
                det_bbox = [int(x), int(y), int(x + w), int(y + h)]

                self.track_info[track_id]["bbox"] = det_bbox

                results.append({
                    "track_id": track_id,
                    "bbox": det_bbox,
                    "category": self.track_info[track_id]["category"]
                })
            else:
                tracks_to_remove.append(track_id)

        # 移除失效的追踪器
        for track_id in tracks_to_remove:
            del self.trackers[track_id]
            del self.track_info[track_id]

        return results


def interpolate_bbox(bbox1: List[int], bbox2: List[int], t: float) -> List[int]:
    """
    线性插值两个边界框

    Args:
        bbox1: 起始框
        bbox2: 结束框
        t: 插值参数 [0, 1]

    Returns:
        插值后的框
    """
    return [
        int(bbox1[i] + (bbox2[i] - bbox1[i]) * t)
        for i in range(4)
    ]


def smooth_trajectory(
    detections: List[Dict],
    fps: int,
    smoothing_window: int = 5
) -> List[Dict]:
    """
    平滑轨迹，填充帧间空白

    Args:
        detections: 检测结果列表 [{"frame_id": int, "bbox": [...], "category": str}]
        fps: 视频帧率
        smoothing_window: 平滑窗口大小

    Returns:
        平滑后的检测结果（包含插值帧）
    """
    if not detections:
        return []

    # 按frame_id排序
    sorted_dets = sorted(detections, key=lambda x: x.get("frame_id", 0))

    # 按类别分组
    by_category = defaultdict(list)
    for det in sorted_dets:
        by_category[det.get("category", "未知")].append(det)

    all_interpolated = []

    for category, dets in by_category.items():
        for i in range(len(dets) - 1):
            current = dets[i]
            next_det = dets[i + 1]

            current_frame = current.get("frame_id", 0)
            next_frame = next_det.get("frame_id", current_frame + 1)

            # 添加当前帧
            all_interpolated.append(current)

            # 如果帧间隔太大，进行插值
            frame_gap = next_frame - current_frame
            if frame_gap > 1 and frame_gap <= fps * 3:  # 最多插值3秒
                for j in range(1, frame_gap):
                    t = j / frame_gap
                    interp_bbox = interpolate_bbox(
                        current.get("bbox", [0, 0, 100, 100]),
                        next_det.get("bbox", [0, 0, 100, 100]),
                        t
                    )
                    all_interpolated.append({
                        "frame_id": current_frame + j,
                        "bbox": interp_bbox,
                        "category": category,
                        "interpolated": True
                    })

        # 添加最后一个检测
        all_interpolated.append(dets[-1])

    # 按帧ID排序
    all_interpolated.sort(key=lambda x: x.get("frame_id", 0))

    return all_interpolated