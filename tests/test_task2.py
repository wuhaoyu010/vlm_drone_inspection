"""
Tests for processors/task2.py
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from processors.task2 import Task2Processor, get_video_info


class TestGetVideoInfo:
    """Tests for get_video_info function"""

    def test_get_video_info_valid(self, sample_video_path):
        """Test getting info of valid video"""
        width, height, frames, fps = get_video_info(sample_video_path)
        assert width == 640
        assert height == 480
        assert frames == 30
        assert fps > 0

    def test_get_video_info_invalid_path(self):
        """Test getting info of non-existent video"""
        with pytest.raises(ValueError):
            get_video_info("/nonexistent/video.mp4")


class TestTask2Processor:
    """Tests for Task2Processor class"""

    def test_init(self, mock_vlm_client):
        """Test initialization"""
        processor = Task2Processor(mock_vlm_client)
        assert processor.vlm_client == mock_vlm_client

    def test_get_vehicle_type(self, mock_vlm_client):
        """Test vehicle type mapping"""
        processor = Task2Processor(mock_vlm_client)

        assert processor.get_vehicle_type(2) == "小型轿车"
        assert processor.get_vehicle_type(3) == "摩托车"
        assert processor.get_vehicle_type(5) == "客车"
        assert processor.get_vehicle_type(7) == "货车"
        assert processor.get_vehicle_type(99) == "车辆"

    def test_detect_vehicles_no_model(self, mock_vlm_client):
        """Test detection when YOLO model is not available"""
        processor = Task2Processor(mock_vlm_client)
        processor.yolo_model = None

        detections = processor.detect_vehicles(np.zeros((480, 640, 3), dtype=np.uint8))
        assert detections.shape == (0, 6)

    def test_is_stationary_false_short_history(self, mock_vlm_client):
        """Test is_stationary with short history"""
        processor = Task2Processor(mock_vlm_client)

        # Short history (< 2 seconds at 30fps)
        history = [(i, [100, 100, 200, 200]) for i in range(30)]
        result = processor.is_stationary(history, fps=30, threshold_seconds=2.0)
        assert result is False

    def test_is_stationary_true_stationary(self, mock_vlm_client):
        """Test is_stationary with stationary vehicle"""
        processor = Task2Processor(mock_vlm_client)

        # Stationary vehicle (same position for 60 frames = 2 seconds at 30fps)
        history = [(i, [100, 100, 200, 200]) for i in range(60)]
        result = processor.is_stationary(history, fps=30, threshold_seconds=2.0)
        assert result is True

    def test_is_stationary_false_moving(self, mock_vlm_client):
        """Test is_stationary with moving vehicle"""
        processor = Task2Processor(mock_vlm_client)

        # Moving vehicle (position changes significantly)
        history = [(i, [100 + i * 20, 100, 200 + i * 20, 200]) for i in range(60)]
        result = processor.is_stationary(history, fps=30, threshold_seconds=2.0)
        assert result is False

    def test_is_stationary_single_frame(self, mock_vlm_client):
        """Test is_stationary with single frame"""
        processor = Task2Processor(mock_vlm_client)

        history = [(0, [100, 100, 200, 200])]
        result = processor.is_stationary(history, fps=30, threshold_seconds=2.0)
        assert result is False

    def test_calc_iou_overlap(self, mock_vlm_client):
        """Test IoU calculation with overlap"""
        processor = Task2Processor(mock_vlm_client)

        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        iou = processor._calc_iou(box1, box2)
        assert 0 < iou < 1

    def test_calc_iou_no_overlap(self, mock_vlm_client):
        """Test IoU calculation with no overlap"""
        processor = Task2Processor(mock_vlm_client)

        box1 = [0, 0, 100, 100]
        box2 = [200, 200, 300, 300]
        iou = processor._calc_iou(box1, box2)
        assert iou == 0.0

    def test_calc_iou_identical(self, mock_vlm_client):
        """Test IoU calculation with identical boxes"""
        processor = Task2Processor(mock_vlm_client)

        box = [0, 0, 100, 100]
        iou = processor._calc_iou(box, box)
        assert iou == 1.0

    def test_calc_iou_contained(self, mock_vlm_client):
        """Test IoU calculation with one box contained in another"""
        processor = Task2Processor(mock_vlm_client)

        box1 = [0, 0, 100, 100]
        box2 = [25, 25, 75, 75]
        iou = processor._calc_iou(box1, box2)
        assert 0 < iou < 1

    def test_normalize_response_for_task2(self, mock_vlm_client):
        """Test response normalization for task2"""
        processor = Task2Processor(mock_vlm_client)

        result = processor.normalize_response_for_task2({}, scene_id=1)
        assert "l1_result" in result
        assert "has_violation" in result["l1_result"]
        assert "violations" in result["l1_result"]

    def test_normalize_response_for_task2_partial(self, mock_vlm_client):
        """Test response normalization with partial data"""
        processor = Task2Processor(mock_vlm_client)

        result = processor.normalize_response_for_task2(
            {"l1_result": {"has_violation": True, "violations": []}},
            scene_id=1
        )
        assert result["l1_result"]["violations"] == []
        assert result["l1_result"]["has_violation"] is True

    def test_normalize_response_for_task2_missing_violations(self, mock_vlm_client):
        """Test response normalization when violations is missing"""
        processor = Task2Processor(mock_vlm_client)

        result = processor.normalize_response_for_task2(
            {"l1_result": {"has_violation": True}},
            scene_id=1
        )
        # The method doesn't add violations if l1_result exists
        assert result["l1_result"]["has_violation"] is True

    @patch('processors.task2.get_video_info')
    def test_process_video_error(self, mock_get_info, mock_vlm_client):
        """Test processing when video info fails"""
        mock_get_info.side_effect = Exception("Video error")
        processor = Task2Processor(mock_vlm_client)

        result = processor.process("test.mp4")
        assert "error" in result["l1_result"]["video_info"]

    @patch('processors.task2.get_video_info')
    def test_process_no_detections(self, mock_get_info, mock_vlm_client, sample_video_path):
        """Test processing with no vehicle detections"""
        mock_get_info.return_value = (640, 480, 30, 30.0)
        processor = Task2Processor(mock_vlm_client)
        processor.yolo_model = None  # Disable YOLO

        result = processor.process(str(sample_video_path))
        assert "l1_result" in result
        assert result["l1_result"]["has_violation"] is False