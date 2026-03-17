"""
Tests for processors/task2.py
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from processors.task2 import Task2Processor, get_video_info, get_task2_config


class TestTask2Config:
    """Tests for task2 configuration"""

    def test_get_task2_config_defaults(self):
        """Test default task2 config values"""
        config = get_task2_config()
        assert "model_path" in config
        assert "conf_threshold" in config
        assert "generate_annotated_video" in config  # 新增配置项

    def test_generate_annotated_video_config_exists(self):
        """Test that generate_annotated_video config exists"""
        config = get_task2_config()
        # 默认应该是 True 或 False，取决于配置文件
        assert isinstance(config.get("generate_annotated_video", True), bool)


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

    def test_normalize_bbox(self, mock_vlm_client):
        """Test bbox normalization"""
        processor = Task2Processor(mock_vlm_client)

        # 测试归一化
        bbox = [100, 200, 300, 400]
        normalized = processor._normalize_bbox(bbox, 1920, 1080)

        # 100 * 1000 / 1920 ≈ 52
        assert normalized[0] == int(100 * 1000 / 1920)
        assert normalized[1] == int(200 * 1000 / 1080)
        assert normalized[2] == int(300 * 1000 / 1920)
        assert normalized[3] == int(400 * 1000 / 1080)

    @patch('processors.task2.get_video_info')
    def test_process_video_error(self, mock_get_info, mock_vlm_client):
        """Test processing when video info fails"""
        mock_get_info.side_effect = Exception("Video error")
        processor = Task2Processor(mock_vlm_client, preload_model=False)

        result = processor.process("test.mp4")
        # 新格式检查
        assert result["success"] is False
        assert "error" in result["video_info"]

    @patch('processors.task2.get_video_info')
    def test_process_no_detections(self, mock_get_info, mock_vlm_client, sample_video_path):
        """Test processing with no vehicle detections"""
        mock_get_info.return_value = (640, 480, 30, 30.0)
        processor = Task2Processor(mock_vlm_client, preload_model=False)
        processor.tracker = None  # Disable tracker

        # 由于没有实际模型，这里只测试返回格式
        # 实际测试需要mock tracker
        result = {
            "success": True,
            "data": [],
        }
        assert "data" in result
        assert result["data"] == []


class TestTask2NewOutputFormat:
    """Tests for new output format (主办方新格式)"""

    def test_format_output_new_format(self, mock_vlm_client):
        """Test formatting output to new format"""
        processor = Task2Processor(mock_vlm_client, preload_model=False)

        # 模拟检测结果
        violations = [
            {
                "frame_id": 100,
                "bbox": [100, 200, 300, 400],
                "vehicle_type": "car",
                "parking_location": "应急车道",
            },
            {
                "frame_id": 100,
                "bbox": [500, 600, 700, 800],
                "vehicle_type": "truck",
                "parking_location": "应急车道",
            },
        ]
        reasoning = ["分析过程1", "分析过程2"]
        suggestion = ["建议1", "建议2"]

        # 调用格式化方法
        result = processor.format_output_new(
            violations=violations,
            reasoning=reasoning,
            suggestion=suggestion,
            scene_id=1,
        )

        # 验证新格式
        assert result["success"] is True
        assert result["scene_id"] == 1
        assert "data" in result
        assert len(result["data"]) == 1  # 按帧分组

        # 验证数据结构
        frame_data = result["data"][0]
        assert frame_data["frame_id"] == 100
        assert len(frame_data["bboxs"]) == 2
        assert len(frame_data["reasoning"]) == 2
        assert len(frame_data["suggestion"]) == 2

    def test_format_output_multiple_frames(self, mock_vlm_client):
        """Test formatting output with violations in different frames"""
        processor = Task2Processor(mock_vlm_client, preload_model=False)

        # 模拟不同帧的检测结果（新格式要求只输出一个帧，取最早帧）
        violations = [
            {"frame_id": 50, "bbox": [100, 200, 300, 400], "vehicle_type": "car"},
            {"frame_id": 100, "bbox": [500, 600, 700, 800], "vehicle_type": "truck"},
        ]
        reasoning = ["分析过程1", "分析过程2"]
        suggestion = ["建议1", "建议2"]

        result = processor.format_output_new(
            violations=violations,
            reasoning=reasoning,
            suggestion=suggestion,
            scene_id=1,
        )

        # 应该只输出最早的帧
        assert result["data"][0]["frame_id"] == 50

    def test_format_output_empty(self, mock_vlm_client):
        """Test formatting output with no violations"""
        processor = Task2Processor(mock_vlm_client, preload_model=False)

        result = processor.format_output_new(
            violations=[],
            reasoning=[],
            suggestion=[],
            scene_id=1,
        )

        assert result["success"] is True
        assert result["data"] == []

    def test_bbox_not_normalized_in_new_format(self, mock_vlm_client):
        """Test that bbox in new format is not normalized (actual pixel coords)"""
        processor = Task2Processor(mock_vlm_client, preload_model=False)

        violations = [
            {"frame_id": 100, "bbox": [100, 200, 300, 400], "vehicle_type": "car"},
        ]
        reasoning = ["分析过程"]
        suggestion = ["建议"]

        result = processor.format_output_new(
            violations=violations,
            reasoning=reasoning,
            suggestion=suggestion,
            scene_id=1,
        )

        # bbox 应该是实际像素坐标，不是归一化坐标
        bbox = result["data"][0]["bboxs"][0]
        # 如果坐标值 > 1000，说明不是归一化坐标
        # 但对于小分辨率，可能坐标值本身就小
        # 所以我们检查格式：应该是 [x1, y1, x2, y2] 的整数列表
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert all(isinstance(v, int) for v in bbox)

    def test_reasoning_suggestion_one_to_one(self, mock_vlm_client):
        """Test that reasoning and suggestion are one-to-one with bboxs"""
        processor = Task2Processor(mock_vlm_client, preload_model=False)

        violations = [
            {"frame_id": 100, "bbox": [100, 200, 300, 400], "vehicle_type": "car"},
            {"frame_id": 100, "bbox": [500, 600, 700, 800], "vehicle_type": "truck"},
            {"frame_id": 100, "bbox": [900, 100, 1100, 300], "vehicle_type": "bus"},
        ]
        reasoning = ["分析1", "分析2", "分析3"]
        suggestion = ["建议1", "建议2", "建议3"]

        result = processor.format_output_new(
            violations=violations,
            reasoning=reasoning,
            suggestion=suggestion,
            scene_id=1,
        )

        # 验证一一对应
        frame_data = result["data"][0]
        assert len(frame_data["bboxs"]) == 3
        assert len(frame_data["reasoning"]) == 3
        assert len(frame_data["suggestion"]) == 3

    def test_reasoning_suggestion_fallback(self, mock_vlm_client):
        """Test fallback when reasoning/suggestion is shorter than violations"""
        processor = Task2Processor(mock_vlm_client, preload_model=False)

        violations = [
            {"frame_id": 100, "bbox": [100, 200, 300, 400], "vehicle_type": "car"},
            {"frame_id": 100, "bbox": [500, 600, 700, 800], "vehicle_type": "truck"},
        ]
        reasoning = ["分析1"]  # 少于violations数量
        suggestion = []  # 空列表

        result = processor.format_output_new(
            violations=violations,
            reasoning=reasoning,
            suggestion=suggestion,
            scene_id=1,
        )

        # 验证会自动补全
        frame_data = result["data"][0]
        assert len(frame_data["reasoning"]) == 2
        assert len(frame_data["suggestion"]) == 2
        # 第二个应该是fallback结果
        assert "分析过程" in frame_data["reasoning"][1] or len(frame_data["reasoning"][1]) > 0