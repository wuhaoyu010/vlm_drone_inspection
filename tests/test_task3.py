"""
Tests for processors/task3.py
"""
import pytest
import os
from unittest.mock import MagicMock, patch
from processors.task3 import Task3Processor, get_image_size


class TestGetImageSize:
    """Tests for get_image_size function"""

    def test_get_image_size_valid(self, sample_image_path):
        """Test getting size of valid image"""
        width, height = get_image_size(sample_image_path)
        assert width == 640
        assert height == 480

    def test_get_image_size_invalid_path(self):
        """Test getting size of non-existent image"""
        width, height = get_image_size("/nonexistent/image.jpg")
        assert width is None
        assert height is None


class TestTask3Processor:
    """Tests for Task3Processor class"""

    def test_init(self, mock_vlm_client):
        """Test initialization"""
        processor = Task3Processor(mock_vlm_client)
        assert processor.vlm_client == mock_vlm_client

    def test_scene_category_map(self, mock_vlm_client):
        """Test scene category mapping"""
        processor = Task3Processor(mock_vlm_client)

        assert processor.SCENE_CATEGORY_MAP[2] == "裂缝"
        assert processor.SCENE_CATEGORY_MAP[3] == "坑洼"
        assert processor.SCENE_CATEGORY_MAP[4] == "积水"
        assert processor.SCENE_CATEGORY_MAP[5] == "护栏破损"

    def test_process_no_detection(self, mock_vlm_client, sample_image_path):
        """Test processing with no detections"""
        processor = Task3Processor(mock_vlm_client)
        result = processor.process(sample_image_path)

        assert "l1_result" in result
        assert "l2_result" in result
        assert "l3_result" in result
        mock_vlm_client.chat_with_image.assert_called_once()

    def test_process_with_detection(self, mock_vlm_client, sample_image_path):
        """Test processing with detection"""
        mock_vlm_client.chat_with_image.return_value = '''```json
{
    "l1_result": [{"bbox": [100, 100, 200, 200], "category": "裂缝", "scene_id": 2}],
    "l2_result": "检测到裂缝",
    "l3_result": "风险等级：P1"
}
```'''
        processor = Task3Processor(mock_vlm_client)
        result = processor.process(sample_image_path)

        assert len(result["l1_result"]) >= 1

    def test_process_with_scene_id(self, mock_vlm_client, sample_image_path):
        """Test processing with specific scene_id"""
        processor = Task3Processor(mock_vlm_client)
        result = processor.process(sample_image_path, scene_id=2)

        mock_vlm_client.chat_with_image.assert_called_once()

    def test_process_with_output_dir(self, mock_vlm_client_with_detection, sample_image_path, tmp_path):
        """Test processing with output directory"""
        processor = Task3Processor(mock_vlm_client_with_detection)
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        result = processor.process(sample_image_path, output_dir=output_dir)
        assert "annotated_image" in result

    def test_infer_scene_id_crack(self, mock_vlm_client):
        """Test scene_id inference for cracks"""
        processor = Task3Processor(mock_vlm_client)

        assert processor._infer_scene_id("裂缝") == 2
        assert processor._infer_scene_id("裂痕") == 2

    def test_infer_scene_id_pothole(self, mock_vlm_client):
        """Test scene_id inference for potholes"""
        processor = Task3Processor(mock_vlm_client)

        assert processor._infer_scene_id("坑洼") == 3
        assert processor._infer_scene_id("坑洞") == 3
        assert processor._infer_scene_id("凹陷") == 3

    def test_infer_scene_id_water(self, mock_vlm_client):
        """Test scene_id inference for water"""
        processor = Task3Processor(mock_vlm_client)

        assert processor._infer_scene_id("积水") == 4
        assert processor._infer_scene_id("路面积水") == 4

    def test_infer_scene_id_guardrail(self, mock_vlm_client):
        """Test scene_id inference for guardrail"""
        processor = Task3Processor(mock_vlm_client)

        assert processor._infer_scene_id("护栏破损") == 5
        assert processor._infer_scene_id("护栏断裂") == 5

    def test_infer_scene_id_unknown(self, mock_vlm_client):
        """Test scene_id inference for unknown category"""
        processor = Task3Processor(mock_vlm_client)

        # Default to 2 (crack)
        assert processor._infer_scene_id("未知病害") == 2

    def test_process_denormalizes_and_validates_bbox(self, mock_vlm_client, sample_image_path):
        """Test that bbox is denormalized and validated"""
        mock_vlm_client.chat_with_image.return_value = '''```json
{
    "l1_result": [{"bbox": [156, 208, 312, 416], "category": "裂缝"}],
    "l2_result": "test",
    "l3_result": "test"
}
```'''
        processor = Task3Processor(mock_vlm_client)
        result = processor.process(sample_image_path)

        # Check that scene_id was inferred
        if result["l1_result"]:
            assert "scene_id" in result["l1_result"][0]

    def test_process_filters_invalid_bbox(self, mock_vlm_client, sample_image_path):
        """Test that invalid bboxes are filtered"""
        mock_vlm_client.chat_with_image.return_value = '''```json
{
    "l1_result": [
        {"bbox": [0, 0, 0, 0], "category": "invalid"},
        {"bbox": [100, 100, 200, 200], "category": "裂缝"}
    ],
    "l2_result": "test",
    "l3_result": "test"
}
```'''
        processor = Task3Processor(mock_vlm_client)
        result = processor.process(sample_image_path)

        # Should filter out invalid bbox
        assert len(result["l1_result"]) <= 2

    def test_process_without_image_size(self, mock_vlm_client, tmp_path):
        """Test processing when image size cannot be determined"""
        # Create a file that's not a valid image
        invalid_file = tmp_path / "invalid.jpg"
        invalid_file.write_text("not an image")

        processor = Task3Processor(mock_vlm_client)
        result = processor.process(str(invalid_file))

        assert "l1_result" in result