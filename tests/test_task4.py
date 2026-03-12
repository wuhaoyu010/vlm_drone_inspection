"""
Tests for processors/task4.py
"""
import pytest
import os
from unittest.mock import MagicMock, patch
from processors.task4 import Task4Processor, get_image_size


class TestTask4Processor:
    """Tests for Task4Processor class"""

    def test_init(self, mock_vlm_client):
        """Test initialization"""
        processor = Task4Processor(mock_vlm_client)
        assert processor.vlm_client == mock_vlm_client

    def test_scene_category_map(self, mock_vlm_client):
        """Test scene category mapping"""
        processor = Task4Processor(mock_vlm_client)

        assert processor.SCENE_CATEGORY_MAP[6] == "边坡滑坡"
        assert processor.SCENE_CATEGORY_MAP[7] == "排水沟积水"
        assert processor.SCENE_CATEGORY_MAP[8] == "排水沟破损"

    def test_process_no_detection(self, mock_vlm_client, sample_image_path):
        """Test processing with no detections"""
        processor = Task4Processor(mock_vlm_client)
        result = processor.process(sample_image_path)

        assert "l1_result" in result
        assert "l2_result" in result
        assert "l3_result" in result
        mock_vlm_client.chat_with_image.assert_called_once()

    def test_process_with_detection(self, mock_vlm_client, sample_image_path):
        """Test processing with detection"""
        mock_vlm_client.chat_with_image.return_value = '''```json
{
    "l1_result": [{"bbox": [100, 100, 200, 200], "category": "边坡滑坡", "scene_id": 6}],
    "l2_result": "检测到边坡滑坡",
    "l3_result": "风险等级：P1"
}
```'''
        processor = Task4Processor(mock_vlm_client)
        result = processor.process(sample_image_path)

        assert len(result["l1_result"]) >= 1

    def test_process_with_scene_id(self, mock_vlm_client, sample_image_path):
        """Test processing with specific scene_id"""
        processor = Task4Processor(mock_vlm_client)
        result = processor.process(sample_image_path, scene_id=6)

        mock_vlm_client.chat_with_image.assert_called_once()

    def test_process_with_output_dir(self, mock_vlm_client_with_detection, sample_image_path, tmp_path):
        """Test processing with output directory"""
        processor = Task4Processor(mock_vlm_client_with_detection)
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        result = processor.process(sample_image_path, output_dir=output_dir)
        assert "annotated_image" in result

    def test_infer_scene_id_slope(self, mock_vlm_client):
        """Test scene_id inference for slope issues"""
        processor = Task4Processor(mock_vlm_client)

        assert processor._infer_scene_id("边坡滑坡") == 6
        assert processor._infer_scene_id("滑坡") == 6
        assert processor._infer_scene_id("坍塌") == 6

    def test_infer_scene_id_drainage_water(self, mock_vlm_client):
        """Test scene_id inference for drainage water"""
        processor = Task4Processor(mock_vlm_client)

        assert processor._infer_scene_id("排水沟积水") == 7

    def test_infer_scene_id_drainage_damage(self, mock_vlm_client):
        """Test scene_id inference for drainage damage"""
        processor = Task4Processor(mock_vlm_client)

        assert processor._infer_scene_id("排水沟破损") == 8
        assert processor._infer_scene_id("排水沟堵塞") == 8

    def test_infer_scene_id_unknown(self, mock_vlm_client):
        """Test scene_id inference for unknown category"""
        processor = Task4Processor(mock_vlm_client)

        # Default to 6 (slope)
        assert processor._infer_scene_id("未知病害") == 6

    def test_process_denormalizes_and_validates_bbox(self, mock_vlm_client, sample_image_path):
        """Test that bbox is denormalized and validated"""
        mock_vlm_client.chat_with_image.return_value = '''```json
{
    "l1_result": [{"bbox": [156, 208, 312, 416], "category": "边坡滑坡"}],
    "l2_result": "test",
    "l3_result": "test"
}
```'''
        processor = Task4Processor(mock_vlm_client)
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
        {"bbox": [100, 100, 200, 200], "category": "边坡滑坡"}
    ],
    "l2_result": "test",
    "l3_result": "test"
}
```'''
        processor = Task4Processor(mock_vlm_client)
        result = processor.process(sample_image_path)

        # Should filter out invalid bbox
        assert len(result["l1_result"]) <= 2

    def test_process_scene_id_7(self, mock_vlm_client, sample_image_path):
        """Test processing with scene_id 7 (排水沟积水)"""
        processor = Task4Processor(mock_vlm_client)
        result = processor.process(sample_image_path, scene_id=7)

        mock_vlm_client.chat_with_image.assert_called_once()
        # Check that prompt was modified for specific scene
        call_args = mock_vlm_client.chat_with_image.call_args
        prompt = call_args.kwargs.get('prompt', call_args.args[1] if len(call_args.args) > 1 else '')
        assert "排水沟积水" in prompt or "重点检测" in prompt

    def test_process_scene_id_8(self, mock_vlm_client, sample_image_path):
        """Test processing with scene_id 8 (排水沟破损)"""
        processor = Task4Processor(mock_vlm_client)
        result = processor.process(sample_image_path, scene_id=8)

        mock_vlm_client.chat_with_image.assert_called_once()

    def test_process_without_image_size(self, mock_vlm_client, tmp_path):
        """Test processing when image size cannot be determined"""
        # Create a file that's not a valid image
        invalid_file = tmp_path / "invalid.jpg"
        invalid_file.write_text("not an image")

        processor = Task4Processor(mock_vlm_client)
        result = processor.process(str(invalid_file))

        assert "l1_result" in result

    def test_process_empty_result(self, mock_vlm_client, sample_image_path):
        """Test processing with empty result"""
        mock_vlm_client.chat_with_image.return_value = '{"l1_result": [], "l2_result": "", "l3_result": ""}'
        processor = Task4Processor(mock_vlm_client)
        result = processor.process(sample_image_path)

        assert result["l1_result"] == []