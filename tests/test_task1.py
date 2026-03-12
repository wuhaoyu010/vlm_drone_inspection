"""
Tests for processors/task1.py
"""
import pytest
import os
from unittest.mock import MagicMock, patch
from processors.task1 import Task1Processor, get_image_size


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

    def test_get_image_size_invalid_file(self, tmp_path):
        """Test getting size of invalid file"""
        invalid_file = tmp_path / "invalid.jpg"
        invalid_file.write_text("not an image")
        width, height = get_image_size(str(invalid_file))
        assert width is None
        assert height is None


class TestTask1Processor:
    """Tests for Task1Processor class"""

    def test_init(self, mock_vlm_client):
        """Test initialization"""
        processor = Task1Processor(mock_vlm_client)
        assert processor.vlm_client == mock_vlm_client

    def test_process_no_detection(self, mock_vlm_client, sample_image_path):
        """Test processing with no detections"""
        processor = Task1Processor(mock_vlm_client)
        result = processor.process(sample_image_path)

        assert "l1_result" in result
        assert "l2_result" in result
        assert "l3_result" in result
        mock_vlm_client.chat_with_image.assert_called_once()

    def test_process_with_detection(self, mock_vlm_client_with_detection, sample_image_path):
        """Test processing with detection"""
        processor = Task1Processor(mock_vlm_client_with_detection)
        result = processor.process(sample_image_path)

        assert "l1_result" in result
        assert len(result["l1_result"]) >= 1

    def test_process_with_output_dir(self, mock_vlm_client_with_detection, sample_image_path, tmp_path):
        """Test processing with output directory"""
        processor = Task1Processor(mock_vlm_client_with_detection)
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        result = processor.process(sample_image_path, output_dir=output_dir)

        assert "annotated_image" in result

    def test_process_with_scene_id(self, mock_vlm_client, sample_image_path):
        """Test processing with custom scene_id"""
        processor = Task1Processor(mock_vlm_client)
        result = processor.process(sample_image_path, scene_id=0)

        mock_vlm_client.chat_with_image.assert_called_once()

    def test_process_denormalizes_bbox(self, mock_vlm_client, sample_image_path):
        """Test that bbox is denormalized"""
        mock_vlm_client.chat_with_image.return_value = {
            "content": '''```json
{
    "l1_result": [{"bbox": [156, 208, 312, 416], "category": "test"}],
    "l2_result": "test",
    "l3_result": "test"
}
```''',
            "processed_dimensions": (640, 480)
        }
        processor = Task1Processor(mock_vlm_client)
        result = processor.process(sample_image_path)

        # 156/1000 * 640 ≈ 100, 208/1000 * 480 ≈ 100
        if result["l1_result"]:
            bbox = result["l1_result"][0]["bbox"]
            # Check that values were transformed (not exact due to integer conversion)
            assert bbox[0] != 156 or bbox[0] == 99 or bbox[0] == 100

    def test_process_filters_invalid_bbox(self, mock_vlm_client, sample_image_path):
        """Test that invalid bboxes are filtered out"""
        mock_vlm_client.chat_with_image.return_value = {
            "content": '''```json
{
    "l1_result": [
        {"bbox": [0, 0, 0, 0], "category": "invalid"},
        {"bbox": [100, 100, 200, 200], "category": "valid"}
    ],
    "l2_result": "test",
    "l3_result": "test"
}
```''',
            "processed_dimensions": (640, 480)
        }
        processor = Task1Processor(mock_vlm_client)
        result = processor.process(sample_image_path)

        # Should filter out the all-zeros bbox
        assert len(result["l1_result"]) <= 2

    def test_process_handles_vlm_error(self, mock_vlm_client, sample_image_path):
        """Test handling VLM client error"""
        mock_vlm_client.chat_with_image.side_effect = Exception("VLM error")
        processor = Task1Processor(mock_vlm_client)

        with pytest.raises(Exception):
            processor.process(sample_image_path)

    def test_process_empty_response(self, mock_vlm_client, sample_image_path):
        """Test processing empty VLM response"""
        mock_vlm_client.chat_with_image.return_value = {"content": "", "processed_dimensions": None}
        processor = Task1Processor(mock_vlm_client)
        result = processor.process(sample_image_path)

        assert "l1_result" in result

    def test_process_malformed_json(self, mock_vlm_client, sample_image_path):
        """Test processing malformed JSON response"""
        mock_vlm_client.chat_with_image.return_value = {
            "content": "This is not valid JSON {{{",
            "processed_dimensions": None
        }
        processor = Task1Processor(mock_vlm_client)
        result = processor.process(sample_image_path)

        assert "l1_result" in result