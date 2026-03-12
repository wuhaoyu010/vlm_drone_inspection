"""
Tests for processors/base.py
"""
import pytest
from processors.base import BaseProcessor, denormalize_bbox, is_normalized_bbox


# Concrete implementation for testing abstract class
class ConcreteProcessor(BaseProcessor):
    """Concrete implementation for testing"""

    def process(self, input_data, scene_id=None):
        return {"l1_result": [], "l2_result": "", "l3_result": ""}


class TestDenormalizeBbox:
    """Tests for denormalize_bbox function"""

    def test_denormalize_basic(self):
        """Test basic denormalization"""
        bbox = [100, 100, 200, 200]
        result = denormalize_bbox(bbox, 640, 480)
        # 100/1000 * 640 = 64, 100/1000 * 480 = 48
        assert result == [64, 48, 128, 96]

    def test_denormalize_with_zero_values(self):
        """Test with zero values"""
        bbox = [0, 0, 0, 0]
        result = denormalize_bbox(bbox, 640, 480)
        assert result == [0, 0, 0, 0]

    def test_denormalize_with_max_values(self):
        """Test with max normalized values"""
        bbox = [1000, 1000, 1000, 1000]
        result = denormalize_bbox(bbox, 640, 480)
        assert result == [640, 480, 640, 480]

    def test_denormalize_already_pixel_coords(self):
        """Test with already pixel coordinates (> 1000)"""
        bbox = [100, 200, 300, 400]  # values < 1000, will be treated as normalized
        result = denormalize_bbox(bbox, 640, 480)
        # Will be treated as normalized since all values <= 1000
        assert result == [64, 96, 192, 192]

    def test_denormalize_pixel_coords_large(self):
        """Test with pixel coordinates > 1000"""
        bbox = [1200, 1300, 1400, 1500]  # values > 1000, treated as pixel coords
        result = denormalize_bbox(bbox, 1920, 1080)
        assert result == [1200, 1300, 1400, 1500]

    def test_denormalize_empty_bbox(self):
        """Test with empty bbox"""
        result = denormalize_bbox([], 640, 480)
        assert result == []

    def test_denormalize_invalid_bbox_length(self):
        """Test with invalid bbox length"""
        result = denormalize_bbox([100, 200], 640, 480)
        assert result == [100, 200]

    def test_denormalize_none_bbox(self):
        """Test with None bbox"""
        result = denormalize_bbox(None, 640, 480)
        assert result is None

    def test_denormalize_string_values(self):
        """Test with string values (should handle gracefully)"""
        bbox = ["100", "100", "200", "200"]
        result = denormalize_bbox(bbox, 640, 480)
        assert result == [64, 48, 128, 96]


class TestIsNormalizedBbox:
    """Tests for is_normalized_bbox function"""

    def test_is_normalized_true(self):
        """Test with normalized bbox"""
        assert is_normalized_bbox([100, 100, 200, 200]) is True

    def test_is_normalized_false_large_values(self):
        """Test with pixel coordinates"""
        assert is_normalized_bbox([1200, 1300, 1400, 1500]) is False

    def test_is_normalized_empty(self):
        """Test with empty bbox"""
        assert is_normalized_bbox([]) is False

    def test_is_normalized_invalid_length(self):
        """Test with invalid length"""
        assert is_normalized_bbox([100, 200]) is False

    def test_is_normalized_none(self):
        """Test with None"""
        assert is_normalized_bbox(None) is False


class TestBaseProcessor:
    """Tests for BaseProcessor class"""

    def test_init(self, mock_vlm_client):
        """Test initialization"""
        processor = ConcreteProcessor(mock_vlm_client)
        assert processor.vlm_client == mock_vlm_client

    def test_parse_json_response_direct(self, mock_vlm_client):
        """Test parsing direct JSON response"""
        processor = ConcreteProcessor(mock_vlm_client)
        response = '{"l1_result": [], "l2_result": "test", "l3_result": ""}'
        result = processor.parse_json_response(response)
        assert result == {"l1_result": [], "l2_result": "test", "l3_result": ""}

    def test_parse_json_response_code_block(self, mock_vlm_client):
        """Test parsing JSON from code block"""
        processor = ConcreteProcessor(mock_vlm_client)
        response = '''Here is the result:
```json
{"l1_result": [{"bbox": [1, 2, 3, 4]}], "l2_result": "test"}
```
'''
        result = processor.parse_json_response(response)
        assert result["l1_result"] == [{"bbox": [1, 2, 3, 4]}]

    def test_parse_json_response_plain_code_block(self, mock_vlm_client):
        """Test parsing JSON from plain code block"""
        processor = ConcreteProcessor(mock_vlm_client)
        response = '''```
{"l1_result": [], "l2_result": "test"}
```'''
        result = processor.parse_json_response(response)
        assert result == {"l1_result": [], "l2_result": "test"}

    def test_parse_json_response_braces(self, mock_vlm_client):
        """Test parsing JSON from braces"""
        processor = ConcreteProcessor(mock_vlm_client)
        response = 'Some text {"l1_result": [], "l2_result": "test"} more text'
        result = processor.parse_json_response(response)
        assert result == {"l1_result": [], "l2_result": "test"}

    def test_parse_json_response_invalid(self, mock_vlm_client):
        """Test parsing invalid JSON"""
        processor = ConcreteProcessor(mock_vlm_client)
        response = 'This is not JSON'
        result = processor.parse_json_response(response)
        assert result == {"l1_result": [], "l2_result": "This is not JSON", "l3_result": ""}

    def test_validate_bbox_valid(self, mock_vlm_client):
        """Test validating valid bbox"""
        processor = ConcreteProcessor(mock_vlm_client)
        assert processor.validate_bbox([100, 100, 200, 200]) is True

    def test_validate_bbox_with_image_size(self, mock_vlm_client):
        """Test validating bbox with image size"""
        processor = ConcreteProcessor(mock_vlm_client)
        assert processor.validate_bbox([100, 100, 200, 200], 640, 480) is True

    def test_validate_bbox_invalid_order(self, mock_vlm_client):
        """Test validating bbox with wrong order"""
        processor = ConcreteProcessor(mock_vlm_client)
        assert processor.validate_bbox([200, 200, 100, 100]) is False

    def test_validate_bbox_negative(self, mock_vlm_client):
        """Test validating bbox with negative values"""
        processor = ConcreteProcessor(mock_vlm_client)
        assert processor.validate_bbox([-10, 100, 200, 200]) is False

    def test_validate_bbox_all_zeros(self, mock_vlm_client):
        """Test validating bbox with all zeros"""
        processor = ConcreteProcessor(mock_vlm_client)
        assert processor.validate_bbox([0, 0, 0, 0]) is False

    def test_validate_bbox_wrong_type(self, mock_vlm_client):
        """Test validating bbox with wrong type"""
        processor = ConcreteProcessor(mock_vlm_client)
        assert processor.validate_bbox("not a list") is False

    def test_validate_bbox_wrong_length(self, mock_vlm_client):
        """Test validating bbox with wrong length"""
        processor = ConcreteProcessor(mock_vlm_client)
        assert processor.validate_bbox([100, 200]) is False

    def test_validate_bbox_none(self, mock_vlm_client):
        """Test validating None bbox"""
        processor = ConcreteProcessor(mock_vlm_client)
        assert processor.validate_bbox(None) is False

    def test_normalize_response_empty(self, mock_vlm_client):
        """Test normalizing empty response"""
        processor = ConcreteProcessor(mock_vlm_client)
        result = processor.normalize_response({})
        assert result["l1_result"] == []
        assert result["l2_result"] == ""
        assert result["l3_result"] == ""

    def test_normalize_response_with_data(self, mock_vlm_client):
        """Test normalizing response with data"""
        processor = ConcreteProcessor(mock_vlm_client)
        result = processor.normalize_response({
            "l1_result": [{"bbox": [1, 2, 3, 4]}],
            "l2_result": "test",
            "l3_result": "risk"
        })
        assert len(result["l1_result"]) == 1

    def test_normalize_response_dict_to_list(self, mock_vlm_client):
        """Test normalizing response with dict l1_result"""
        processor = ConcreteProcessor(mock_vlm_client)
        result = processor.normalize_response({
            "l1_result": {"bbox": [1, 2, 3, 4], "category": "test"}
        })
        assert isinstance(result["l1_result"], list)
        assert len(result["l1_result"]) == 1

    def test_normalize_response_add_scene_id(self, mock_vlm_client):
        """Test normalizing response adds scene_id"""
        processor = ConcreteProcessor(mock_vlm_client)
        result = processor.normalize_response({
            "l1_result": [{"bbox": [1, 2, 3, 4]}]
        }, scene_id=2)
        assert result["l1_result"][0]["scene_id"] == 2

    def test_normalize_response_missing_fields(self, mock_vlm_client):
        """Test normalizing response adds missing fields"""
        processor = ConcreteProcessor(mock_vlm_client)
        result = processor.normalize_response({
            "l1_result": [{}]
        })
        assert result["l1_result"][0]["bbox"] == [0, 0, 0, 0]
        assert result["l1_result"][0]["category"] == ""

    def test_normalize_response_non_string_l2(self, mock_vlm_client):
        """Test normalizing response with non-string l2_result"""
        processor = ConcreteProcessor(mock_vlm_client)
        result = processor.normalize_response({
            "l2_result": 123,
            "l3_result": None
        })
        assert result["l2_result"] == "123"
        assert result["l3_result"] == "None"