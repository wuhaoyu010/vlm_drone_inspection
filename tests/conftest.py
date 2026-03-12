"""
Test fixtures for highway detection system
"""
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_vlm_client():
    """Mock VLM client for testing"""
    client = MagicMock()
    client.chat_with_image = MagicMock(return_value={
        "content": '{"l1_result": [], "l2_result": "test", "l3_result": "test"}',
        "processed_dimensions": (640, 480)
    })
    return client


@pytest.fixture
def mock_vlm_client_with_detection():
    """Mock VLM client with detection results"""
    client = MagicMock()
    client.chat_with_image = MagicMock(return_value={
        "content": '''```json
{
    "l1_result": [
        {"bbox": [100, 100, 200, 200], "category": "抛洒物"}
    ],
    "l2_result": "检测到1个抛洒物",
    "l3_result": "风险等级：P1"
}
```''',
        "processed_dimensions": (640, 480)
    })
    return client


@pytest.fixture
def mock_vlm_client_violation():
    """Mock VLM client with violation detection"""
    client = MagicMock()
    client.chat_with_image = MagicMock(return_value={
        "content": '''```json
{
    "l1_result": {
        "has_violation": true,
        "violations": [
            {
                "bbox": [100, 100, 200, 200],
                "vehicle_type": "小型轿车",
                "parking_location": "应急车道",
                "has_hazard_lights": false,
                "has_warning_sign": false,
                "is_emergency_vehicle": false,
                "violation_reason": "未开启双闪灯"
            }
        ]
    },
    "l2_result": "检测到1辆违停车辆",
    "l3_result": "风险等级：P1"
}
```''',
        "processed_dimensions": (640, 480)
    })
    return client


@pytest.fixture
def mock_vlm_client_for_sahi():
    """Mock VLM client for SAHI tests"""
    client = MagicMock()
    return client


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample image file for testing"""
    from PIL import Image
    img_path = tmp_path / "test_image.jpg"
    img = Image.new('RGB', (640, 480), color='red')
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def sample_video_path(tmp_path):
    """Create a sample video file for testing"""
    import cv2
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    # Write 30 frames (1 second)
    for _ in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return str(video_path)


@pytest.fixture
def sample_detections():
    """Sample YOLO detections"""
    return np.array([
        [100, 100, 200, 200, 0.9, 2],  # car
        [300, 300, 400, 400, 0.8, 5],  # bus
        [500, 500, 600, 600, 0.7, 7],  # truck
    ], dtype=np.float32)


@pytest.fixture
def sample_bbox():
    """Sample bounding box"""
    return [100, 100, 200, 200]


@pytest.fixture
def sample_normalized_bbox():
    """Sample normalized bounding box [0-1000]"""
    return [156, 208, 312, 416]  # roughly corresponds to 100,100,200,200 for 640x480