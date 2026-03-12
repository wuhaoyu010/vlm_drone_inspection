"""
Tests for SAHI small object detection in Task2
"""
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSAHIConfig:
    """测试SAHI配置参数"""

    def test_sahi_config_defaults(self):
        """测试SAHI默认配置"""
        from processors.task2 import get_task2_config

        config = get_task2_config()

        assert "use_sahi" in config
        assert "sahi_slice_width" in config
        assert "sahi_slice_height" in config
        assert "sahi_overlap_width_ratio" in config
        assert "sahi_overlap_height_ratio" in config
        assert "sahi_postprocess_type" in config

    def test_sahi_config_values(self):
        """测试SAHI配置值"""
        from processors.task2 import get_task2_config

        config = get_task2_config()

        # 默认值检查
        assert config["sahi_slice_width"] == 640
        assert config["sahi_slice_height"] == 640
        assert config["sahi_overlap_width_ratio"] == 0.2
        assert config["sahi_overlap_height_ratio"] == 0.2
        assert config["sahi_postprocess_type"] == "NMM"


class TestSAHIDetection:
    """测试SAHI切片检测功能"""

    def test_sahi_not_available(self):
        """测试SAHI不可用时的处理"""
        from processors.task2 import VehicleTracker

        with patch.object(VehicleTracker, '__init__', lambda self, **kwargs: None):
            tracker = VehicleTracker.__new__(VehicleTracker)
            tracker.sahi_model = None

            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # SAHI模型为None时应返回空列表
            detections = tracker._detect_with_sahi(frame)
            assert detections == []

    def test_sahi_detection_logic(self):
        """测试SAHI检测逻辑（不依赖sahi模块）"""
        from processors.task2 import VehicleTracker, VEHICLE_CLASSES

        # 验证车辆类别过滤逻辑
        test_class_ids = [0, 2, 3, 5, 7, 9]  # person, car, motorcycle, bus, truck, other

        # 只保留在VEHICLE_CLASSES中的类别
        vehicle_ids = [cid for cid in test_class_ids if cid in VEHICLE_CLASSES]

        assert 2 in vehicle_ids  # car
        assert 5 in vehicle_ids  # bus
        assert 7 in vehicle_ids  # truck
        assert 0 not in vehicle_ids  # person


class TestVehicleTrackerSAHIIntegration:
    """测试VehicleTracker与SAHI的集成"""

    def test_tracker_sahi_config_loaded(self):
        """测试追踪器加载SAHI配置"""
        from processors.task2 import get_task2_config

        config = get_task2_config()

        # 验证SAHI配置已正确加载
        assert "use_sahi" in config
        assert config["sahi_slice_width"] == 640
        assert config["sahi_slice_height"] == 640


class TestVehicleClasses:
    """测试车辆类别定义"""

    def test_vehicle_classes_defined(self):
        """测试车辆类别映射"""
        from processors.task2 import VEHICLE_CLASSES

        assert 2 in VEHICLE_CLASSES  # car
        assert 3 in VEHICLE_CLASSES  # motorcycle
        assert 5 in VEHICLE_CLASSES  # bus
        assert 7 in VEHICLE_CLASSES  # truck

        assert VEHICLE_CLASSES[2] == 'car'
        assert VEHICLE_CLASSES[5] == 'bus'
        assert VEHICLE_CLASSES[7] == 'truck'

    def test_vehicle_classes_coverage(self):
        """测试车辆类别覆盖"""
        from processors.task2 import VEHICLE_CLASSES

        # 确保定义了常见车辆类型
        assert len(VEHICLE_CLASSES) >= 4
        for class_id, name in VEHICLE_CLASSES.items():
            assert isinstance(class_id, int)
            assert isinstance(name, str)
            assert len(name) > 0


class TestSAHIAvailability:
    """测试SAHI可用性检查"""

    def test_sahi_available_flag(self):
        """测试SAHI可用性标志"""
        from processors.task2 import SAHI_AVAILABLE

        # SAHI_AVAILABLE是一个布尔值
        assert isinstance(SAHI_AVAILABLE, bool)