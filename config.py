"""
项目配置文件 - YAML格式
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# 配置文件路径
CONFIG_FILE = Path(__file__).parent / "config.yaml"


class Config:
    """配置管理类"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load_config()
        return cls._instance
    
    @classmethod
    def _load_config(cls):
        """加载配置文件"""
        if cls._config is not None:
            return
        
        # 尝试多个配置文件路径
        config_paths = [
            CONFIG_FILE,
            Path("config.yaml"),
            Path("d:/Projects/比赛相关/高速公路检测/config.yaml"),
        ]
        
        config_data = None
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                break
        
        if config_data is None:
            # 使用默认配置
            config_data = cls._get_default_config()
        
        cls._config = config_data
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "server": {
                "host": "0.0.0.0",
                "port": 8000
            },
            "vlm": {
                "provider": "openai",
                "api_key": os.environ.get("VLM_API_KEY", ""),
                "base_url": os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1"),
                "model": os.environ.get("VLM_MODEL", "gpt-4o"),
                "max_tokens": 4096,
                "temperature": 0.1,
                "timeout": 120
            },
            "detection": {
                "iou_threshold": 0.5,
                "confidence_threshold": 0.3
            },
            "scenes": {
                0: "路面抛洒物",
                1: "车辆违停",
                2: "路面裂缝",
                3: "路面坑洼",
                4: "路面积水",
                5: "护栏破损",
                6: "边坡滑坡",
                7: "排水沟积水",
                8: "排水沟破损"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    @property
    def server(self) -> Dict[str, Any]:
        return self._config.get("server", {})
    
    @property
    def vlm(self) -> Dict[str, Any]:
        return self._config.get("vlm", {})

    @property
    def vlms(self) -> list:
        """获取多VLM配置列表"""
        return self._config.get("vlms", [])

    @property
    def detection(self) -> Dict[str, Any]:
        return self._config.get("detection", {})
    
    @property
    def scenes(self) -> Dict[int, str]:
        return self._config.get("scenes", {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        return self._config.get("logging", {})

    @property
    def models(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self._config.get("models", {})

    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """获取指定Task的模型配置

        Args:
            task_name: 任务名称，如 "task1", "task2", "task3", "task4"

        Returns:
            该Task的模型配置字典
        """
        return self.models.get(task_name, {})

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value


# 创建全局配置实例
config = Config()

# 便捷访问
HOST = config.server.get("host", "0.0.0.0")
PORT = config.server.get("port", 8000)

VLM_PROVIDER = config.vlm.get("provider", "openai")
VLM_API_KEY = config.vlm.get("api_key", "") or os.environ.get("VLM_API_KEY", "")
VLM_BASE_URL = config.vlm.get("base_url", "https://api.openai.com/v1")
VLM_MODEL = config.vlm.get("model", "gpt-4o")
VLM_MAX_TOKENS = config.vlm.get("max_tokens", 4096)
VLM_TEMPERATURE = config.vlm.get("temperature", 0.1)
VLM_TIMEOUT = config.vlm.get("timeout", 120)

# 多VLM配置列表
VLM_SERVICES = config.vlms

SCENE_MAPPING = config.scenes
IOU_THRESHOLD = config.detection.get("iou_threshold", 0.5)

# Task到scene_id的映射
TASK_SCENE_MAP = {
    1: [0],          # Task1: 抛洒物
    2: [1],          # Task2: 违停
    3: [2, 3, 4, 5], # Task3: 路面裂缝/坑洼/积水/护栏破损
    4: [6, 7, 8]     # Task4: 边坡滑坡/排水沟积水/排水沟破损
}
