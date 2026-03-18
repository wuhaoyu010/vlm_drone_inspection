"""
任务处理器模块
"""
from .task1 import Task1Processor
from .task3 import Task3Processor
from .task4 import Task4Processor

# Task2需要ultralytics，可选导入
try:
    from .task2 import Task2Processor
except ImportError:
    Task2Processor = None

__all__ = ['Task1Processor', 'Task2Processor', 'Task3Processor', 'Task4Processor']
