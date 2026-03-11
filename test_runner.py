#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量测试脚本 - 高速公路病害检测系统

功能:
1. 遍历竞赛图片视频材料目录
2. 根据目录名确定scene_id
3. 调用检测处理器进行处理
4. 保存结果JSON到 output/{Task}/results/ 目录
5. 生成带标注的图片/视频到 output/{Task}/annotated/ 目录

用法:
    python test_runner.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--api-url API_URL]
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from vlm_client import get_vlm_client, reset_vlm_client
from processors import Task1Processor, Task2Processor, Task3Processor, Task4Processor


# 目录到scene_id映射
TASK_SCENE_MAP = {
    'Task_1': [0],                    # 抛洒物
    'Task_2': [1],                    # 违停
    'Task_2-1': [1],                  # 违停
    'Task_3': [2, 3, 4, 5],           # 路面病害（需要根据子目录确定）
    'Task_3/路面裂缝': [2],           # 裂缝
    'Task_3/路面坑洼': [3],           # 坑洼
    'Task_3/路面积水': [4],           # 积水
    'Task_3/护栏破损': [5],           # 护栏破损
    'Task_4': [6, 7, 8],              # 路外病害（需要根据子目录确定）
    'Task_4/边坡异常': [6],           # 边坡滑坡
    'Task_4/排水沟积水': [7],         # 排水沟积水
    'Task_4/排水沟破损': [8],         # 排水沟破损
    'Task_4/标志牌异常': [8],         # 标志牌异常（归类为路外病害）
}

# 图片/视频扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}


def get_scene_id(relative_path: str) -> Optional[int]:
    """
    根据相对路径确定scene_id

    Args:
        relative_path: 相对于输入目录的路径

    Returns:
        scene_id 或 None
    """
    # 尝试精确匹配
    for path_pattern, scene_ids in TASK_SCENE_MAP.items():
        if path_pattern in relative_path:
            # 如果只返回一个scene_id，直接使用
            if len(scene_ids) == 1:
                return scene_ids[0]
            # 如果有多个，返回第一个（后续需要用户确认）
            return scene_ids[0]

    return None


def is_image_file(file_path: str) -> bool:
    """判断是否为图片文件"""
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(file_path: str) -> bool:
    """判断是否为视频文件"""
    return Path(file_path).suffix.lower() in VIDEO_EXTENSIONS


def get_processor(scene_id: int):
    """获取对应的处理器"""
    vlm_client = get_vlm_client()

    if scene_id == 0:  # 抛洒物
        return Task1Processor(vlm_client)
    elif scene_id == 1:  # 违停
        return Task2Processor(vlm_client)
    elif scene_id in [2, 3, 4, 5]:  # 路面护栏病害
        return Task3Processor(vlm_client)
    elif scene_id in [6, 7, 8]:  # 路外病害
        return Task4Processor(vlm_client)
    else:
        raise ValueError(f"未知的场景ID: {scene_id}")


def process_file(
    file_path: str,
    scene_id: int,
    output_dir: str
) -> Dict[str, Any]:
    """
    处理单个文件

    Args:
        file_path: 输入文件路径
        scene_id: 场景ID
        output_dir: 输出目录

    Returns:
        处理结果
    """
    input_path = Path(file_path)
    task_name = input_path.parent.name
    if task_name not in ['Task_1', 'Task_2', 'Task_2-1', 'Task_3', 'Task_4']:
        task_name = input_path.parent.parent.name  # 处理子目录情况

    # 创建输出目录
    results_dir = os.path.join(output_dir, task_name, 'results')
    annotated_dir = os.path.join(output_dir, task_name, 'annotated')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    # 获取处理器
    processor = get_processor(scene_id)

    # 处理文件
    start_time = time.time()

    try:
        if is_video_file(file_path):
            # 视频处理（Task 2）
            result = processor.process(file_path, scene_id)

            # 确保标注视频路径正确
            if result.get("l1_result", {}).get("annotated_video"):
                # 复制标注视频到输出目录
                src_video = result["l1_result"]["annotated_video"]
                if os.path.exists(src_video):
                    import shutil
                    dst_video = os.path.join(annotated_dir, f"{input_path.stem}_annotated.mp4")
                    shutil.copy(src_video, dst_video)
                    result["l1_result"]["annotated_video"] = dst_video

        else:
            # 图片处理
            result = processor.process(file_path, scene_id, annotated_dir)

        elapsed_time = time.time() - start_time

        # 构建响应
        response = {
            "success": True,
            "message": "",
            "scene_id": scene_id,
            "data": result,
            "meta": {
                "input_file": str(file_path),
                "processing_time": round(elapsed_time, 2),
                "timestamp": datetime.now().isoformat()
            }
        }

        # 保存JSON结果
        result_file = os.path.join(results_dir, f"{input_path.stem}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        print(f"[OK] {file_path} -> {result_file} ({elapsed_time:.2f}s)")

        return response

    except Exception as e:
        import traceback
        error_response = {
            "success": False,
            "message": str(e),
            "scene_id": scene_id,
            "data": {},
            "meta": {
                "input_file": str(file_path),
                "error_traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
        }

        # 保存错误结果
        result_file = os.path.join(results_dir, f"{input_path.stem}_error.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(error_response, f, ensure_ascii=False, indent=2)

        print(f"[FAIL] {file_path} -> ERROR: {str(e)}")

        return error_response


def run_tests(
    input_dir: str,
    output_dir: str,
    limit: Optional[int] = None,
    force_task: Optional[str] = None
) -> Dict[str, Any]:
    """
    运行所有测试

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        limit: 限制处理的文件数量（用于测试）
        force_task: 强制指定任务名称（如 Task_1）

    Returns:
        测试统计信息
    """
    input_path = Path(input_dir)
    all_files = []

    # 收集所有文件
    for ext in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        all_files.extend(input_path.glob(f"**/*{ext}"))
        all_files.extend(input_path.glob(f"**/*{ext.upper()}"))

    # 排除压缩文件
    all_files = [f for f in all_files if f.suffix.lower() not in ['.zip']]

    # 按Task排序
    all_files.sort(key=lambda x: str(x))

    # 限制数量
    if limit:
        all_files = all_files[:limit]

    # 统计信息
    stats = {
        "total": len(all_files),
        "success": 0,
        "failed": 0,
        "by_task": {},
        "start_time": datetime.now().isoformat(),
        "end_time": None
    }

    print(f"\n{'='*60}")
    print(f"开始批量测试: {len(all_files)} 个文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")

    # 处理每个文件
    for i, file_path in enumerate(all_files, 1):
        relative_path = str(file_path.relative_to(input_path))

        # 确定scene_id
        scene_id = get_scene_id(relative_path)

        # 如果无法从相对路径确定scene_id，尝试从force_task确定
        if scene_id is None and force_task:
            # 从force_task获取scene_id
            if force_task in TASK_SCENE_MAP:
                scene_id = TASK_SCENE_MAP[force_task][0]
            else:
                # 尝试匹配目录名
                for pattern, ids in TASK_SCENE_MAP.items():
                    if force_task in pattern or pattern in force_task:
                        scene_id = ids[0]
                        break

        if scene_id is None:
            print(f"[{i}/{len(all_files)}] 跳过 {relative_path} (无法确定scene_id)")
            continue

        print(f"[{i}/{len(all_files)}] 处理 {relative_path} (scene_id={scene_id})")

        # 处理文件
        result = process_file(str(file_path), scene_id, output_dir)

        # 更新统计
        task_name = file_path.parent.name
        if task_name not in ['Task_1', 'Task_2', 'Task_2-1', 'Task_3', 'Task_4']:
            task_name = file_path.parent.parent.name

        if task_name not in stats["by_task"]:
            stats["by_task"][task_name] = {"total": 0, "success": 0, "failed": 0}

        stats["by_task"][task_name]["total"] += 1

        if result.get("success"):
            stats["success"] += 1
            stats["by_task"][task_name]["success"] += 1
        else:
            stats["failed"] += 1
            stats["by_task"][task_name]["failed"] += 1

    stats["end_time"] = datetime.now().isoformat()

    # 保存统计信息
    stats_file = os.path.join(output_dir, "test_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n{'='*60}")
    print("测试完成!")
    print(f"总计: {stats['total']} 个文件")
    print(f"成功: {stats['success']} 个")
    print(f"失败: {stats['failed']} 个")
    print(f"\n各任务统计:")
    for task, task_stats in stats["by_task"].items():
        print(f"  {task}: {task_stats['success']}/{task_stats['total']} 成功")
    print(f"\n统计信息已保存到: {stats_file}")
    print(f"{'='*60}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="高速公路病害检测批量测试脚本")
    parser.add_argument(
        '--input', '-i',
        default='竞赛图片视频材料',
        help='输入目录路径 (默认: 竞赛图片视频材料)'
    )
    parser.add_argument(
        '--output', '-o',
        default='output',
        help='输出目录路径 (默认: output)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='限制处理的文件数量 (用于测试)'
    )
    parser.add_argument(
        '--task', '-t',
        type=str,
        default=None,
        help='只处理指定Task (如: Task_1, Task_2)'
    )

    args = parser.parse_args()

    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 初始化VLM客户端
    print("正在初始化VLM客户端...")
    try:
        get_vlm_client()
        print("VLM客户端初始化成功\n")
    except Exception as e:
        print(f"VLM客户端初始化失败: {e}")
        sys.exit(1)

    # 如果指定了特定Task
    if args.task:
        task_dir = os.path.join(args.input, args.task)
        if os.path.exists(task_dir):
            run_tests(task_dir, args.output, args.limit, force_task=args.task)
        else:
            print(f"错误: Task目录不存在: {task_dir}")
            sys.exit(1)
    else:
        run_tests(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()