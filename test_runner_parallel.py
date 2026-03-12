#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并行测试脚本 - 支持多VLM服务并行处理

功能:
1. 支持多个VLM服务同时处理不同文件
2. 负载均衡分配任务
3. 进程级并行，每个进程使用独立的VLM客户端

用法:
    python test_runner_parallel.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--workers N]
"""
import os
import sys
import json
import argparse
import time
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


# 目录到scene_id映射
TASK_SCENE_MAP = {
    'Task_1': [0],
    'Task_2': [1],
    'Task_2-1': [1],
    'Task_3': [2, 3, 4, 5],
    'Task_3/路面裂缝': [2],
    'Task_3/路面坑洼': [3],
    'Task_3/路面积水': [4],
    'Task_3/护栏破损': [5],
    'Task_4': [6, 7, 8],
    'Task_4/边坡异常': [6],
    'Task_4/排水沟积水': [7],
    'Task_4/排水沟破损': [8],
    'Task_4/标志牌异常': [8],
}

# 图片/视频扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}


def get_scene_id(relative_path: str) -> Optional[int]:
    """根据相对路径确定scene_id"""
    for path_pattern, scene_ids in TASK_SCENE_MAP.items():
        if path_pattern in relative_path:
            if len(scene_ids) == 1:
                return scene_ids[0]
            return scene_ids[0]
    return None


def is_image_file(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in VIDEO_EXTENSIONS


def process_file_with_vlm(
    file_path: str,
    scene_id: int,
    output_dir: str,
    vlm_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    使用指定VLM配置处理单个文件

    Args:
        file_path: 输入文件路径
        scene_id: 场景ID
        output_dir: 输出目录
        vlm_config: VLM配置 (包含name, api_key, base_url, model等)

    Returns:
        处理结果
    """
    # 在子进程中创建独立的VLM客户端
    from openai import OpenAI

    input_path = Path(file_path)
    task_name = input_path.parent.name
    if task_name not in ['Task_1', 'Task_2', 'Task_2-1', 'Task_3', 'Task_4']:
        task_name = input_path.parent.parent.name

    # 创建输出目录
    results_dir = os.path.join(output_dir, task_name, 'results')
    annotated_dir = os.path.join(output_dir, task_name, 'annotated')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    # 创建VLM客户端
    vlm_name = vlm_config.get('name', 'unknown')
    client = OpenAI(
        api_key=vlm_config.get('api_key'),
        base_url=vlm_config.get('base_url'),
        timeout=vlm_config.get('timeout', 300)
    )
    model = vlm_config.get('model')
    max_tokens = vlm_config.get('max_tokens', 4096)
    temperature = vlm_config.get('temperature', 0.0)

    # 获取处理器
    from processors import Task1Processor, Task2Processor, Task3Processor, Task4Processor
    from vlm_client import VLMClient

    # 创建VLMClient实例
    vlm_client = VLMClient(
        api_key=vlm_config.get('api_key'),
        base_url=vlm_config.get('base_url'),
        model=model,
        provider='openai'
    )

    if scene_id == 0:
        processor = Task1Processor(vlm_client)
    elif scene_id == 1:
        processor = Task2Processor(vlm_client)
    elif scene_id in [2, 3, 4, 5]:
        processor = Task3Processor(vlm_client)
    elif scene_id in [6, 7, 8]:
        processor = Task4Processor(vlm_client)
    else:
        raise ValueError(f"未知的场景ID: {scene_id}")

    start_time = time.time()
    vlm_tag = f"[{vlm_name}]"

    try:
        if is_video_file(file_path):
            result = processor.process(file_path, scene_id, annotated_dir)
        else:
            result = processor.process(file_path, scene_id, annotated_dir)

        elapsed_time = time.time() - start_time

        response = {
            "success": True,
            "message": "",
            "scene_id": scene_id,
            "vlm_service": vlm_name,
            "data": result,
            "meta": {
                "input_file": str(file_path),
                "processing_time": round(elapsed_time, 2),
                "timestamp": datetime.now().isoformat()
            }
        }

        result_file = os.path.join(results_dir, f"{input_path.stem}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        print(f"{vlm_tag} [OK] {input_path.name} ({elapsed_time:.2f}s)")
        return response

    except Exception as e:
        import traceback
        elapsed_time = time.time() - start_time
        error_response = {
            "success": False,
            "message": str(e),
            "scene_id": scene_id,
            "vlm_service": vlm_name,
            "data": {},
            "meta": {
                "input_file": str(file_path),
                "processing_time": round(elapsed_time, 2),
                "error_traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
        }

        result_file = os.path.join(results_dir, f"{input_path.stem}_error.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(error_response, f, ensure_ascii=False, indent=2)

        print(f"{vlm_tag} [FAIL] {input_path.name}: {str(e)}")
        return error_response


def run_parallel_tests(
    input_dir: str,
    output_dir: str,
    vlm_services: List[Dict[str, Any]],
    max_workers: Optional[int] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    并行运行测试

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        vlm_services: VLM服务配置列表
        max_workers: 最大并行worker数 (默认等于VLM服务数)
        limit: 限制处理的文件数量

    Returns:
        测试统计信息
    """
    input_path = Path(input_dir)
    all_files = []

    # 收集所有文件
    for ext in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        all_files.extend(input_path.glob(f"**/*{ext}"))
        all_files.extend(input_path.glob(f"**/*{ext.upper()}"))

    all_files = [f for f in all_files if f.suffix.lower() not in ['.zip']]
    all_files.sort(key=lambda x: str(x))

    if limit:
        all_files = all_files[:limit]

    # 确定worker数量
    num_vlms = len(vlm_services)
    if max_workers is None:
        max_workers = num_vlms
    max_workers = min(max_workers, num_vlms)

    # 准备任务列表 (文件, scene_id, VLM配置索引)
    tasks = []
    vlm_index = 0

    for file_path in all_files:
        relative_path = str(file_path.relative_to(input_path))
        scene_id = get_scene_id(relative_path)

        if scene_id is None:
            print(f"跳过 {relative_path} (无法确定scene_id)")
            continue

        # 轮询分配VLM服务
        vlm_config = vlm_services[vlm_index % num_vlms]
        tasks.append((str(file_path), scene_id, output_dir, vlm_config))
        vlm_index += 1

    # 统计信息
    stats = {
        "total": len(tasks),
        "success": 0,
        "failed": 0,
        "vlm_stats": {s.get('name', f'vlm_{i}'): {"total": 0, "success": 0, "failed": 0}
                      for i, s in enumerate(vlm_services)},
        "start_time": datetime.now().isoformat(),
        "end_time": None
    }

    print(f"\n{'='*60}")
    print(f"开始并行测试: {len(tasks)} 个文件")
    print(f"VLM服务数: {num_vlms}")
    print(f"并行worker数: {max_workers}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")

    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_file_with_vlm,
                task[0], task[1], task[2], task[3]
            ): task for task in tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                vlm_name = result.get("vlm_service", "unknown")

                if vlm_name in stats["vlm_stats"]:
                    stats["vlm_stats"][vlm_name]["total"] += 1

                if result.get("success"):
                    stats["success"] += 1
                    if vlm_name in stats["vlm_stats"]:
                        stats["vlm_stats"][vlm_name]["success"] += 1
                else:
                    stats["failed"] += 1
                    if vlm_name in stats["vlm_stats"]:
                        stats["vlm_stats"][vlm_name]["failed"] += 1

            except Exception as e:
                print(f"[ERROR] 任务执行失败: {e}")
                stats["failed"] += 1

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
    print(f"\n各VLM服务统计:")
    for vlm_name, vlm_stats in stats["vlm_stats"].items():
        if vlm_stats["total"] > 0:
            print(f"  {vlm_name}: {vlm_stats['success']}/{vlm_stats['total']} 成功")
    print(f"\n统计信息已保存到: {stats_file}")
    print(f"{'='*60}\n")

    return stats


def clear_output_dir(output_dir: str):
    """清空输出目录"""
    import shutil
    if os.path.exists(output_dir):
        print(f"清空输出目录: {output_dir}")
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    os.makedirs(output_dir, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="高速公路病害检测并行测试脚本")
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
        '--workers', '-w',
        type=int,
        default=2,  # 默认2个并行worker，RT-DETR+ByteTrack内存占用较大
        help='最大并行worker数 (默认2)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='限制处理的文件数量 (用于测试)'
    )

    args = parser.parse_args()

    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        sys.exit(1)

    # 清空并创建输出目录
    clear_output_dir(args.output)

    # 加载VLM服务配置
    from config import config
    vlm_services = config.vlms

    if not vlm_services:
        print("错误: 未配置多VLM服务 (vlms)")
        print("请在config.yaml中配置vlms列表")
        sys.exit(1)

    print(f"发现 {len(vlm_services)} 个VLM服务:")
    for i, vlm in enumerate(vlm_services):
        print(f"  [{i+1}] {vlm.get('name', 'unnamed')}: {vlm.get('base_url')} - {vlm.get('model')}")
    print()

    # 运行并行测试
    run_parallel_tests(
        args.input,
        args.output,
        vlm_services,
        args.workers,
        args.limit
    )


if __name__ == "__main__":
    # Windows多进程支持
    mp.freeze_support()
    main()