"""
FastAPI 服务主程序
"""

import os
import tempfile
import time
import uuid
import asyncio
from functools import wraps
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import HOST, PORT, SCENE_MAPPING
from vlm_client import get_vlm_client
from processors import Task1Processor, Task3Processor, Task4Processor

# 创建FastAPI应用
app = FastAPI(
    title="高速公路病害检测API",
    description="基于VLM的高速公路病害检测与诊断系统",
    version="1.0.0",
)

# 初始化处理器
processors = {}

# 超时控制装饰器（复赛要求：所有场景60秒超时）
def timeout_handler(seconds: int):
    """超时控制装饰器

    Args:
        seconds: 超时秒数（复赛要求60秒）

    Raises:
        HTTPException: 超时时返回408错误
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=408,
                    detail=f"处理超时（超过{seconds}秒限制），该样本L1/L2/L3将判0分"
                )
        return wrapper
    return decorator


def get_processor(scene_id: int):
    """获取对应的处理器"""
    vlm_client = get_vlm_client()

    # 场景编号映射（符合复赛评分细则）
    # Scene0: 抛洒物 (Task1)
    # Scene1-4: 路面病害 (Task2: 裂缝/坑洼/积水/护栏破损)
    # Scene5-7: 路外病害 (Task3: 边坡破损/排水沟积水/排水沟破损)
    if scene_id == 0:  # 抛洒物
        return Task1Processor(vlm_client)
    elif scene_id in [1, 2, 3, 4]:  # 路面病害（裂缝/坑洼/积水/护栏破损）
        return Task3Processor(vlm_client)
    elif scene_id in [5, 6, 7]:  # 路外病害（边坡破损/排水沟积水/排水沟破损）
        return Task4Processor(vlm_client)
    else:
        raise HTTPException(status_code=400, detail=f"无效的场景ID: {scene_id}。有效范围: 0-7")


class DetectionResponse(BaseModel):
    """检测响应模型"""

    success: bool
    message: str
    scene_id: int
    data: dict


@app.on_event("startup")
async def startup_event():
    """启动时初始化 - 预热所有组件"""
    start_time = time.time()

    print("\n" + "=" * 60)
    print("正在预热服务组件...")
    print("=" * 60)

    # 1. 初始化 VLM 客户端
    print("\n[1/5] 初始化 VLM 客户端...")
    try:
        get_vlm_client()
        print("      [OK] VLM 客户端初始化成功")
    except Exception as e:
        print(f"      [FAIL] VLM 客户端初始化失败: {e}")
        return

    # 2. 初始化 RAG 知识库
    print("\n[2/5] 初始化 RAG 知识库...")
    try:
        from rag_knowledge import get_knowledge_base, check_rag_availability

        rag_status = check_rag_availability()
        if rag_status.get("rag_enabled"):
            kb = get_knowledge_base()
            if kb.is_available():
                print("      [OK] RAG 知识库初始化成功")
            else:
                print("      [WARN] RAG 知识库未就绪，将使用默认模板")
        else:
            print("      [WARN] RAG 未启用")
    except Exception as e:
        print(f"      [WARN] RAG 初始化失败: {e}")

    # 3. 预热所有处理器（触发模型加载）
    print("\n[3/5] 预热处理器（加载模型）...")
    try:
        vlm_client = get_vlm_client()

        # Task1 处理器（抛洒物）
        from processors import Task1Processor

        processors[0] = Task1Processor(vlm_client)
        print("      [OK] Task1 处理器已加载")

        # Task3 处理器（路面病害）
        from processors import Task3Processor

        processors[2] = Task3Processor(vlm_client)
        print("      [OK] Task3 处理器已加载（路面病害检测）")

        # Task4 处理器（路外病害）
        from processors import Task4Processor

        processors[3] = Task4Processor(vlm_client)
        print("      [OK] Task4 处理器已加载（路外病害检测）")

    except Exception as e:
        print(f"      [FAIL] 处理器预热失败: {e}")
        import traceback

        traceback.print_exc()

    # 4. 可选：发送一个预热请求到 VLM
    print("\n[4/5] 预热 VLM 连接...")
    try:
        vlm_client.chat("你好", "你是一个助手。")
        print("      [OK] VLM 连接已建立")
    except Exception as e:
        print(f"      [WARN] VLM 连接预热失败: {e}")

    # 5. 显示内存使用情况
    print("\n[5/5] 检查系统状态...")
    try:
        import psutil

        mem = psutil.virtual_memory()
        print(
            f"      内存使用: {mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB ({mem.percent}%)"
        )
    except ImportError:
        print("      (psutil 未安装，跳过内存检查)")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"预热完成！耗时: {elapsed:.2f} 秒")
    print("=" * 60 + "\n")


@app.get("/")
async def root():
    """根路径"""
    return {"message": "高速公路病害检测API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/v1/detect", response_model=DetectionResponse)
async def detect(
    image_or_video: UploadFile = File(..., description="上传的图片或视频文件"),
    scene_id: int = Form(
        ...,
        description="场景ID: 0-抛洒物, 1-裂缝, 2-坑洼, 3-积水, 4-护栏破损, 5-边坡破损, 6-排水沟积水, 7-排水沟破损",
    ),
):
    """
    检测接口

    - **image_or_video**: 上传的图片或视频文件
    - **scene_id**: 场景ID
        - 0: 路面抛洒物
        - 1: 路面裂缝
        - 2: 路面坑洼
        - 3: 路面积水
        - 4: 护栏破损
        - 5: 边坡破损
        - 6: 排水沟积水
        - 7: 排水沟破损
    """
    try:
        # 验证scene_id
        if scene_id not in SCENE_MAPPING:
            raise HTTPException(status_code=400, detail=f"无效的场景ID: {scene_id}")

        # 保存上传的文件
        file_ext = os.path.splitext(image_or_video.filename)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)

        try:
            content = await image_or_video.read()
            temp_file.write(content)
            temp_file.close()

            # 获取处理器并处理
            processor = get_processor(scene_id)
            result = processor.process(temp_file.name, scene_id)

            return DetectionResponse(
                success=True, message="检测完成", scene_id=scene_id, data=result
            )

        finally:
            # 清理临时文件
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/v1/detect/path")
async def detect_by_path(
    file_path: str = Form(..., description="图片或视频的本地路径"),
    scene_id: int = Form(..., description="场景ID"),
):
    """
    通过文件路径检测接口（用于本地测试）
    """
    try:
        # 验证文件存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail=f"文件不存在: {file_path}")

        # 验证scene_id
        if scene_id not in SCENE_MAPPING:
            raise HTTPException(status_code=400, detail=f"无效的场景ID: {scene_id}")

        # 获取处理器并处理
        processor = get_processor(scene_id)
        result = processor.process(file_path, scene_id)

        if scene_id == 1:
            return result  # 违停场景直接返回原始结果（兼容比赛要求）

        return DetectionResponse(
            success=True, message="检测完成", scene_id=scene_id, data=result
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


# 兼容比赛要求的接口格式
@app.post("/api/v1/")
@timeout_handler(60)  # 复赛要求：所有场景60秒超时
async def detect_compatible(
    image_or_video: UploadFile = File(..., description="上传的图片或视频文件"),
    scene_id: int = Form(..., description="场景ID"),
):
    """
    兼容比赛要求的API接口

    使用multipart/form-data格式上传：
    - **image_or_video**: 上传的图片或视频文件
    - **scene_id**: 场景ID
        - 0: 路面抛洒物
        - 1: 路面裂缝
        - 2: 路面坑洼
        - 3: 路面积水
        - 4: 护栏破损
        - 5: 边坡破损
        - 6: 排水沟积水
        - 7: 排水沟破损
    """
    request_start_time = time.time()
    try:
        # 验证scene_id
        if scene_id not in SCENE_MAPPING:
            raise HTTPException(status_code=400, detail=f"无效的场景ID: {scene_id}")

        # 保存上传的文件到临时目录
        file_ext = os.path.splitext(image_or_video.filename)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)

        try:
            # 读取上传的文件内容并写入临时文件
            content = await image_or_video.read()
            temp_file.write(content)
            temp_file.close()

            # 获取处理器并处理
            processor = get_processor(scene_id)
            result = processor.process(temp_file.name, scene_id)

            # 打印推理耗时
            elapsed_time = time.time() - request_start_time
            print(f"[推理耗时] scene_id={scene_id}, 耗时={elapsed_time:.2f}秒")

            return {
                "success": True,
                "message": "",
                "scene_id": scene_id,
                "data": result,
            }

        finally:
            # 清理临时文件
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        # 打印推理耗时（失败情况）
        elapsed_time = time.time() - request_start_time
        print(f"[推理耗时] scene_id={scene_id}, 耗时={elapsed_time:.2f}秒 (失败)")
        return {"success": False, "message": str(e), "scene_id": scene_id, "data": {}}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
