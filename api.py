"""
FastAPI 服务主程序
"""
import os
import tempfile
import uuid
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import HOST, PORT, SCENE_MAPPING
from vlm_client import get_vlm_client
from processors import Task1Processor, Task2Processor, Task3Processor, Task4Processor

# 创建FastAPI应用
app = FastAPI(
    title="高速公路病害检测API",
    description="基于VLM的高速公路病害检测与诊断系统",
    version="1.0.0"
)

# 初始化处理器
processors = {}


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
        raise HTTPException(status_code=400, detail=f"未知的场景ID: {scene_id}")


class DetectionResponse(BaseModel):
    """检测响应模型"""
    success: bool
    message: str
    scene_id: int
    data: dict


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    print("正在初始化VLM客户端...")
    try:
        get_vlm_client()
        print("VLM客户端初始化成功")
    except Exception as e:
        print(f"VLM客户端初始化失败: {e}")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "高速公路病害检测API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/v1/detect", response_model=DetectionResponse)
async def detect(
    image_or_video: UploadFile = File(..., description="上传的图片或视频文件"),
    scene_id: int = Form(..., description="场景ID: 0-抛洒物, 1-违停, 2-裂缝, 3-坑洼, 4-积水, 5-护栏破损, 6-边坡滑坡, 7-排水沟积水, 8-排水沟破损")
):
    """
    检测接口
    
    - **image_or_video**: 上传的图片或视频文件
    - **scene_id**: 场景ID
        - 0: 路面抛洒物
        - 1: 车辆违停
        - 2: 路面裂缝
        - 3: 路面坑洼
        - 4: 路面积水
        - 5: 护栏破损
        - 6: 边坡滑坡
        - 7: 排水沟积水
        - 8: 排水沟破损
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
                success=True,
                message="检测完成",
                scene_id=scene_id,
                data=result
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
    scene_id: int = Form(..., description="场景ID")
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
        
        return DetectionResponse(
            success=True,
            message="检测完成",
            scene_id=scene_id,
            data=result
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


# 兼容比赛要求的接口格式
@app.post("/v1/")
async def detect_compatible(request: Request):
    """
    兼容比赛要求的API接口
    
    请求体格式:
    {
        "image_or_video": "文件路径或base64",
        "scene_id": 0
    }
    """
    try:
        # 尝试解析为JSON
        try:
            body = await request.json()
        except:
            # 如果是multipart/form-data
            form = await request.form()
            body = {
                "image_or_video": form.get("image_or_video"),
                "scene_id": int(form.get("scene_id", 0))
            }
        
        scene_id = body.get("scene_id", 0)
        file_input = body.get("image_or_video", "")
        
        # 验证scene_id
        if scene_id not in SCENE_MAPPING:
            raise HTTPException(status_code=400, detail=f"无效的场景ID: {scene_id}")
        
        # 处理输入
        if os.path.exists(file_input):
            # 本地文件路径
            processor = get_processor(scene_id)
            result = processor.process(file_input, scene_id)
        elif file_input.startswith("data:"):
            # Base64编码
            import base64
            header, data = file_input.split(",", 1)
            # 保存临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file.write(base64.b64decode(data))
            temp_file.close()
            
            processor = get_processor(scene_id)
            result = processor.process(temp_file.name, scene_id)
            os.unlink(temp_file.name)
        else:
            raise HTTPException(status_code=400, detail="无效的输入格式")
        
        return {
            "success": True,
            "message": "",
            "scene_id": scene_id,
            "data": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": str(e),
            "scene_id": body.get("scene_id", 0) if 'body' in dir() else 0,
            "data": {}
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
