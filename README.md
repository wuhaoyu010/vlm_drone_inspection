# 高速公路病害检测系统

基于VLM（视觉语言模型）的高速公路病害检测与诊断系统，用于2026道通Physical AI人工智能大赛。

## 特性

- ✅ 支持多种VLM服务商（OpenAI、阿里云、智谱、DeepSeek等）
- ✅ OpenAI兼容接口，易于切换模型
- ✅ YAML配置文件，配置更灵活
- ✅ Windows兼容，跨平台运行
- ✅ 完整的4个检测任务支持

## 项目结构

```
├── api.py                 # FastAPI服务主程序
├── run_server.py          # 启动API服务
├── quick_test.py          # 快速测试脚本
├── config.yaml            # 配置文件（YAML格式）
├── config.py              # 配置管理
├── vlm_client.py          # VLM客户端
├── prompts.py             # 提示词模板
├── processors/            # 任务处理器
│   ├── __init__.py
│   ├── base.py           # 基类
│   ├── task1.py          # 抛洒物检测
│   ├── task2.py          # 违停检测
│   ├── task3.py          # 路面护栏病害
│   └── task4.py          # 路外病害
├── test_all.py           # 自动化测试脚本
├── requirements.txt      # Python依赖
├── Dockerfile            # Docker配置
└── README.md             # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型

编辑 `config.yaml` 文件：

```yaml
vlm:
  # 模型服务商: openai, dashscope, zhipu, deepseek, custom
  provider: "openai"
  
  # API配置
  api_key: "your-api-key-here"
  base_url: "https://api.openai.com/v1"
  
  # 模型名称
  model: "gpt-4o"
```

#### 不同服务商配置示例

**OpenAI:**
```yaml
provider: "openai"
api_key: "sk-xxx"
base_url: "https://api.openai.com/v1"
model: "gpt-4o"
```

**阿里云DashScope:**
```yaml
provider: "dashscope"
api_key: "sk-xxx"
base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
model: "qwen-vl-max"
```

**智谱AI:**
```yaml
provider: "zhipu"
api_key: "xxx.xxx"
base_url: "https://open.bigmodel.cn/api/paas/v4"
model: "glm-4v"
```

**DeepSeek:**
```yaml
provider: "deepseek"
api_key: "sk-xxx"
base_url: "https://api.deepseek.com/v1"
model: "deepseek-vl"
```

**自定义服务:**
```yaml
provider: "custom"
api_key: "your-key"
base_url: "http://your-server:8000/v1"
model: "your-model"
```

### 3. 运行测试

```bash
python quick_test.py
```

### 4. 启动服务

```bash
python run_server.py
```

服务启动后访问:
- API地址: http://localhost:8000
- API文档: http://localhost:8000/docs

## API接口

### 检测接口

**POST /v1/detect**

请求参数 (multipart/form-data):
- `image_or_video`: 上传的图片或视频文件
- `scene_id`: 场景ID (0-8)

**响应示例:**

```json
{
    "success": true,
    "message": "",
    "scene_id": 0,
    "data": {
        "l1_result": [
            {
                "bbox": [100, 200, 300, 400],
                "category": "纸箱",
                "scene_id": 0
            }
        ],
        "l2_result": "分析过程...",
        "l3_result": "风险等级：P0。处理建议：..."
    }
}
```

### 场景ID说明

| scene_id | 场景 | 任务 |
|----------|------|------|
| 0 | 路面抛洒物 | Task 1 |
| 1 | 车辆违停 | Task 2 |
| 2 | 路面裂缝 | Task 3 |
| 3 | 路面坑洼 | Task 3 |
| 4 | 路面积水 | Task 3 |
| 5 | 护栏破损 | Task 3 |
| 6 | 边坡滑坡 | Task 4 |
| 7 | 排水沟积水 | Task 4 |
| 8 | 排水沟破损 | Task 4 |

## Docker部署

```bash
# 构建镜像
docker build -t highway-detection:latest .

# 运行容器
docker run -d \
    --name highway-detection \
    -p 8000:8000 \
    -v $(pwd)/config.yaml:/app/config.yaml \
    highway-detection:latest
```

## 比赛要求

- **L1评分 (60%)**: F1分数评估检测精度
- **L2评分 (20%)**: 推理过程合理性
- **L3评分 (20%)**: 风险等级与处置建议
- **准入门槛**: L1 F1 ≥ 0.7

## Windows兼容性

本项目已针对Windows进行优化：
- 使用 `pathlib.Path` 处理路径
- 视频编码使用兼容的 `mp4v` 格式
- 临时文件使用正确的编码方式

## 常见问题

**Q: 如何切换不同的模型？**
A: 编辑 `config.yaml` 文件中的 `vlm` 配置项即可。

**Q: 视频处理失败怎么办？**
A: 确保安装了 opencv-python：`pip install opencv-python`

**Q: 如何使用环境变量配置？**
A: 设置环境变量 `VLM_API_KEY`、`VLM_BASE_URL`、`VLM_MODEL` 会覆盖配置文件。

## License

MIT
