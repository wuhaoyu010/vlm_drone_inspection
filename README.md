# 高速公路病害检测系统

基于VLM（视觉语言模型）的高速公路病害检测与诊断系统，用于2026道通Physical AI人工智能大赛。

## 特性

- ✅ 支持多种VLM服务商（OpenAI、阿里云、智谱、DeepSeek等）
- ✅ OpenAI兼容接口，易于切换模型
- ✅ YAML配置文件，配置更灵活
- ✅ Windows兼容，跨平台运行
- ✅ 完整的4个检测任务支持
- ✅ **切片推理** - 大图分块检测，提升小目标识别率
- ✅ **RAG知识库增强** - L2/L3输出专业化
- ✅ **并行测试** - 多VLM实例并发处理

## 项目结构

```
├── api.py                 # FastAPI服务主程序
├── run_server.py          # 启动API服务
├── test_runner.py         # 单进程测试脚本
├── test_runner_parallel.py # 并行测试脚本（多VLM实例）
├── config.yaml            # 配置文件（YAML格式）
├── config.py              # 配置管理
├── vlm_client.py          # VLM客户端
├── prompts.py             # 提示词模板
├── rag_knowledge.py       # RAG知识库模块
├── processors/            # 任务处理器
│   ├── __init__.py
│   ├── base.py           # 基类
│   ├── task1.py          # 抛洒物检测（切片推理）
│   ├── task2.py          # 违停检测
│   ├── task3.py          # 路面护栏病害（切片推理）
│   └── task4.py          # 路外病害（切片推理）
├── utils/                 # 工具模块
│   └── visualization.py  # 可视化标注
├── knowledge_base/        # RAG知识库文档
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

# 切片推理配置（可选，提升小目标检测）
task1:
  use_tiled_inference: true
  tile_size: 640
  tile_overlap: 0.2
  merge_iou_threshold: 0.3
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

#### 使用PowerShell（推荐）

```powershell
# 激活conda环境
conda activate llm

# 运行指定Task测试
python test_runner.py -t Task_1      # 仅测试Task_1
python test_runner.py -t Task_2      # 仅测试Task_2
python test_runner.py -t Task_3      # 仅测试Task_3
python test_runner.py -t Task_4      # 仅测试Task_4

# 限制处理文件数量
python test_runner.py -t Task_1 -l 5  # 只处理前5个文件

# 运行全部Task测试
python test_runner.py

# 并行测试（多VLM实例，更快）
python test_runner_parallel.py
```

#### 命令行参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `-t, --task` | 指定测试的Task | `-t Task_4` |
| `-l, --limit` | 限制处理文件数 | `-l 10` |
| `-i, --input` | 输入目录 | `-i 竞赛图片视频材料` |
| `-o, --output` | 输出目录 | `-o output` |

#### 输出结果

测试结果保存在 `output/` 目录：
```
output/
├── Task_1/
│   ├── results/      # JSON检测结果
│   └── annotated/    # 标注后的图片
├── Task_2/
│   ├── results/
│   └── annotated/
├── Task_3/
│   ├── results/
│   └── annotated/
├── Task_4/
│   ├── results/
│   └── annotated/
└── test_stats.json   # 测试统计信息
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
- `scene_id`: 场景ID (0-9)

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
        "l2_result": "经检测，影像中存在1处抛洒物隐患（对应第二章隐患清单中「抛洒物」类别），分别为白色纸箱。根据外观与体积特征判断：纸箱体积较大，材质为硬质纸板。抛洒物位于道路车行区域，可能迫使后方车辆紧急避让；本任务未提供历史巡检对比信息，故本次仅基于单帧材质特性（硬）、尺寸观感及所在车道位置进行风险研判。",
        "l3_result": "隐患严重程度：P2。风险说明：抛洒物出现在车行区域，可能对高速车辆形成直接撞击与失控风险。处理建议：及时清理路面抛洒物，优先对行车道/超车道内抛洒物进行快速清障。"
    }
}
```

### 场景ID说明

| scene_id | 场景 | 任务 | 特殊说明 |
|----------|------|------|----------|
| 0 | 路面抛洒物 | Task 1 | 排除安全警示物品 |
| 1 | 车辆违停 | Task 2 | 视频处理 |
| 2 | 路面裂缝 | Task 3 | 切片推理 |
| 3 | 路面坑洼 | Task 3 | 切片推理 |
| 4 | 路面积水 | Task 3 | 切片推理 |
| 5 | 护栏破损 | Task 3 | - |
| 6 | 边坡滑坡 | Task 4 | - |
| 7 | 排水沟积水 | Task 4 | 近义词支持 |
| 8 | 排水沟破损 | Task 4 | 近义词支持 |
| 9 | 标志牌异常 | Task 4 | 新增 |

## 核心功能

### 1. 切片推理（Tiled Inference）

针对无人机高拍场景，大图像中的小目标（如远处抛洒物）检测优化：

```yaml
task1:
  use_tiled_inference: true
  tile_size: 640      # 切片尺寸
  tile_overlap: 0.2   # 重叠比例
  merge_iou_threshold: 0.3  # 合并阈值
```

**原理**：将大图切成多个小片，分别检测后合并结果，提升小目标识别率。

#### Task4 场景级切片推理配置

Task4 包含多种场景，目标尺寸差异大，支持按场景单独配置：

```yaml
task4:
  use_tiled_inference: true        # 全局默认
  scene_tiled_inference:           # 按场景覆盖（优先级更高）
    6: false   # 边坡异常 - 大目标，切片会破坏完整性
    7: true    # 排水沟积水
    8: true    # 排水沟破损
    9: true    # 标志牌异常 - 小目标，推荐切片
```

| scene_id | 场景 | 建议配置 | 原因 |
|----------|------|----------|------|
| 6 | 边坡异常 | `false` | 边坡是大面积目标，切片会破坏完整性 |
| 7 | 排水沟积水 | `true` | 积水区域可能较远，切片有助识别 |
| 8 | 排水沟破损 | `true` | 破损区域可能较小 |
| 9 | 标志牌异常 | `true` | 远处标志牌是小目标，强烈推荐切片 |

### 2. RAG知识库增强

L2/L3输出基于专业知识库增强：
- 病害特征描述专业化
- 风险评估有理有据
- 处置建议符合规范

### 3. 排除规则

**Task1 排除规则**（不检测以下物品为抛洒物）：
- 路面标线（白色/黄色车道线、导向箭头等）
- 安全警示物品：
  - 三角架警示牌
  - 施工警示条纹柱
  - 条纹圆锥形柱子（交通锥、警示桶）

**Task3 排除规则**（不检测以下情况为病害）：
- 路面引导线/标线
- 已修补的裂缝
- 正常路面接缝
- 阴影、轮胎痕迹

### 4. 排水沟近义词支持

系统识别以下词汇均指向"排水沟"：
- 边沟、路侧沟、排水渠、明沟、水沟、路缘沟

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

**Q: 切片推理太慢怎么办？**
A: 可以调大 `tile_size` 或设置 `use_tiled_inference: false` 关闭。

**Q: 如何配置多VLM并行测试？**
A: 编辑 `config.yaml` 中的 `vlms` 列表，添加多个VLM服务地址。

**Q: Task_4 边坡异常图片为什么 l1_result 为空？**
A: 这通常是VLM模型识别偏差导致。系统会根据VLM输出的`category`自动推断`scene_id`，并与目标场景比对过滤：
- 如果VLM把边坡识别成排水沟积水，推断的scene_id=7会被正确过滤掉
- 最终结果为空表示该图片确实没有被识别为边坡滑坡
- 解决方案：尝试使用更强的模型，或在prompt中强调目标类型

## License

MIT