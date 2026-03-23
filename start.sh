#!/bin/bash
# 高速公路检测系统启动脚本
# 1. 启动 vLLM 服务
# 2. 等待 vLLM 就绪
# 3. 启动 FastAPI 服务
#
# 用法:
#   ./start.sh                           # 使用默认配置
#   ./start.sh --vllm-port 8001 --api-port 9000  # 自定义端口
#   ./start.sh --model /path/to/model    # 自定义模型路径

set -e

# ==================== 默认配置 ====================
VLLM_HOST="0.0.0.0"
VLLM_PORT=8001
VLLM_MODEL="/data"
VLLM_GPU_MEMORY=0.92
VLLM_MAX_LEN=6000
VLLM_MODEL_NAME="Qwen3_VL_8B"

API_HOST="0.0.0.0"
API_PORT=8000

# 最大等待时间（秒）
MAX_WAIT_TIME=600
# 检查间隔（秒）
CHECK_INTERVAL=5

# ==================== 解析命令行参数 ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --vllm-host)
            VLLM_HOST="$2"
            shift 2
            ;;
        --vllm-port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --model)
            VLLM_MODEL="$2"
            shift 2
            ;;
        --gpu-memory)
            VLLM_GPU_MEMORY="$2"
            shift 2
            ;;
        --api-host)
            API_HOST="$2"
            shift 2
            ;;
        --api-port)
            API_PORT="$2"
            shift 2
            ;;
        --model-name)
            VLLM_MODEL_NAME="$2"
            shift 2
            ;;
        --max-wait)
            MAX_WAIT_TIME="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --vllm-host HOST      vLLM服务地址 (默认: 0.0.0.0)"
            echo "  --vllm-port PORT      vLLM服务端口 (默认: 8000)"
            echo "  --model PATH          模型路径 (默认: /data)"
            echo "  --gpu-memory RATIO    GPU内存使用比例 (默认: 0.6)"
            echo "  --api-host HOST       API服务地址 (默认: 0.0.0.0)"
            echo "  --api-port PORT       API服务端口 (默认: 8080)"
            echo "  --model-name NAME     模型名称 (默认: Qwen3_VL_8B)"
            echo "  --max-wait SECONDS    最大等待时间 (默认: 600)"
            echo "  -h, --help            显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== 函数 ====================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 vLLM 是否就绪
check_vllm_ready() {
    local url="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"

    # 使用 curl 测试 OpenAI 兼容接口
    response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$url" 2>/dev/null)

    if [ "$response" = "200" ]; then
        return 0
    fi
    return 1
}

# 测试 vLLM 生成能力
test_vllm_generation() {
    local url="http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions"

    local response
    response=$(curl -s --connect-timeout 10 --max-time 30 "$url" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$VLLM_MODEL_NAME"'",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }' 2>/dev/null)

    if echo "$response" | grep -q "choices"; then
        return 0
    fi
    return 1
}

# 清理函数
cleanup() {
    log_warning "收到终止信号，正在清理..."

    # 终止 vLLM 相关进程
    log_info "停止 vLLM 服务..."
    pkill -f "vllm serve" 2>/dev/null || true
    if [ ! -z "$VLLM_PID" ]; then
        kill $VLLM_PID 2>/dev/null || true
    fi

    # 终止 API 服务
    if [ ! -z "$API_PID" ]; then
        log_info "停止 API 服务 (PID: $API_PID)..."
        kill $API_PID 2>/dev/null || true
        wait $API_PID 2>/dev/null || true
    fi

    log_info "清理完成"
    exit 0
}

# 捕获终止信号
trap cleanup SIGINT SIGTERM

# ==================== 主流程 ====================

echo ""
echo "=========================================="
echo "   高速公路病害检测系统启动脚本"
echo "=========================================="
echo ""

# 确保日志目录存在
mkdir -p logs

# 步骤1: 启动 vLLM
log_info "步骤1: 启动 vLLM 服务..."
log_info "  - 模型: $VLLM_MODEL"
log_info "  - 地址: http://${VLLM_HOST}:${VLLM_PORT}"
log_info "  - GPU内存: ${VLLM_GPU_MEMORY}"
log_info "  - 日志将实时显示..."
echo ""

# 启动 vLLM 并实时显示日志（优化批处理参数）
vllm serve \
    --gpu-memory-utilization $VLLM_GPU_MEMORY \
    --host $VLLM_HOST \
    --port $VLLM_PORT \
    --max-model-len $VLLM_MAX_LEN \
    --trust-remote-code \
    --model $VLLM_MODEL \
    --seed 0 \
    --tensor_parallel_size 1 \
    --dtype auto \
    --mm-processor-cache-gb 0 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --served-model-name $VLLM_MODEL_NAME \
    2>&1 | tee logs/vllm.log &

VLLM_PID=$!

# 等待日志文件创建
sleep 2
log_info "vLLM 已启动 (PID: $VLLM_PID)"

# 步骤2: 等待 vLLM 就绪
log_info "步骤2: 等待 vLLM 服务就绪..."

waited=0
while [ $waited -lt $MAX_WAIT_TIME ]; do
    if check_vllm_ready; then
        log_success "vLLM 服务已响应 (等待 ${waited}s)"

        # 进一步测试生成能力
        log_info "测试 vLLM 生成能力..."
        if test_vllm_generation; then
            log_success "vLLM 生成测试通过"
            break
        else
            log_warning "生成测试失败，继续等待..."
        fi
    fi

    sleep $CHECK_INTERVAL
    waited=$((waited + CHECK_INTERVAL))

    # 显示进度
    if [ $((waited % 30)) -eq 0 ]; then
        log_info "已等待 ${waited}s，继续等待 vLLM 启动..."
    fi
done

if [ $waited -ge $MAX_WAIT_TIME ]; then
    log_error "vLLM 启动超时 (${MAX_WAIT_TIME}s)"
    log_info "查看日志: tail -f logs/vllm.log"
    cleanup
    exit 1
fi

log_success "vLLM 服务已完全就绪!"

# 步骤3: 启动 FastAPI 服务
log_info "步骤3: 启动 FastAPI 服务..."
log_info "  - 地址: http://${API_HOST}:${API_PORT}"

# 启动 API 服务
cd "$(dirname "$0")"

# 使用 uvicorn 启动（实时显示日志）
python3 -m uvicorn api:app --host $API_HOST --port $API_PORT 2>&1 | tee logs/api.log &

API_PID=$!
log_info "API 服务已启动 (PID: $API_PID)"

# 等待 API 服务启动
sleep 10

# 检查 API 是否正常
if curl -s "http://${API_HOST}:${API_PORT}/docs" > /dev/null 2>&1; then
    log_success "API 服务已就绪"
else
    log_warning "API 服务可能未完全启动，请检查日志"
fi

# 显示服务信息
echo ""
echo "=========================================="
echo "   服务启动完成!"
echo "=========================================="
echo ""
echo "  vLLM 服务:"
echo "    - 地址: http://${VLLM_HOST}:${VLLM_PORT}"
echo "    - 日志: logs/vllm.log"
echo "    - PID:  $VLLM_PID"
echo ""
echo "  API 服务:"
echo "    - 地址: http://${API_HOST}:${API_PORT}"
echo "    - 文档: http://${API_HOST}:${API_PORT}/docs"
echo "    - 日志: logs/api.log"
echo "    - PID:  $API_PID"
echo ""
echo "  按 Ctrl+C 停止所有服务"
echo "=========================================="
echo ""

# 保持脚本运行，等待终止信号
wait