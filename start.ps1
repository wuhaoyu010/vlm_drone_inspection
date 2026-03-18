# 高速公路检测系统启动脚本 (Windows PowerShell)
# 1. 启动 vLLM 服务
# 2. 等待 vLLM 就绪
# 3. 启动 FastAPI 服务
#
# 用法:
#   .\start.ps1                           # 使用默认配置
#   .\start.ps1 -VllmPort 8001 -ApiPort 9000  # 自定义端口
#   .\start.ps1 -VllmModel "D:\models\qwen"   # 自定义模型路径

param(
    [string]$VllmHost = "0.0.0.0",
    [int]$VllmPort = 8000,
    [string]$VllmModel = "/data",
    [float]$GpuMemory = 0.6,
    [int]$MaxWaitTime = 600,
    [string]$ApiHost = "0.0.0.0",
    [int]$ApiPort = 8080,
    [string]$ModelName = "Qwen3_VL_8B"
)

# 错误时停止
$ErrorActionPreference = "Stop"

# 日志函数
function Write-Log-Info { param($msg) Write-Host "[INFO] $msg" -ForegroundColor Blue }
function Write-Log-Success { param($msg) Write-Host "[SUCCESS] $msg" -ForegroundColor Green }
function Write-Log-Warning { param($msg) Write-Host "[WARNING] $msg" -ForegroundColor Yellow }
function Write-Log-Error { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

# 检查 vLLM 是否就绪
function Test-VllmReady {
    param($Host_, $Port)

    try {
        $url = "http://${Host_}:${Port}/v1/models"
        $response = Invoke-WebRequest -Uri $url -Method Get -TimeoutSec 5 -UseBasicParsing
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

# 测试 vLLM 生成能力
function Test-VllmGeneration {
    param($Host_, $Port, $ModelName)

    try {
        $url = "http://${Host_}:${Port}/v1/chat/completions"
        $body = @{
            model = $ModelName
            messages = @(@{role="user"; content="Hello"})
            max_tokens = 10
        } | ConvertTo-Json -Depth 3

        $response = Invoke-WebRequest -Uri $url -Method Post -Body $body -ContentType "application/json" -TimeoutSec 30 -UseBasicParsing
        return $response.Content -match "choices"
    }
    catch {
        return $false
    }
}

# 创建日志目录
$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

# 主流程
Write-Host ""
Write-Host "=========================================="
Write-Host "   高速公路病害检测系统启动脚本"
Write-Host "=========================================="
Write-Host ""

# 步骤1: 启动 vLLM
Write-Log-Info "步骤1: 启动 vLLM 服务..."
Write-Log-Info "  - 模型: $VllmModel"
Write-Log-Info "  - 地址: http://${VllmHost}:${VllmPort}"
Write-Log-Info "  - GPU内存: $GpuMemory"

$vllmLog = Join-Path $logDir "vllm.log"
$vllmErrLog = Join-Path $logDir "vllm_error.log"

$vllmArgs = @(
    "serve",
    "--gpu-memory-utilization", $GpuMemory,
    "--host", $VllmHost,
    "--port", $VllmPort,
    "--max-model-len", "6000",
    "--trust-remote-code",
    "--model", $VllmModel,
    "--seed", "0",
    "--tensor_parallel_size", "1",
    "--dtype", "auto",
    "--mm-processor-cache-gb", "0",
    '--limit-mm-per-prompt', '{"image":1,"video":0}',
    "--served-model-name", $ModelName
)

try {
    $vllmProcess = Start-Process -FilePath "vllm" -ArgumentList $vllmArgs -RedirectStandardOutput $vllmLog -RedirectStandardError $vllmErrLog -PassThru -NoNewWindow
    Write-Log-Info "vLLM 已启动 (PID: $($vllmProcess.Id))"
}
catch {
    Write-Log-Error "启动 vLLM 失败: $_"
    Write-Log-Info "请确保 vllm 已安装并在 PATH 中"
    exit 1
}

# 步骤2: 等待 vLLM 就绪
Write-Log-Info "步骤2: 等待 vLLM 服务就绪..."

$waited = 0
$checkInterval = 5

while ($waited -lt $MaxWaitTime) {
    if (Test-VllmReady -Host_ $VllmHost -Port $VllmPort) {
        Write-Log-Success "vLLM 服务已响应 (等待 ${waited}s)"

        Write-Log-Info "测试 vLLM 生成能力..."
        if (Test-VllmGeneration -Host_ $VllmHost -Port $VllmPort -ModelName $ModelName) {
            Write-Log-Success "vLLM 生成测试通过"
            break
        }
        else {
            Write-Log-Warning "生成测试失败，继续等待..."
        }
    }

    Start-Sleep -Seconds $checkInterval
    $waited += $checkInterval

    if ($waited % 30 -eq 0) {
        Write-Log-Info "已等待 ${waited}s，继续等待 vLLM 启动..."
    }
}

if ($waited -ge $MaxWaitTime) {
    Write-Log-Error "vLLM 启动超时 (${MaxWaitTime}s)"
    Write-Log-Info "查看日志: Get-Content $vllmLog"
    Stop-Process -Id $vllmProcess.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Log-Success "vLLM 服务已完全就绪!"

# 提醒更新配置
Write-Log-Warning "请确保 config.yaml 中的 vlm.base_url 已更新为: http://${VllmHost}:${VllmPort}/v1"

# 步骤3: 启动 FastAPI 服务
Write-Log-Info "步骤3: 启动 FastAPI 服务..."
Write-Log-Info "  - 地址: http://${ApiHost}:${ApiPort}"

$apiLog = Join-Path $logDir "api.log"
$apiErrLog = Join-Path $logDir "api_error.log"
Set-Location $PSScriptRoot

try {
    $apiProcess = Start-Process -FilePath "python" -ArgumentList @("-m", "uvicorn", "api:app", "--host", $ApiHost, "--port", $ApiPort) -RedirectStandardOutput $apiLog -RedirectStandardError $apiErrLog -PassThru -NoNewWindow
    Write-Log-Info "API 服务已启动 (PID: $($apiProcess.Id))"
}
catch {
    Write-Log-Error "启动 API 服务失败: $_"
    Stop-Process -Id $vllmProcess.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

Start-Sleep -Seconds 3

# 显示服务信息
Write-Host ""
Write-Host "=========================================="
Write-Host "   服务启动完成!"
Write-Host "=========================================="
Write-Host ""
Write-Host "  vLLM 服务:"
Write-Host "    - 地址: http://${VllmHost}:${VllmPort}"
Write-Host "    - 日志: $vllmLog"
Write-Host "    - PID:  $($vllmProcess.Id)"
Write-Host ""
Write-Host "  API 服务:"
Write-Host "    - 地址: http://${ApiHost}:${ApiPort}"
Write-Host "    - 文档: http://${ApiHost}:${ApiPort}/docs"
Write-Host "    - 日志: $apiLog"
Write-Host "    - PID:  $($apiProcess.Id)"
Write-Host ""
Write-Host "  配置提醒:"
Write-Host "    - 请确保 config.yaml 中 vlm.base_url = http://${VllmHost}:${VllmPort}/v1"
Write-Host ""
Write-Host "  按 Ctrl+C 停止所有服务"
Write-Host "=========================================="
Write-Host ""

# 等待进程
try {
    Wait-Process -Id $vllmProcess.Id, $apiProcess.Id
}
catch {
    # 用户中断时清理
    Write-Log-Warning "正在停止服务..."
    Stop-Process -Id $vllmProcess.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
    Write-Log-Info "服务已停止"
}