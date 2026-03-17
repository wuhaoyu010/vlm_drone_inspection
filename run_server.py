"""
启动API服务
Windows兼容版本

启动时会自动预热：
- VLM 客户端（大模型连接）
- RAG 知识库
- 所有处理器（Task1-Task4）
- YOLO 模型（Task2 用）
"""

import os
import sys
from pathlib import Path

# 确保项目根目录在路径中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_config():
    """检查配置"""
    from config import VLM_API_KEY, VLM_MODEL, VLM_PROVIDER, VLM_BASE_URL, HOST, PORT

    print("=" * 60)
    print("高速公路病害检测API服务")
    print("=" * 60)

    if not VLM_API_KEY:
        print("\n错误: 未配置 VLM API Key")
        print("-" * 60)
        print("请编辑 config.yaml 文件，设置 vlm.api_key")
        print("\n配置示例:")
        print("""
vlm:
  provider: "openai"          # 服务商
  api_key: "sk-xxx"           # API密钥
  base_url: "https://api.openai.com/v1"  # API地址
  model: "gpt-4o"             # 模型名称
""")
        print("\n或设置环境变量:")
        print("  Windows: set VLM_API_KEY=sk-xxx")
        print("  Linux/Mac: export VLM_API_KEY=sk-xxx")
        return False

    print(f"\n配置信息:")
    print(f"  服务商: {VLM_PROVIDER}")
    print(f"  模型: {VLM_MODEL}")
    print(f"  API地址: {VLM_BASE_URL}")
    print(f"  服务地址: http://{HOST}:{PORT}")
    print(f"  API文档: http://{HOST}:{PORT}/docs")
    print("=" * 60)

    return True


def main():
    """主函数"""
    if not check_config():
        sys.exit(1)

    print("\n提示: 服务启动时会预热所有组件（VLM、RAG、YOLO等）")
    print("      首次启动可能需要 30-60 秒，请耐心等待...\n")

    import uvicorn
    from api import app
    from config import HOST, PORT

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
