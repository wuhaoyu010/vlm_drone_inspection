#!/bin/bash

# 启动API服务
echo "启动高速公路病害检测API服务..."

# 检查环境变量
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "错误: 请设置DASHSCOPE_API_KEY环境变量"
    echo "export DASHSCOPE_API_KEY=your_api_key"
    exit 1
fi

# 启动服务
python api.py
