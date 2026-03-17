#!/bin/bash
# 清理残留的 test_runner_parallel.py 进程

echo "=== 清理残留的 test_runner_parallel.py 进程 ==="

# 显示当前运行的进程
echo "当前运行的进程:"
ps -ef | grep "test_runner_parallel.py" | grep -v grep

# 杀掉所有 test_runner_parallel.py 进程
echo ""
echo "正在终止所有 test_runner_parallel.py 进程..."
pkill -9 -f "test_runner_parallel.py"

# 等待一秒
sleep 1

# 检查是否还有残留
remaining=$(ps -ef | grep "test_runner_parallel.py" | grep -v grep | wc -l)

if [ "$remaining" -eq 0 ]; then
    echo "✓ 所有 test_runner_parallel.py 进程已清理完毕"
else
    echo "⚠ 还有 $remaining 个残留进程:"
    ps -ef | grep "test_runner_parallel.py" | grep -v grep
    echo ""
    echo "尝试使用 kill -9 强制终止..."
    ps -ef | grep "test_runner_parallel.py" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
fi

echo ""
echo "=== 完成 ==="