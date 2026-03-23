FROM vllm/vllm-openai:v0.17.1
# 不能访问dockerhub 用镜像源
# FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.17.1

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV HOST=0.0.0.0
ENV PORT=8000

COPY ./start.sh /app/start.sh
# 【关键步骤】赋予 start.sh 执行权限
RUN chmod +x /app/start.sh

# 使用 ENTRYPOINT 启动
# 因为 WORKDIR 已经是 /app，这里直接写脚本名即可
ENTRYPOINT ["bash", "start.sh"]

