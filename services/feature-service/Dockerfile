FROM python:3.10-slim

WORKDIR /app

# 复制requirements.txt
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .

# 复制数据目录（如果有）
RUN mkdir -p /app/data /app/features

# 暴露端口
EXPOSE 8002

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
