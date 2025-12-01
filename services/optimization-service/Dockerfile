FROM python:3.10-slim

WORKDIR /app

# 复制requirements.txt
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建目录结构
RUN mkdir -p /app/src /app/data

# 复制src目录下的相关文件
COPY ../../src/milp_optimizer.py /app/src/

# 复制应用代码
COPY app.py .

# 暴露端口
EXPOSE 8004

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
