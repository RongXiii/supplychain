import requests
import time

try:
    start_time = time.time()
    r = requests.get('http://localhost:8000/api/items')
    end_time = time.time()
    print(f'API响应状态码: {r.status_code}')
    print(f'响应时间: {end_time - start_time:.2f}秒')
except Exception as e:
    print(f'连接失败: {e}')