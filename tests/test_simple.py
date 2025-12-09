import requests

url = "http://localhost:8000/api/forecast/real-time"
params = {
    "product_id": 1,
    "forecast_days": 7,
    "model_tag": "latest"
}

try:
    # 设置10秒超时
    response = requests.post(url, params=params, timeout=10)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text[:500]}...")
except requests.exceptions.Timeout:
    print("请求超时，请检查服务器是否在运行")
except Exception as e:
    print(f"错误: {e}")
