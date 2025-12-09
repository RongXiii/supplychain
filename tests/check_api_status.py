import requests
import time

print("正在检查API服务状态...")
print("等待服务完全初始化...")
time.sleep(3)  # 等待服务初始化

try:
    # 先尝试使用0.0.0.0地址
    response = requests.get('http://0.0.0.0:8000/', timeout=10)
    print("[OK] API服务正常运行")
    print(f"    响应状态码: {response.status_code}")
    print(f"    响应内容: {response.text[:200]}...")
except requests.exceptions.ConnectionError:
    print("[ERROR] API服务未启动或无法连接")
except requests.exceptions.Timeout:
    print("[ERROR] 请求超时，API服务可能响应缓慢")
except Exception as e:
    print(f"[ERROR] 发生错误: {e}")
    # 再尝试使用localhost地址
    print("\n尝试使用localhost地址...")
    try:
        response = requests.get('http://localhost:8000/', timeout=10)
        print("[OK] 使用localhost连接成功")
        print(f"    响应状态码: {response.status_code}")
    except Exception as e2:
        print(f"[ERROR] 使用localhost也连接失败: {e2}")
