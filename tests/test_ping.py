import requests

try:
    r = requests.get('http://localhost:8000/', timeout=5)
    print('根路径响应: ' + str(r.status_code) + ' ' + r.text[:100])
except Exception as e:
    print('错误: ' + str(e))
