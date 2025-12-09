import requests

try:
    url = 'http://localhost:8000/docs'
    print(f'正在访问: {url}')
    response = requests.get(url, timeout=15)
    
    if response.status_code == 200:
        print('✅ API文档访问成功！')
        print('响应状态码:', response.status_code)
        print('文档页面长度:', len(response.text), '字符')
        print('\n在浏览器中访问以下地址:')
        print('  http://localhost:8000/docs')
        print('  或')
        print('  http://127.0.0.1:8000/docs')
    else:
        print('❌ API文档访问失败')
        print('响应状态码:', response.status_code)
        
except requests.exceptions.ConnectionError:
    print('❌ 无法连接到API服务')
    print('请确保服务正在运行: python start_api.py')
except requests.exceptions.Timeout:
    print('❌ 请求超时')
    print('服务可能正在启动中，请稍后再试')
except Exception as e:
    print('❌ 发生错误:', e)
