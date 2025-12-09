import requests

try:
    # 测试根路径
    print("Testing root path...")
    root_response = requests.get('http://localhost:8000/', timeout=10)
    print("Root path success:", root_response.status_code)
    print("Response:", root_response.text[:50])
    
    # 测试API文档路径
    print("\nTesting docs path...")
    docs_response = requests.get('http://localhost:8000/docs', timeout=10)
    print("Docs path success:", docs_response.status_code)
    print("Docs response length:", len(docs_response.text))
    
    print("\nAll tests completed!")
    
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
