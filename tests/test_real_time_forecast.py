import requests
import json

def test_real_time_forecast():
    """
    测试实时预测端点是否正常工作
    """
    # API端点URL
    url = "http://localhost:8000/api/forecast/real-time"
    
    # 请求参数
    params = {
        "product_id": 1,
        "forecast_days": 7,
        "model_tag": "latest"
    }
    
    # 发送POST请求
    try:
        response = requests.post(url, params=params)
        
        print(f"请求URL: {url}")
        print(f"请求参数: {json.dumps(params, indent=2)}")
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            # 打印响应结果
            data = response.json()
            print("\n响应成功!")
            print(f"预测产品ID: {data.get('product_id')}")
            print(f"使用模型: {data.get('model_used')}")
            print(f"预测天数: {len(data.get('forecast', []))}")
            print(f"\n前5天预测结果:")
            for i, day in enumerate(data.get('forecast', [])[:5]):
                print(f"第{i+1}天 - 日期: {day.get('date')}, 预测值: {day.get('predicted_value')}")
        else:
            # 打印错误信息
            print(f"\n响应失败!")
            try:
                error_data = response.json()
                print(f"错误信息: {json.dumps(error_data, indent=2)}")
            except:
                print(f"错误内容: {response.text}")
                
    except Exception as e:
        print(f"请求过程中发生错误: {e}")

if __name__ == "__main__":
    test_real_time_forecast()