import sys
import os

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath('.'))

# 导入测试函数
from tests.test_forecast_continuous_learning import test_feedback_loop

# 运行测试函数
try:
    test_feedback_loop()
except Exception as e:
    import traceback
    traceback.print_exc()