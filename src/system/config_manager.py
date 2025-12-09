import os
import threading
from dotenv import load_dotenv

class ConfigManager:
    """
    配置管理类，用于集中管理所有配置
    支持从环境变量和.env文件加载配置
    实现了线程安全的单例模式
    """
    
    _instance = None
    _lock = threading.Lock()  # 用于确保单例创建的线程安全
    
    def __new__(cls):
        # 双检查锁定模式，提高性能
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._init_lock = threading.Lock()  # 用于确保初始化的线程安全
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 使用锁确保初始化过程的线程安全
        with self._init_lock:
            if self._initialized:
                return
                
            # 加载.env文件
            load_dotenv()
            
            # 初始化配置
            self._load_configs()
            
            self._initialized = True
    
    def _load_configs(self):
        """
        加载所有配置
        """
        # 数据源配置
        self._data_source_type = os.getenv('DATA_SOURCE_TYPE', 'csv')
        self._data_dir = os.getenv('DATA_DIR', 'data')
        self._database_connection_string = os.getenv('DATABASE_CONNECTION_STRING', '')
        self._api_base_url = os.getenv('API_BASE_URL', '')
        self._api_token = os.getenv('API_TOKEN', '')
        
        # 缓存配置
        self._cache_type = os.getenv('CACHE_TYPE', 'memory')
        self._redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self._cache_expire_default = int(os.getenv('CACHE_EXPIRE_DEFAULT', '3600'))  # 默认1小时
        
        # 并行计算配置
        self._parallel_mode = os.getenv('PARALLEL_MODE', 'process')  # single, thread, process
        self._max_workers = int(os.getenv('MAX_WORKERS', '4'))
        
        # GPU加速配置
        self._gpu_mode = os.getenv('GPU_MODE', 'auto')  # auto, cpu, gpu
        
        # 日志配置
        self._log_level = os.getenv('LOG_LEVEL', 'INFO')  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        self._log_file = os.getenv('LOG_FILE', 'logs/general.log')
        self._error_log_file = os.getenv('ERROR_LOG_FILE', 'logs/error.log')
        
        # 预测模型配置
        self._forecast_horizon = int(os.getenv('FORECAST_HORIZON', '7'))  # 预测周期（天）
        self._seasonality_period = int(os.getenv('SEASONALITY_PERIOD', '12'))  # 季节性周期
        self._confidence_interval = float(os.getenv('CONFIDENCE_INTERVAL', '0.95'))  # 置信区间
        
        # 库存优化配置
        self._default_lead_time = int(os.getenv('DEFAULT_LEAD_TIME', '7'))  # 默认提前期（天）
        self._default_service_level = float(os.getenv('DEFAULT_SERVICE_LEVEL', '0.95'))  # 默认服务水平
        
        # API配置
        self._api_host = os.getenv('API_HOST', '0.0.0.0')
        self._api_port = int(os.getenv('API_PORT', '8000'))
        self._api_workers = int(os.getenv('API_WORKERS', '4'))
    
    # 数据源配置
    @property
    def data_source_type(self):
        return self._data_source_type
    
    @property
    def data_dir(self):
        return self._data_dir
    
    @property
    def database_connection_string(self):
        return self._database_connection_string
    
    @property
    def api_base_url(self):
        return self._api_base_url
    
    @property
    def api_token(self):
        return self._api_token
    
    # 缓存配置
    @property
    def cache_type(self):
        return self._cache_type
    
    @property
    def redis_url(self):
        return self._redis_url
    
    @property
    def cache_expire_default(self):
        return self._cache_expire_default
    
    # 并行计算配置
    @property
    def parallel_mode(self):
        return self._parallel_mode
    
    @property
    def max_workers(self):
        return self._max_workers
    
    # GPU加速配置
    @property
    def gpu_mode(self):
        return self._gpu_mode
    
    # 日志配置
    @property
    def log_level(self):
        return self._log_level
    
    @property
    def log_file(self):
        return self._log_file
    
    @property
    def error_log_file(self):
        return self._error_log_file
    
    # 预测模型配置
    @property
    def forecast_horizon(self):
        return self._forecast_horizon
    
    @property
    def seasonality_period(self):
        return self._seasonality_period
    
    @property
    def confidence_interval(self):
        return self._confidence_interval
    
    # 库存优化配置
    @property
    def default_lead_time(self):
        return self._default_lead_time
    
    @property
    def default_service_level(self):
        return self._default_service_level
    
    # API配置
    @property
    def api_host(self):
        return self._api_host
    
    @property
    def api_port(self):
        return self._api_port
    
    @property
    def api_workers(self):
        return self._api_workers
    
    def get(self, key, default=None):
        """
        动态获取配置值
        
        Args:
            key: 配置键名
            default: 默认值
            
        Returns:
            配置值
        """
        return getattr(self, f"_{key}", default)
    
    def reload(self):
        """
        重新加载配置
        """
        self._load_configs()

# 创建全局配置管理器实例
config_manager = ConfigManager()
