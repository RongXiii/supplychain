import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

class LoggingManager:
    """
    日志管理类，负责配置和管理项目的日志系统
    支持不同级别、不同模块的日志记录
    支持文件轮转和定时轮转
    """
    
    def __init__(self, log_dir='logs', level=logging.INFO):
        """
        初始化日志管理器
        
        Args:
            log_dir: 日志文件存储目录
            level: 日志级别，默认为INFO
        """
        self.log_dir = log_dir
        self.level = level
        
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 初始化日志配置
        self._setup_logging()
    
    def _setup_logging(self):
        """
        设置日志配置
        """
        # 创建根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(self.level)
        
        # 移除默认处理器
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        )
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # 创建按大小轮转的通用日志文件处理器
        general_log_path = os.path.join(self.log_dir, 'general.log')
        file_handler = RotatingFileHandler(
            general_log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # 创建按天轮转的错误日志文件处理器
        error_log_path = os.path.join(self.log_dir, 'error.log')
        error_handler = TimedRotatingFileHandler(
            error_log_path,
            when='midnight',
            interval=1,
            backupCount=30
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        # 创建性能日志处理器
        performance_log_path = os.path.join(self.log_dir, 'performance.log')
        performance_handler = TimedRotatingFileHandler(
            performance_log_path,
            when='midnight',
            interval=1,
            backupCount=7
        )
        performance_handler.setLevel(logging.INFO)
        performance_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - PERFORMANCE - %(module)s.%(funcName)s - %(message)s'
        )
        performance_handler.setFormatter(performance_formatter)
        
        # 为性能日志创建单独的记录器
        performance_logger = logging.getLogger('performance')
        performance_logger.setLevel(logging.INFO)
        performance_logger.addHandler(performance_handler)
        performance_logger.propagate = False
    
    def get_logger(self, name):
        """
        获取指定名称的日志记录器
        
        Args:
            name: 日志记录器名称
            
        Returns:
            logging.Logger: 日志记录器实例
        """
        return logging.getLogger(name)
    
    def get_performance_logger(self):
        """
        获取性能日志记录器
        
        Returns:
            logging.Logger: 性能日志记录器实例
        """
        return logging.getLogger('performance')
    
    def set_level(self, level):
        """
        设置全局日志级别
        
        Args:
            level: 日志级别
        """
        self.level = level
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        for handler in root_logger.handlers:
            handler.setLevel(level)
    
    def log_performance(self, operation, duration, **kwargs):
        """
        记录性能指标
        
        Args:
            operation: 操作名称
            duration: 操作耗时（秒）
            **kwargs: 其他性能相关参数
        """
        perf_logger = self.get_performance_logger()
        
        # 构建性能日志消息
        msg = f"operation={operation}, duration={duration:.4f}s"
        for key, value in kwargs.items():
            msg += f", {key}={value}"
        
        perf_logger.info(msg)

# 初始化全局日志管理器
global_logger = None

def init_logging(log_dir='logs', level=logging.INFO):
    """
    初始化日志系统
    
    Args:
        log_dir: 日志文件存储目录
        level: 日志级别
    """
    global global_logger
    global_logger = LoggingManager(log_dir, level)
    return global_logger

def get_logger(name='default'):
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger: 日志记录器实例
    """
    if global_logger is None:
        init_logging()
    return global_logger.get_logger(name)

def get_performance_logger():
    """
    获取性能日志记录器
    
    Returns:
        logging.Logger: 性能日志记录器实例
    """
    if global_logger is None:
        init_logging()
    return global_logger.get_performance_logger()

def log_performance(operation, duration, **kwargs):
    """
    记录性能指标
    
    Args:
        operation: 操作名称
        duration: 操作耗时（秒）
        **kwargs: 其他性能相关参数
    """
    if global_logger is None:
        init_logging()
    global_logger.log_performance(operation, duration, **kwargs)
