import threading
from data.data_processor import DataProcessor
from forecast.forecast_models import ForecastModelSelector
from replenishment.milp_optimizer import MILPOptimizer
from replenishment.automated_replenishment import AutomatedReplenishment
from mlops.mlops_engine import MLOpsEngine
from data.feature_store import FeatureStore
from data.data_warehouse import DataWarehouse
from replenishment.inventory_strategies import InventoryStrategies
from replenishment.inventory_optimization import InventoryOptimization
from system.main import ReplenishmentSystem
from system.config_manager import config_manager

class DependencyInjector:
    """
    依赖注入容器，用于管理和提供所有服务实例
    实现了线程安全的单例模式，确保每个服务只有一个实例
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
                
            # 初始化依赖实例
            self._data_processor = DataProcessor()
            
            # 根据配置决定是否使用GPU
            use_gpu = config_manager.gpu_mode == 'gpu' or (config_manager.gpu_mode == 'auto' and 
                                                        config_manager.get('has_gpu', False))
            
            self._model_selector = ForecastModelSelector(use_gpu=use_gpu)
            self._milp_optimizer = MILPOptimizer()
            self._mlops_engine = MLOpsEngine()
            self._feature_store = FeatureStore()
            self._data_warehouse = DataWarehouse()
            self._inventory_strategies = InventoryStrategies()
            self._inventory_optimization = InventoryOptimization()
            
            # 特殊处理依赖当前实例的服务
            self._replenishment_system = ReplenishmentSystem(
                data_processor=self._data_processor,
                model_selector=self._model_selector,
                milp_optimizer=self._milp_optimizer,
                mlops_engine=self._mlops_engine,
                feature_store=self._feature_store,
                data_warehouse=self._data_warehouse,
                inventory_strategies=self._inventory_strategies,
                inventory_optimization=self._inventory_optimization
            )
            
            # 初始化自动化补货模块（依赖replenishment_system）
            self._automated_replenishment = AutomatedReplenishment(self._replenishment_system)
            
            # 更新replenishment_system的automated_replenishment引用
            self._replenishment_system.automated_replenishment = self._automated_replenishment
            
            self._initialized = True
    
    @property
    def data_processor(self):
        return self._data_processor
    
    @property
    def model_selector(self):
        return self._model_selector
    
    @property
    def milp_optimizer(self):
        return self._milp_optimizer
    
    @property
    def mlops_engine(self):
        return self._mlops_engine
    
    @property
    def feature_store(self):
        return self._feature_store
    
    @property
    def data_warehouse(self):
        return self._data_warehouse
    
    @property
    def inventory_strategies(self):
        return self._inventory_strategies
    
    @property
    def inventory_optimization(self):
        return self._inventory_optimization
    
    @property
    def replenishment_system(self):
        return self._replenishment_system
    
    @property
    def automated_replenishment(self):
        return self._automated_replenishment
    
    def register_custom_model(self, model_name, model_instance):
        """
        注册自定义模型到模型选择器
        
        Args:
            model_name: 模型名称
            model_instance: 模型实例
        """
        if hasattr(self, '_model_selector'):
            self._model_selector.all_models[model_name] = model_instance
    
    def reset(self):
        """
        重置依赖注入容器，重新初始化所有服务实例
        """
        self._initialized = False
        self.__init__()

# 创建全局依赖注入容器实例
dependency_injector = DependencyInjector()
