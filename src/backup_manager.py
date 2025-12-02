import os
import shutil
import zipfile
import datetime
from pathlib import Path
import logging
from .logging_manager import get_logger

class BackupManager:
    """
    备份管理器，负责数据和模型的定期备份、恢复和管理
    """
    
    def __init__(self, backup_dir='backups', keep_backups=7):
        """
        初始化备份管理器
        
        Args:
            backup_dir: 备份目录
            keep_backups: 保留的备份数量
        """
        self.logger = get_logger('backup_manager')
        self.logger.info("Initializing BackupManager")
        
        self.backup_dir = backup_dir
        self.keep_backups = keep_backups
        
        # 创建备份目录
        Path(self.backup_dir).mkdir(exist_ok=True)
        self.logger.debug(f"Backup directory: {self.backup_dir}")
    
    def backup_data(self, data_dirs):
        """
        备份数据文件
        
        Args:
            data_dirs: 要备份的数据目录列表
            
        Returns:
            backup_path: 备份文件路径
        """
        self.logger.info(f"Starting data backup for directories: {data_dirs}")
        
        # 生成备份文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"data_backup_{timestamp}.zip"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for data_dir in data_dirs:
                    if not os.path.exists(data_dir):
                        self.logger.warning(f"Data directory not found: {data_dir}")
                        continue
                    
                    # 遍历目录并添加到zip文件
                    for root, dirs, files in os.walk(data_dir):
                        for file in files:
                            # 跳过临时文件和__pycache__目录
                            if file.endswith('.pyc') or '__pycache__' in root:
                                continue
                            
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, os.path.dirname(data_dir))
                            zipf.write(file_path, arcname)
                            self.logger.debug(f"Added to backup: {file_path}")
            
            self.logger.info(f"Data backup completed: {backup_path}")
            self._cleanup_old_backups('data_backup_')
            return backup_path
        except Exception as e:
            self.logger.error(f"Data backup failed: {e}", exc_info=True)
            return None
    
    def backup_models(self, models_dir):
        """
        备份模型文件
        
        Args:
            models_dir: 模型目录
            
        Returns:
            backup_path: 备份文件路径
        """
        self.logger.info(f"Starting model backup for directory: {models_dir}")
        
        # 生成备份文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"model_backup_{timestamp}.zip"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if not os.path.exists(models_dir):
                    self.logger.warning(f"Models directory not found: {models_dir}")
                    return None
                
                # 遍历模型目录并添加到zip文件
                for root, dirs, files in os.walk(models_dir):
                    for file in files:
                        # 跳过临时文件
                        if file.endswith('.pyc') or '__pycache__' in root:
                            continue
                        
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(models_dir))
                        zipf.write(file_path, arcname)
                        self.logger.debug(f"Added to backup: {file_path}")
            
            self.logger.info(f"Model backup completed: {backup_path}")
            self._cleanup_old_backups('model_backup_')
            return backup_path
        except Exception as e:
            self.logger.error(f"Model backup failed: {e}", exc_info=True)
            return None
    
    def restore_data(self, backup_path, restore_dir):
        """
        恢复数据备份
        
        Args:
            backup_path: 备份文件路径
            restore_dir: 恢复目录
            
        Returns:
            bool: 恢复是否成功
        """
        self.logger.info(f"Restoring data from backup: {backup_path} to {restore_dir}")
        
        try:
            if not os.path.exists(backup_path):
                self.logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # 创建恢复目录
            Path(restore_dir).mkdir(exist_ok=True)
            
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(restore_dir)
            
            self.logger.info(f"Data restoration completed: {restore_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Data restoration failed: {e}", exc_info=True)
            return False
    
    def restore_models(self, backup_path, restore_dir):
        """
        恢复模型备份
        
        Args:
            backup_path: 备份文件路径
            restore_dir: 恢复目录
            
        Returns:
            bool: 恢复是否成功
        """
        self.logger.info(f"Restoring models from backup: {backup_path} to {restore_dir}")
        
        try:
            if not os.path.exists(backup_path):
                self.logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # 创建恢复目录
            Path(restore_dir).mkdir(exist_ok=True)
            
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(restore_dir)
            
            self.logger.info(f"Model restoration completed: {restore_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Model restoration failed: {e}", exc_info=True)
            return False
    
    def _cleanup_old_backups(self, prefix):
        """
        清理旧备份文件，保留指定数量的备份
        
        Args:
            prefix: 备份文件前缀
        """
        self.logger.debug(f"Cleaning up old backups with prefix: {prefix}")
        
        # 获取所有备份文件
        backup_files = []
        for file in os.listdir(self.backup_dir):
            if file.startswith(prefix) and file.endswith('.zip'):
                file_path = os.path.join(self.backup_dir, file)
                backup_files.append((os.path.getmtime(file_path), file_path))
        
        # 按修改时间排序（最新的在后）
        backup_files.sort(key=lambda x: x[0])
        
        # 删除旧备份，保留指定数量
        while len(backup_files) > self.keep_backups:
            _, old_backup = backup_files.pop(0)
            try:
                os.remove(old_backup)
                self.logger.debug(f"Removed old backup: {old_backup}")
            except Exception as e:
                self.logger.error(f"Failed to remove old backup: {old_backup}, error: {e}")
    
    def get_backup_list(self, backup_type='all'):
        """
        获取备份列表
        
        Args:
            backup_type: 备份类型 ('all', 'data', 'model')
            
        Returns:
            backup_list: 备份文件列表，按时间倒序排列
        """
        self.logger.info(f"Getting backup list for type: {backup_type}")
        
        backup_files = []
        
        for file in os.listdir(self.backup_dir):
            if file.endswith('.zip'):
                if backup_type == 'data' and 'data_backup_' in file:
                    backup_files.append(file)
                elif backup_type == 'model' and 'model_backup_' in file:
                    backup_files.append(file)
                elif backup_type == 'all':
                    backup_files.append(file)
        
        # 按时间倒序排序
        def get_timestamp(file_name):
            try:
                # 提取文件名中的时间戳
                timestamp_part = file_name.split('_')[2].replace('.zip', '')
                return datetime.datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
            except:
                return datetime.datetime.min
        
        backup_files.sort(key=get_timestamp, reverse=True)
        
        self.logger.debug(f"Found {len(backup_files)} backups")
        return backup_files

# 全局备份管理器实例
backup_manager = BackupManager()

# 导出备份函数
def backup_all():
    """
    备份所有数据和模型
    """
    logger = get_logger('backup_manager')
    logger.info("Starting full backup")
    
    # 备份数据目录
    data_dirs = ['data', 'metrics']
    data_backup = backup_manager.backup_data(data_dirs)
    
    # 备份模型目录
    models_dir = 'models'
    model_backup = backup_manager.backup_models(models_dir)
    
    logger.info(f"Full backup completed. Data: {data_backup}, Model: {model_backup}")
    return data_backup, model_backup
