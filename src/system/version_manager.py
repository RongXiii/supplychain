import os
import json
import datetime
from pathlib import Path
import subprocess
from .logging_manager import get_logger

class VersionManager:
    """
    版本管理器，负责模型、代码和配置的版本管理
    """
    
    def __init__(self, version_dir='versions'):
        """
        初始化版本管理器
        
        Args:
            version_dir: 版本管理目录
        """
        self.logger = get_logger('version_manager')
        self.logger.info("Initializing VersionManager")
        
        self.version_dir = version_dir
        self.versions_file = os.path.join(version_dir, 'versions.json')
        
        # 创建版本目录
        Path(self.version_dir).mkdir(exist_ok=True)
        self.logger.debug(f"Version directory: {self.version_dir}")
        
        # 初始化版本记录文件
        if not os.path.exists(self.versions_file):
            with open(self.versions_file, 'w') as f:
                json.dump({'versions': []}, f, indent=2)
            self.logger.debug(f"Created versions file: {self.versions_file}")
    
    def create_version(self, version_name, description, model_paths=None, config_paths=None):
        """
        创建新版本
        
        Args:
            version_name: 版本名称
            description: 版本描述
            model_paths: 模型文件路径列表
            config_paths: 配置文件路径列表
            
        Returns:
            version_info: 版本信息
        """
        self.logger.info(f"Creating new version: {version_name}")
        
        # 获取当前时间
        timestamp = datetime.datetime.now().isoformat()
        
        # 获取Git提交信息（如果有的话）
        git_info = self._get_git_info()
        
        # 创建版本信息
        version_info = {
            'version': version_name,
            'description': description,
            'timestamp': timestamp,
            'git_info': git_info,
            'models': model_paths or [],
            'configs': config_paths or [],
            'status': 'active'
        }
        
        # 加载现有版本记录
        try:
            with open(self.versions_file, 'r') as f:
                versions_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load versions file: {e}", exc_info=True)
            versions_data = {'versions': []}
        
        # 检查版本是否已存在
        for existing_version in versions_data['versions']:
            if existing_version['version'] == version_name:
                self.logger.error(f"Version already exists: {version_name}")
                return None
        
        # 添加新版本
        versions_data['versions'].append(version_info)
        
        # 保存版本记录
        try:
            with open(self.versions_file, 'w') as f:
                json.dump(versions_data, f, indent=2)
            
            self.logger.info(f"Created version: {version_name}")
            return version_info
        except Exception as e:
            self.logger.error(f"Failed to save version: {version_name}, error: {e}", exc_info=True)
            return None
    
    def get_version_info(self, version_name):
        """
        获取版本信息
        
        Args:
            version_name: 版本名称
            
        Returns:
            version_info: 版本信息
        """
        self.logger.info(f"Getting version info: {version_name}")
        
        try:
            with open(self.versions_file, 'r') as f:
                versions_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load versions file: {e}", exc_info=True)
            return None
        
        for version in versions_data['versions']:
            if version['version'] == version_name:
                self.logger.debug(f"Found version info: {version_name}")
                return version
        
        self.logger.warning(f"Version not found: {version_name}")
        return None
    
    def list_versions(self, status=None):
        """
        列出所有版本
        
        Args:
            status: 版本状态过滤 ('active', 'inactive')
            
        Returns:
            versions: 版本列表
        """
        self.logger.info(f"Listing versions with status: {status}")
        
        try:
            with open(self.versions_file, 'r') as f:
                versions_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load versions file: {e}", exc_info=True)
            return []
        
        versions = versions_data['versions']
        
        # 状态过滤
        if status:
            versions = [v for v in versions if v.get('status') == status]
        
        # 按时间倒序排序
        versions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        self.logger.debug(f"Found {len(versions)} versions")
        return versions
    
    def set_version_status(self, version_name, status):
        """
        设置版本状态
        
        Args:
            version_name: 版本名称
            status: 版本状态 ('active', 'inactive')
            
        Returns:
            bool: 是否成功
        """
        self.logger.info(f"Setting version status: {version_name} -> {status}")
        
        try:
            with open(self.versions_file, 'r') as f:
                versions_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load versions file: {e}", exc_info=True)
            return False
        
        # 更新版本状态
        updated = False
        for version in versions_data['versions']:
            if version['version'] == version_name:
                version['status'] = status
                updated = True
                break
        
        if not updated:
            self.logger.error(f"Version not found: {version_name}")
            return False
        
        # 保存更新后的版本记录
        try:
            with open(self.versions_file, 'w') as f:
                json.dump(versions_data, f, indent=2)
            
            self.logger.info(f"Updated version status: {version_name} -> {status}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save version status: {version_name}, error: {e}", exc_info=True)
            return False
    
    def _get_git_info(self):
        """
        获取Git提交信息
        
        Returns:
            git_info: Git提交信息
        """
        git_info = {
            'commit_id': None,
            'branch': None,
            'author': None,
            'date': None
        }
        
        # 检查是否在Git仓库中
        try:
            # 获取当前分支
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                          stderr=subprocess.STDOUT, text=True).strip()
            git_info['branch'] = branch
            
            # 获取最近提交ID
            commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                             stderr=subprocess.STDOUT, text=True).strip()
            git_info['commit_id'] = commit_id
            
            # 获取最近提交作者和日期
            commit_info = subprocess.check_output(['git', 'log', '-1', '--format=%an|%ad'], 
                                               stderr=subprocess.STDOUT, text=True).strip()
            if '|' in commit_info:
                author, date = commit_info.split('|', 1)
                git_info['author'] = author
                git_info['date'] = date
            
        except subprocess.CalledProcessError as e:
            self.logger.debug(f"Not in a Git repository or Git command failed: {e.output}")
        except Exception as e:
            self.logger.error(f"Failed to get Git info: {e}", exc_info=True)
        
        return git_info
    
    def export_version(self, version_name, export_dir):
        """
        导出版本信息和相关文件
        
        Args:
            version_name: 版本名称
            export_dir: 导出目录
            
        Returns:
            bool: 是否成功
        """
        self.logger.info(f"Exporting version: {version_name} to {export_dir}")
        
        # 获取版本信息
        version_info = self.get_version_info(version_name)
        if not version_info:
            self.logger.error(f"Version not found: {version_name}")
            return False
        
        # 创建导出目录
        export_path = os.path.join(export_dir, f"version_{version_name}")
        Path(export_path).mkdir(exist_ok=True, parents=True)
        
        try:
            # 保存版本信息
            version_file = os.path.join(export_path, 'version_info.json')
            with open(version_file, 'w') as f:
                json.dump(version_info, f, indent=2)
            
            # 复制模型文件
            models_dir = os.path.join(export_path, 'models')
            Path(models_dir).mkdir(exist_ok=True)
            for model_path in version_info['models']:
                if os.path.exists(model_path):
                    dest_path = os.path.join(models_dir, os.path.basename(model_path))
                    os.system(f'copy "{model_path}" "{dest_path}"')  # Windows命令
                    self.logger.debug(f"Copied model: {model_path} -> {dest_path}")
            
            # 复制配置文件
            configs_dir = os.path.join(export_path, 'configs')
            Path(configs_dir).mkdir(exist_ok=True)
            for config_path in version_info['configs']:
                if os.path.exists(config_path):
                    dest_path = os.path.join(configs_dir, os.path.basename(config_path))
                    os.system(f'copy "{config_path}" "{dest_path}"')  # Windows命令
                    self.logger.debug(f"Copied config: {config_path} -> {dest_path}")
            
            self.logger.info(f"Exported version: {version_name} to {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export version: {version_name}, error: {e}", exc_info=True)
            return False

# 全局版本管理器实例
version_manager = VersionManager()

# 导出版本管理函数
def create_version(version_name, description, model_paths=None, config_paths=None):
    """
    创建新版本（简化接口）
    """
    return version_manager.create_version(version_name, description, model_paths, config_paths)

def get_version(version_name):
    """
    获取版本信息（简化接口）
    """
    return version_manager.get_version_info(version_name)

def list_versions(status=None):
    """
    列出版本（简化接口）
    """
    return version_manager.list_versions(status)
