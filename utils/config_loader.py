"""
Configuration Loader for CPC-SNN-GW

Centralized configuration management to eliminate hardcoded values.
All parameters should be loaded from YAML files, not hardcoded in Python.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfigPaths:
    """Standard configuration file paths."""
    default_config: Path = Path("configs/default.yaml")
    user_config: Path = Path("configs/user.yaml")
    experiment_config: Path = Path("configs/experiment.yaml")
    
    def __post_init__(self):
        """Ensure all paths are Path objects."""
        self.default_config = Path(self.default_config)
        self.user_config = Path(self.user_config) 
        self.experiment_config = Path(self.experiment_config)


class ConfigLoader:
    """
    Professional configuration loader for CPC-SNN-GW.
    
    Loads configuration from YAML files with hierarchical override:
    1. Default config (required)
    2. User config (optional) - overrides defaults
    3. Experiment config (optional) - overrides user/defaults
    4. Environment variables (optional) - override everything
    """
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration loader.
        
        Args:
            base_dir: Base directory for configuration files
        """
        if base_dir is None:
            base_dir = Path.cwd()
        else:
            base_dir = Path(base_dir)
            
        self.base_dir = base_dir
        self.paths = ConfigPaths()
        self._config_cache = {}
        
    def load_config(self, 
                   config_name: str = "default",
                   user_config: Optional[str] = None,
                   experiment_config: Optional[str] = None,
                   use_env_override: bool = True) -> Dict[str, Any]:
        """
        Load configuration with hierarchical overrides.
        
        Args:
            config_name: Name of default config file (without .yaml)
            user_config: Optional user config file name
            experiment_config: Optional experiment config file name
            use_env_override: Whether to apply environment variable overrides
            
        Returns:
            Merged configuration dictionary
        """
        cache_key = f"{config_name}_{user_config}_{experiment_config}_{use_env_override}"
        if cache_key in self._config_cache:
            logger.debug(f"Using cached config: {cache_key}")
            return self._config_cache[cache_key]
            
        # 1. Load default configuration (required)
        default_path = self.base_dir / "configs" / f"{config_name}.yaml"
        config = self._load_yaml_file(default_path, required=True)
        logger.info(f"✅ Loaded default config: {default_path}")
        
        # 2. Load user configuration (optional override)
        if user_config:
            user_path = self.base_dir / "configs" / f"{user_config}.yaml"
            user_cfg = self._load_yaml_file(user_path, required=False)
            if user_cfg:
                config = self._deep_merge(config, user_cfg)
                logger.info(f"✅ Applied user config: {user_path}")
                
        # 3. Load experiment configuration (optional override)
        if experiment_config:
            exp_path = self.base_dir / "configs" / f"{experiment_config}.yaml"
            exp_cfg = self._load_yaml_file(exp_path, required=False)
            if exp_cfg:
                config = self._deep_merge(config, exp_cfg)
                logger.info(f"✅ Applied experiment config: {exp_path}")
                
        # 4. Apply environment variable overrides
        if use_env_override:
            config = self._apply_env_overrides(config)
            logger.debug("✅ Applied environment variable overrides")
            
        # 5. Validate and resolve paths
        config = self._resolve_paths(config)
        self._validate_config(config)
        
        # Cache the result
        self._config_cache[cache_key] = config
        logger.info(f"✅ Configuration loaded successfully")
        
        return config
    
    def _load_yaml_file(self, path: Path, required: bool = True) -> Optional[Dict[str, Any]]:
        """Load YAML file with error handling."""
        try:
            if not path.exists():
                if required:
                    raise FileNotFoundError(f"Required config file not found: {path}")
                else:
                    logger.debug(f"Optional config file not found: {path}")
                    return None
                    
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
                
            if config is None:
                config = {}
                
            logger.debug(f"Loaded YAML file: {path} ({len(config)} top-level keys)")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config file {path}: {e}")
            raise
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Common environment variable mappings
        env_mappings = {
            'CPC_SNN_DATA_DIR': ('system', 'data_dir'),
            'CPC_SNN_BATCH_SIZE': ('training', 'batch_size'),
            'CPC_SNN_LEARNING_RATE': ('training', 'learning_rate'),
            'CPC_SNN_NUM_EPOCHS': ('training', 'num_epochs'),
            'CPC_SNN_DEVICE': ('system', 'device'),
            'CPC_SNN_USE_WANDB': ('system', 'use_wandb'),
            'CPC_SNN_LOG_LEVEL': ('system', 'log_level'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Type conversion
                if key in ['batch_size', 'num_epochs']:
                    env_value = int(env_value)
                elif key in ['learning_rate', 'memory_fraction']:
                    env_value = float(env_value)
                elif key in ['use_wandb', 'use_tensorboard']:
                    env_value = env_value.lower() in ('true', '1', 'yes')
                    
                # Apply override
                if section not in config:
                    config[section] = {}
                config[section][key] = env_value
                logger.debug(f"Environment override: {env_var}={env_value}")
                
        return config
    
    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve relative paths to absolute paths."""
        path_keys = [
            ('system', 'data_dir'),
            ('system', 'output_dir'),
            ('system', 'cache_dir'),
            ('system', 'checkpoint_dir'),
            ('logging', 'log_file'),
            ('logging', 'tensorboard', 'log_dir'),
        ]
        
        for path_spec in path_keys:
            try:
                section = config
                for key in path_spec[:-1]:
                    section = section.get(key, {})
                    
                if path_spec[-1] in section:
                    path_value = section[path_spec[-1]]
                    if isinstance(path_value, str) and not os.path.isabs(path_value):
                        # Resolve relative to base directory
                        abs_path = (self.base_dir / path_value).resolve()
                        section[path_spec[-1]] = str(abs_path)
                        logger.debug(f"Resolved path: {path_value} -> {abs_path}")
                        
            except (KeyError, TypeError):
                # Path section doesn't exist, skip
                continue
                
        return config
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration values."""
        try:
            # Validate required sections
            required_sections = ['system', 'data', 'model', 'training']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required config section: {section}")
                    
            # Validate data types and ranges
            validations = [
                ('data', 'sample_rate', int, lambda x: x > 0),
                ('training', 'batch_size', int, lambda x: x > 0),
                ('training', 'learning_rate', (int, float), lambda x: 0 < x < 1),
                ('training', 'num_epochs', int, lambda x: x > 0),
                ('system', 'memory_fraction', (int, float), lambda x: 0 < x <= 1),
            ]
            
            for section, key, expected_type, validator in validations:
                try:
                    value = config[section][key]
                    if not isinstance(value, expected_type):
                        logger.warning(f"Config {section}.{key} should be {expected_type}, got {type(value)}")
                    if not validator(value):
                        raise ValueError(f"Invalid value for {section}.{key}: {value}")
                except KeyError:
                    logger.debug(f"Optional config key not found: {section}.{key}")
                    
            logger.debug("✅ Configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation.
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated key path (e.g., 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Config key not found: {key_path}, using default: {default}")
            return default
    
    def save_config(self, config: Dict[str, Any], output_path: Union[str, Path]):
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
            
        logger.info(f"✅ Configuration saved to: {output_path}")


# Global configuration loader instance
_global_loader = None

def get_config_loader(base_dir: Optional[Union[str, Path]] = None) -> ConfigLoader:
    """Get global configuration loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigLoader(base_dir)
    return _global_loader

def load_config(config_name: str = "default", **kwargs) -> Dict[str, Any]:
    """Convenience function to load configuration."""
    loader = get_config_loader()
    return loader.load_config(config_name, **kwargs)

def get_config_value(key_path: str, default: Any = None, config_name: str = "default") -> Any:
    """Convenience function to get single configuration value."""
    config = load_config(config_name)
    loader = get_config_loader()
    return loader.get_value(config, key_path, default)
