import os
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path

class ParameterHandler:
    """
    Handles configuration parameters for time series forecasting models and processes.
    
    This class manages loading, validating, and transforming configuration parameters
    from various sources (files, command line, defaults) and provides a unified interface
    for accessing these parameters throughout the application.
    """
    
    def __init__(self, config=None, config_path=None):
        """
        Initialize the parameter handler with configuration.
        
        Args:
            config (dict, optional): Configuration dict to use
            config_path (str, optional): Path to a config file to load
        """
        self.logger = logging.getLogger(__name__)
        self.config = {}
        self.runtime_params = {}
        
        # Initialize with default configs
        self._set_defaults()
        
        # Load from file if provided
        if config_path:
            self.load_config(config_path)
        
        # Update with provided config dict
        if config:
            self.update_config(config)
        
        # Add runtime parameters
        self._set_runtime_parameters()
    
    def _set_defaults(self):
        """Set default configuration values."""
        self.config = {
            'output_dir': 'outputs',
            'results_dir': 'outputs/results',
            'model_dir': 'outputs/models',
            'log_dir': 'logs',
            'log_level': 'INFO',
            'forecast_horizon': 10,
            'validation_split': 0.2,
            'test_split': 0.1
        }
    
    def load_config(self, config_path):
        """
        Load configuration from a YAML or JSON file.
        
        Args:
            config_path (str): Path to the configuration file
            
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            if not os.path.exists(config_path):
                self.logger.error(f"Configuration file not found: {config_path}")
                return False
            
            file_ext = os.path.splitext(config_path)[1].lower()
            
            if file_ext in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            elif file_ext == '.json':
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
            else:
                self.logger.error(f"Unsupported config file type: {file_ext}")
                return False
            
            self.update_config(loaded_config)
            self.logger.info(f"Loaded configuration from {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            return False
    
    def update_config(self, new_config):
        """
        Update configuration with new values.
        
        Args:
            new_config (dict): New configuration values
        """
        if not new_config:
            return
            
        for key, value in new_config.items():
            if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                # Recursively update nested dictionaries
                self.config[key].update(value)
            else:
                # Direct update for non-dict values or new keys
                self.config[key] = value
    
    def _set_runtime_parameters(self):
        """Set runtime parameters such as timestamps and run IDs."""
        now = datetime.now()
        
        self.runtime_params = {
            'timestamp': now.strftime("%Y%m%d_%H%M%S"),
            'date': now.strftime("%Y%m%d"),
            'time': now.strftime("%H%M%S"),
            'run_id': f"run_{now.strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Get model name if available
        model_name = self.config.get('model_name', self.config.get('model', 'unknown'))
        self.runtime_params['model_name'] = model_name
        
        # Create full run ID with model name
        self.runtime_params['full_run_id'] = f"{model_name}_{self.runtime_params['timestamp']}"
        
        # Create output directories
        self._create_output_directories()
    
    def _create_output_directories(self):
        """Create necessary output directories based on configuration."""
        for dir_key in ['output_dir', 'results_dir', 'model_dir', 'log_dir']:
            if dir_key in self.config:
                # Replace any runtime parameters in the path
                dir_path = self.config[dir_key]
                for param_key, param_value in self.runtime_params.items():
                    placeholder = f"{{{param_key}}}"
                    if placeholder in dir_path:
                        dir_path = dir_path.replace(placeholder, param_value)
                
                # Create directory
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                
                # Update config with expanded path
                self.config[dir_key] = dir_path
    
    def get(self, key, default=None):
        """
        Get a configuration parameter.
        
        Args:
            key (str): Parameter key to retrieve
            default: Default value if key not found
            
        Returns:
            Parameter value or default
        """
        # Check in runtime params first
        if key in self.runtime_params:
            return self.runtime_params[key]
        
        # Then check in main config
        return self.config.get(key, default)
    
    def set(self, key, value):
        """
        Set a configuration parameter.
        
        Args:
            key (str): Parameter key to set
            value: Value to set
            
        Returns:
            bool: True if set, False if error
        """
        try:
            if key in self.runtime_params:
                self.runtime_params[key] = value
            else:
                self.config[key] = value
            return True
        except Exception as e:
            self.logger.error(f"Error setting parameter {key}: {str(e)}")
            return False
    
    def get_all(self):
        """
        Get the complete configuration including runtime parameters.
        
        Returns:
            dict: Complete configuration
        """
        # Create a deep copy to avoid modification of the original
        all_config = self.config.copy()
        
        # Add runtime parameters
        all_config['runtime'] = self.runtime_params.copy()
        
        return all_config
    
    def save_config(self, output_path=None):
        """
        Save the current configuration to a file.
        
        Args:
            output_path (str, optional): Path to save the configuration
            
        Returns:
            str: Path to the saved configuration file
        """
        if not output_path:
            # Default to results directory with timestamp
            output_path = os.path.join(
                self.get('results_dir', 'outputs/results'),
                f"config_{self.runtime_params['timestamp']}.yaml"
            )
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get all config including runtime params
            all_config = self.get_all()
            
            # Determine format based on file extension
            if output_path.endswith('.json'):
                with open(output_path, 'w') as f:
                    json.dump(all_config, f, indent=2)
            else:
                with open(output_path, 'w') as f:
                    yaml.dump(all_config, f, default_flow_style=False)
            
            self.logger.info(f"Configuration saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            return None
    
    def get_consolidated_path(self, path_type, filename=None):
        """
        Get a consolidated path based on configuration.
        
        Args:
            path_type (str): Type of path (e.g., 'output', 'model', 'result')
            filename (str, optional): Optional filename to append
            
        Returns:
            str: Consolidated path
        """
        base_paths = {
            'output': self.get('output_dir'),
            'model': self.get('model_dir'),
            'result': self.get('results_dir'),
            'log': self.get('log_dir')
        }
        
        base_path = base_paths.get(path_type, self.get('output_dir'))
        
        if filename:
            return os.path.join(base_path, filename)
        
        return base_path
