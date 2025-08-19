import os
import logging
import logging.handlers
from datetime import datetime
import sys
import uuid

class Logger:
    """
    Centralized logging system for the time series forecasting pipeline.
    
    Features:
    - Configurable log levels
    - Multiple output destinations (console, file)
    - Rotating file handlers to manage log size
    - Request ID tracking for tracing operations across components
    - Different formatting for different outputs
    """
    
    def __init__(self, name='TimeSeriesForecasting', log_level=logging.INFO, 
                 log_dir='logs', request_id=None):
        """
        Initialize the logger with the specified configuration.
        
        Args:
            name (str): Name of the logger
            log_level (int): Logging level (default: INFO)
            log_dir (str): Directory for log files
            request_id (str): Optional request ID for tracking operations
        """
        self.name = name
        self.log_level = log_level
        self.log_dir = log_dir
        self.request_id = request_id or str(uuid.uuid4())[:8]
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Add handlers
        self._add_console_handler()
        self._add_file_handler()
        
        self.logger.info(f"Logger initialized [request_id={self.request_id}]")
    
    def _add_console_handler(self):
        """Add console handler to the logger."""
        # Create console handler with a higher log level
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # Create formatter and add it to the handler
        console_format = '%(asctime)s [%(name)s] [%(levelname)s] [%(request_id)s] %(message)s'
        console_formatter = logging.Formatter(console_format, datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(console_formatter)
        
        # Add handler to the logger
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self):
        """Add file handler with rotation to the logger."""
        # Create a file handler which logs even debug messages
        log_file = os.path.join(self.log_dir, f"{self.name.lower()}_{datetime.now().strftime('%Y%m%d')}.log")
        
        # Use RotatingFileHandler to limit file size
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter and add it to the handler
        file_format = ('%(asctime)s [%(name)s] [%(levelname)s] [%(request_id)s] '
                       '[%(filename)s:%(lineno)d] %(message)s')
        file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        
        # Add handler to the logger
        self.logger.addHandler(file_handler)
    
    def get_logger(self):
        """Get the configured logger instance."""
        return logging.LoggerAdapter(self.logger, {'request_id': self.request_id})
    
    def set_level(self, level):
        """Set the logging level for all handlers."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    def update_request_id(self, request_id):
        """Update the request ID for tracking."""
        self.request_id = request_id
        return logging.LoggerAdapter(self.logger, {'request_id': self.request_id})


def setup_logging(name='TimeSeriesForecasting', level=logging.INFO, log_dir='logs', request_id=None):
    """
    Helper function to set up and get a logger.
    
    Args:
        name (str): Logger name
        level (int): Logging level
        log_dir (str): Directory for log files
        request_id (str): Optional request ID for tracking
        
    Returns:
        LoggerAdapter: Configured logger
    """
    logger_instance = Logger(name, level, log_dir, request_id)
    return logger_instance.get_logger()
