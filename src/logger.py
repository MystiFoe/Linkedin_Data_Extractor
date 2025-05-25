from loguru import logger
import sys
import os

class Logger:
    """Logger class for the application"""
    
    @staticmethod
    def setup():
        """Set up the logger configuration"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        
        # Clear existing handlers
        logger.remove()
        
        # Add console handler
        logger.add(sys.stderr, format=log_format, level="INFO")
        
        # Add file handler
        logger.add(
            os.path.join(log_dir, "linkedin_extractor_{time:YYYY-MM-DD}.log"),
            rotation="500 MB",
            retention="10 days",
            compression="zip",
            format=log_format,
            level="DEBUG"
        )
        
        return logger

# Set up logger globally
app_logger = Logger.setup()