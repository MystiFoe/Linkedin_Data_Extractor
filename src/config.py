# File: src/config.py
import os
import json
from dotenv import load_dotenv
from .logger import app_logger

class Config:
    """Configuration class for the application"""
    
    @staticmethod
    def load_config():
        """Load environment variables from .env file"""
        load_dotenv()
        app_logger.info("Loading environment variables")
        
        config = {
            "TEXAU_API_KEY": os.getenv("TEXAU_API_KEY"),
            "TEXAU_BASE_URL": os.getenv("TEXAU_BASE_URL", "https://api.texau.com/api/v1"),
            "TEXAU_CONTEXT": os.getenv("TEXAU_CONTEXT")
        }
        
        # Check if required environment variables are set
        if not config["TEXAU_API_KEY"]:
            app_logger.error("TEXAU_API_KEY environment variable not set")
            raise ValueError("TEXAU_API_KEY environment variable not set")
        
        if not config["TEXAU_CONTEXT"]:
            app_logger.warning("TEXAU_CONTEXT environment variable not set")
            
        return config