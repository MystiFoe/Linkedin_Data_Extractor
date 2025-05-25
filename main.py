import os
import sys

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ui.app import LinkedInExtractorApp
from src.logger import app_logger

def main():
    """Main entry point for the application"""
    try:
        app_logger.info("Starting LinkedIn Data Extractor")
        app = LinkedInExtractorApp()
        app.run()
    except Exception as e:
        app_logger.error("Application error: {}", str(e))
        import streamlit as st
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()