import os
import sys
import subprocess

# Set environment variables to disable Streamlit's file watcher
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath("."))

# Run the Streamlit app
if __name__ == "__main__":
    # Use subprocess to run streamlit directly with command-line arguments
    # This avoids the issues with bootstrap.run()
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "src/ui/app.py",
        "--server.fileWatcherType=none",
        "--browser.serverAddress=localhost",
        "--server.port=8502"
    ]
    
    # Run the command
    subprocess.run(cmd)