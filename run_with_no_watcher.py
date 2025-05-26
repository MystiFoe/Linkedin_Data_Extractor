import os
import sys

# Set PYTHONPATH to include the current directory
os.environ["PYTHONPATH"] = os.path.abspath(".")

# Disable Streamlit's file watcher
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Run the Streamlit app
os.system("streamlit run src/ui/app.py")