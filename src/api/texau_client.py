# File: src/api/texau_client.py
import requests
import json
from ..logger import app_logger
from ..config import Config

class TexAuClient:
    """Client for interacting with TexAU API"""
    
    def __init__(self):
        """Initialize the TexAU API client"""
        config = Config.load_config()
        self.api_key = config["TEXAU_API_KEY"]
        self.base_url = config.get("TEXAU_BASE_URL", "https://api.texau.com/api/v1")
        
        # Update headers to include X-TexAu-Context
        texau_context = config.get("TEXAU_CONTEXT", "{}")
        if isinstance(texau_context, str):
            try:
                texau_context = json.loads(texau_context)
            except json.JSONDecodeError:
                app_logger.error("Invalid TEXAU_CONTEXT JSON format")
                texau_context = {}
                
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-TexAu-Context": json.dumps(texau_context),
            "Accept": "*/*"
        }
        app_logger.debug("TexAU client initialized with complete authentication")
        
    def _make_request(self, endpoint, method="GET", payload=None):
        """Make a request to the TexAU API
        
        Args:
            endpoint: API endpoint
            method: HTTP method (GET, POST, etc.)
            payload: Request payload for POST requests
            
        Returns:
            Response data from API
        """
        # Ensure endpoint is relative to /api/v1/
        if endpoint.startswith("public/"):
            url = f"{self.base_url}/{endpoint}"
        else:
            url = f"{self.base_url}/public/{endpoint}"
        
        try:
            app_logger.debug("Making API request to endpoint: {}", endpoint)
            
            if method == "GET":
                response = requests.get(url, headers=self.headers)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=payload)
            elif method == "PUT":
                response = requests.put(url, headers=self.headers, json=payload)
            elif method == "DELETE":
                response = requests.delete(url, headers=self.headers, json=payload)
            else:
                app_logger.error("Unsupported HTTP method: {}", method)
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Handle TexAU-specific error codes
            if response.status_code == 401:
                app_logger.error("Authentication failed. Check your API key and context.")
                raise Exception("Authentication failed. Check your API key and context.")
                
            if response.status_code == 403:
                app_logger.error("Access forbidden. Check your organization ID and permissions.")
                raise Exception("Access forbidden. Check your organization ID and permissions.")
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            app_logger.error("API request failed: {}", str(e))
            raise