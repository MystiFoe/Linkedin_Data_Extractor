# File: src/api/linkedin_api.py
from .texau_client import TexAuClient
from ..logger import app_logger
import json

class LinkedInAPI:
    """Class for LinkedIn-specific API operations using TexAU"""

    def __init__(self):
        self.client = TexAuClient()
        app_logger.debug("LinkedIn API client initialized")

    def get_automations(self, platform_id):
        """Get all automations for a platform (e.g., LinkedIn)"""
        endpoint = f"public/automations?platformId={platform_id}"
        return self.client._make_request(endpoint, method="GET")

    def get_automation_by_id(self, automation_id):
        """Get automation details by ID"""
        endpoint = f"public/automations/{automation_id}"
        return self.client._make_request(endpoint, method="GET")

    def run_automation(self, name, description, automation_id, connected_account_id, timezone, inputs):
        """Run a TexAu automation using the /run endpoint"""
        endpoint = "public/run"
        payload = {
            "name": name,
            "description": description,
            "automationId": automation_id,
            "connectedAccountId": connected_account_id,
            "timezone": timezone,
            "inputs": inputs
        }
        app_logger.debug(f"Payload sent to TexAU /run: {json.dumps(payload, indent=2)}")
        return self.client._make_request(endpoint, method="POST", payload=payload)

    def get_execution_result(self, execution_id):
        """Get the result of an execution"""
        endpoint = f"public/results/{execution_id}"
        return self.client._make_request(endpoint, method="GET")

    # Example: Search posts by keywords (requires correct automationId and connectedAccountId)
    def search_posts_by_keywords(self, keywords, automation_id, connected_account_id, timezone="Asia/Kolkata"):
        app_logger.info("Searching LinkedIn posts with keywords: {}", keywords)
        inputs = {"keywords": keywords}
        result = self.run_automation(
            name="Keyword Search",
            description="Search LinkedIn posts by keywords",
            automation_id=automation_id,
            connected_account_id=connected_account_id,
            timezone=timezone,
            inputs=inputs
        )
        return result

    def extract_post_data(self, post_url, automation_id, connected_account_id, timezone="Asia/Kolkata"):
        app_logger.info("Extracting data from LinkedIn post: {}", post_url)
        import time
        # Use the correct input key for LinkedIn Post Scraper
        inputs = {"liPostUrl": post_url}
        run_result = self.run_automation(
            name="Post Extraction",
            description="Extract LinkedIn post data",
            automation_id=automation_id,
            connected_account_id=connected_account_id,
            timezone=timezone,
            inputs=inputs
        )
        app_logger.debug(f"TexAu run_automation response: {json.dumps(run_result, indent=2)}")
        # TexAu returns an execution id or workflowId inside 'data'
        data = run_result.get("data", {})
        execution_id = data.get("id") or data.get("workflowId")
        if not execution_id:
            app_logger.error(f"No execution ID returned from TexAu run_automation. Full response: {json.dumps(run_result, indent=2)}")
            return run_result
        # Poll for result (with timeout)
        for _ in range(30):  # up to ~30 seconds
            result = self.get_execution_result(execution_id)
            if result.get("data"):
                return result
            time.sleep(1)
        app_logger.error("Timeout waiting for TexAu execution result for post extraction")
        return {"error": "Timeout waiting for post data extraction result", "run_result": run_result}

    def extract_profile_data(self, profile_url, automation_id, connected_account_id, timezone="Asia/Kolkata"):
        app_logger.info("Extracting data from LinkedIn profile: {}", profile_url)
        inputs = {"url": profile_url}
        result = self.run_automation(
            name="Profile Extraction",
            description="Extract LinkedIn profile data",
            automation_id=automation_id,
            connected_account_id=connected_account_id,
            timezone=timezone,
            inputs=inputs
        )
        return result

    def extract_company_data(self, company_url, automation_id, connected_account_id, timezone="Asia/Kolkata"):
        app_logger.info("Extracting data from LinkedIn company page: {}", company_url)
        inputs = {"url": company_url}
        result = self.run_automation(
            name="Company Extraction",
            description="Extract LinkedIn company data",
            automation_id=automation_id,
            connected_account_id=connected_account_id,
            timezone=timezone,
            inputs=inputs
        )
        return result

    def extract_recent_posts(self, profile_or_company_url, automation_id, connected_account_id, timezone="Asia/Kolkata"):
        app_logger.info("Extracting recent posts from: {}", profile_or_company_url)
        inputs = {"url": profile_or_company_url}
        result = self.run_automation(
            name="Recent Posts Extraction",
            description="Extract recent LinkedIn posts",
            automation_id=automation_id,
            connected_account_id=connected_account_id,
            timezone=timezone,
            inputs=inputs
        )
        return result