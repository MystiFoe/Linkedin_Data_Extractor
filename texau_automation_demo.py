import os
import requests
from dotenv import load_dotenv

"""
This script demonstrates how to use the TexAu V2 API to:
1. Get all automations for a platform
2. Get automation details by automation ID
"""

load_dotenv()
API_KEY = os.getenv("TEXAU_API_KEY")
TEXAU_CONTEXT = os.getenv("TEXAU_CONTEXT")
BASE_URL = os.getenv("TEXAU_BASE_URL", "https://api.texau.com/api/v1")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "X-TexAu-Context": TEXAU_CONTEXT,
    "Accept": "*/*"
}

LINKEDIN_PLATFORM_ID = "622f03eb770f6bba0b8facaa"

def get_automations(platform_id=LINKEDIN_PLATFORM_ID, start=0, limit=10):
    url = f"{BASE_URL}/public/automations?platformId={platform_id}&start={start}&limit={limit}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def get_automation_by_id(automation_id):
    url = f"{BASE_URL}/public/automations/{automation_id}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def run_profile_scraper_bulk(profile_urls):
    """
    Run the LinkedIn Profile Scraper automation in bulk using a CSV with liProfileUrl column.
    :param profile_urls: List of LinkedIn profile URLs (strings)
    :return: API response
    """
    import io
    import pandas as pd
    # Prepare CSV with required column name
    csv_buffer = io.StringIO()
    pd.DataFrame({"liProfileUrl": profile_urls}).to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    # Prepare payload for TexAU
    payload = {
        "name": "Bulk Profile Scrape",
        "description": "Bulk scrape LinkedIn profiles via CSV input",
        "automationId": "63f48ee97022e05c116fc798",
        "connectedAccountId": None,  # Fill with your connected LinkedIn account ID if required
        "timezone": "Asia/Kolkata",
        "inputs": {"csvInput": csv_buffer.getvalue()}
    }
    url = f"{BASE_URL}/public/run"
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    # Get all automations for LinkedIn
    print("Getting all automations for LinkedIn platform...")
    automations = get_automations()
    print(automations)

    # Example: Get automation by ID (replace with a real automation ID as needed)
    if automations.get("data"):
        print ("Check this one is wanted")
        automation_details = get_automation_by_id('63f48ee97022e05c116fc798')
        print(automation_details)

    # Example: Bulk run profile scraper
    test_urls = [
        "https://www.linkedin.com/in/johndoe",
        "https://www.linkedin.com/in/janedoe"
    ]
    print("Running bulk profile scrape...")
    bulk_result = run_profile_scraper_bulk(test_urls)
    print(bulk_result)
