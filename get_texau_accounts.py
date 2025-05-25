import os
from dotenv import load_dotenv
import requests

load_dotenv()
API_KEY = os.getenv("TEXAU_API_KEY")
TEXAU_CONTEXT = os.getenv("TEXAU_CONTEXT")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "X-TexAu-Context": TEXAU_CONTEXT,
    "Accept": "*/*"
}

response = requests.get(
    "https://api.texau.com/api/v1/public/social-accounts?platformId=622f03eb770f6bba0b8facaa",
    headers=HEADERS,
)
data = response.json()
print(data)
