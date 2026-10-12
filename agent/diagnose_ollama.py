import os
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv()
url = os.environ["OLLAMA_BASE_URL"].rstrip("/")
auth = HTTPBasicAuth(
    os.environ["OLLAMA_USERNAME"],
    os.environ["OLLAMA_PASSWORD"],
)

try:
    response = requests.get(url, auth=auth, timeout=20)
    print("HTTP status:", response.status_code)
    print("Content type:", response.headers.get("content-type", "unknown"))
    print("Response preview:", response.text[:300].replace("\n", " "))
except requests.RequestException as exc:
    print("Request error:", type(exc).__name__)
