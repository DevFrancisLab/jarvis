
import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
username = os.getenv("OLLAMA_USERNAME")
password = os.getenv("OLLAMA_PASSWORD")

if not all([base_url, username, password]):
    raise SystemExit("Missing Ollama settings in .env")

try:
    response = requests.post(
        f"{base_url}/api/chat",
        auth=HTTPBasicAuth(username, password),
        json={
            "model": "llama3.2:3b",
            "messages": [
                {"role": "user", "content": "Say hello. You are Jarvis."}
            ],
            "stream": False,
        },
        timeout=120,
    )

    print("HTTP status:", response.status_code)
    response.raise_for_status()

    data = response.json()
    print("Jarvis:", data["message"]["content"])

except requests.RequestException as exc:
    print("Request failed:", type(exc).__name__, str(exc)[:300])