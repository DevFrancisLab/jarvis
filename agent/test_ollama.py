import os
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv()

base_url = os.environ["OLLAMA_BASE_URL"].rstrip("/")
username = os.environ["OLLAMA_USERNAME"]
password = os.environ["OLLAMA_PASSWORD"]

print("Checking remote Ollama connection...")

try:
    response = requests.get(
        f"{base_url}/api/tags",
        auth=HTTPBasicAuth(username, password),
        timeout=30,
    )
    print("HTTP status:", response.status_code)

    response.raise_for_status()
    data = response.json()

    models = data.get("models", [])
    if not models:
        print("Connected, but no Ollama models were listed.")
    else:
        print("Available models:")
        for model in models:
            print("-", model.get("name", "unknown"))

except requests.RequestException as exc:
    print("Connection failed:", type(exc).__name__)
    if getattr(exc, "response", None) is not None:
        print("Response status:", exc.response.status_code)
