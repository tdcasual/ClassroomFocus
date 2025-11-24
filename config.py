"""
Centralized configuration helpers.

Secrets (API keys, etc.) should be stored in a local `.env` file which is
already git-ignored. Example:
    DASH_SCOPE_API_KEY=sk-...
"""
import os
from dotenv import load_dotenv


# Load environment variables from .env if present
load_dotenv()


def get_dashscope_api_key(default: str = "") -> str:
    """Fetch DashScope API key from environment."""
    return os.getenv("DASH_SCOPE_API_KEY", default)
