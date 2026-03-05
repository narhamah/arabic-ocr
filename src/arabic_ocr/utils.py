"""Shared utility helpers."""

from pathlib import Path
from dotenv import load_dotenv


def load_env() -> None:
    """Load environment variables from .env file if present."""
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
