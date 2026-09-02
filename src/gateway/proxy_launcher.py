"""LiteLLM Proxy Launcher Script.

Starts the LiteLLM Proxy server hosting OpenAI-compatible endpoints with OpenRouter routes.
"""
import os
import sys
import subprocess
from pathlib import Path
from src.common.config import settings, BASE_DIR


def launch_proxy():
    """Launch LiteLLM Proxy using config/litellm_config.yaml."""
    config_file = BASE_DIR / "config" / "litellm_config.yaml"
    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        sys.exit(1)

    print(f"=== Starting LiteLLM Proxy Gateway ===")
    print(f"Config: {config_file}")
    print(f"Host: http://{settings.LITELLM_PROXY_HOST}:{settings.LITELLM_PROXY_PORT}")
    print(f"Master Key: {settings.LITELLM_MASTER_KEY}")

    env = os.environ.copy()
    if settings.OPENROUTER_API_KEY:
        env["OPENROUTER_API_KEY"] = settings.OPENROUTER_API_KEY

    cmd = [
        sys.executable,
        "-m",
        "litellm",
        "--config",
        str(config_file),
        "--host",
        settings.LITELLM_PROXY_HOST,
        "--port",
        str(settings.LITELLM_PROXY_PORT)
    ]

    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nLiteLLM Proxy stopped by user.")


if __name__ == "__main__":
    launch_proxy()
