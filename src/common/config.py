"""Centralized Configuration Settings."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base Directory of AIPoc
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # OpenRouter & LiteLLM Settings
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API Key")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter Base URL")
    LITELLM_PROXY_HOST: str = Field(default="127.0.0.1", description="LiteLLM Host")
    LITELLM_PROXY_PORT: int = Field(default=4000, description="LiteLLM Port")
    LITELLM_MASTER_KEY: str = Field(default="sk-litellm-master-key-poc", description="LiteLLM Master Key")

    # Default Models
    PRIMARY_MODEL: str = Field(default="openrouter/inclusionai/ling-3.0-flash-fin:free")
    FALLBACK_MODEL: str = Field(default="openrouter/meta-llama/llama-3.3-70b-instruct:free")
    PLANNER_MODEL: str = Field(default="openrouter/inclusionai/ling-3.0-flash-fin:free")
    SYNTHESIS_MODEL: str = Field(default="openrouter/inclusionai/ling-3.0-flash-fin:free")

    # Langfuse Settings
    LANGFUSE_PUBLIC_KEY: str = Field(default="pk-lf-mock-key-12345")
    LANGFUSE_SECRET_KEY: str = Field(default="sk-lf-mock-key-12345")
    LANGFUSE_HOST: str = Field(default="https://cloud.langfuse.com")
    LANGFUSE_BASE_URL: str = Field(default="https://cloud.langfuse.com")
    LANGFUSE_ENABLED: bool = Field(default=True)

    # NeMo Guardrails
    GUARDRAILS_CONFIG_PATH: str = Field(default=str(BASE_DIR / "config" / "nemoguardrails"))
    GUARDRAILS_ENABLED: bool = Field(default=True)

    # Agent & Tool Settings
    MAX_RESEARCH_STEPS: int = Field(default=4)
    ENABLE_LIVE_WEB_SEARCH: bool = Field(default=True)
    SIMULATION_FALLBACK: bool = Field(default=True)


settings = Settings()

# Ensure OpenRouter API key is synced into environment for libraries expecting OPENROUTER_API_KEY / OPENAI_API_KEY
if settings.OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = settings.OPENROUTER_API_KEY
    # OpenRouter operates with OpenAI compatible format
    os.environ["OPENAI_API_BASE"] = settings.OPENROUTER_BASE_URL
if settings.LANGFUSE_PUBLIC_KEY:
    host = settings.LANGFUSE_BASE_URL or settings.LANGFUSE_HOST
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = host
    os.environ["LANGFUSE_BASEURL"] = host
