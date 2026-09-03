"""Centralized Configuration and Settings via Pydantic."""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Multi-Provider API Keys
    OPENROUTER_API_KEY: str = Field(default="")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")
    GEMINI_API_KEY: str = Field(default="")
    GROQ_API_KEY: str = Field(default="")

    # Gateway & Proxy Settings
    LITELLM_PROXY_HOST: str = Field(default="127.0.0.1")
    LITELLM_PROXY_PORT: int = Field(default=4000)
    LITELLM_MASTER_KEY: str = Field(default="sk-litellm-master-key")
    SERVER_PORT: int = Field(default=8080)

    # Model Defaults
    PRIMARY_MODEL: str = Field(default="groq/qwen/qwen3.8-27b")
    FALLBACK_MODEL: str = Field(default="openrouter/inclusionai/ling-3.0-flash-fin:free")
    REASONING_MODEL: str = Field(default="groq/groq/compound")
    FAST_GUARD_MODEL: str = Field(default="groq/meta-llama/llama-prompt-guard-2-86m")

    # Langfuse Observability
    LANGFUSE_PUBLIC_KEY: str = Field(default="")
    LANGFUSE_SECRET_KEY: str = Field(default="")
    LANGFUSE_HOST: str = Field(default="https://cloud.langfuse.com")
    LANGFUSE_ENABLED: bool = Field(default=True)

    # NeMo Guardrails
    GUARDRAILS_CONFIG_PATH: str = Field(default="./config/nemoguardrails")
    GUARDRAILS_ENABLED: bool = Field(default=True)

    # Agent Configurations
    MAX_RESEARCH_STEPS: int = Field(default=4)
    ENABLE_LIVE_WEB_SEARCH: bool = Field(default=True)
    SIMULATION_FALLBACK: bool = Field(default=True)


settings = Settings()
