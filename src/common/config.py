"""Centralized Application Configuration and Environment Settings.

This module uses Pydantic Settings (`pydantic_settings.BaseSettings`) to load,
validate, and expose environment variables from the local `.env` file with
type-safe defaults.

Key Capabilities:
    - Multi-provider API keys (OpenRouter, Google Gemini, Groq)
    - Default model routing aliases (Primary, Fallback, Reasoning, Safety Guard)
    - Langfuse cloud observability credentials and endpoints
    - NeMo Guardrails configuration paths and enforcement toggles
    - Agent execution limits (max research iterations, web search toggles)
    - Granular Debug Mode toggle for deep function-level tracing

Usage:
    from src.common.config import settings
    print(settings.PRIMARY_MODEL)
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings schema loaded from environment and `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Gracefully ignore any unrecognized extra env variables
    )

    # =========================================================================
    # 0. Global Debug & Verbosity Controls
    # =========================================================================
    DEBUG_MODE: bool = Field(
        default=False,
        description="Enables detailed function-level debug prints, payload dumps, and state transitions."
    )

    # =========================================================================
    # 1. Multi-Provider Credentials
    # =========================================================================
    OPENROUTER_API_KEY: str = Field(
        default="",
        description="API key for OpenRouter multi-model aggregation gateway."
    )
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base HTTP URL for OpenRouter OpenAI-compatible completions endpoint."
    )
    GEMINI_API_KEY: str = Field(
        default="",
        description="Google AI Studio API key for native Gemini and Gemma models."
    )
    GROQ_API_KEY: str = Field(
        default="",
        description="Groq Cloud API key for ultra-low latency LPU model inference."
    )

    # =========================================================================
    # 2. Gateway & Web Server Network Settings
    # =========================================================================
    LITELLM_PROXY_HOST: str = Field(
        default="127.0.0.1",
        description="Host interface for optional local LiteLLM proxy."
    )
    LITELLM_PROXY_PORT: int = Field(
        default=4000,
        description="Port for optional local LiteLLM proxy."
    )
    LITELLM_MASTER_KEY: str = Field(
        default="sk-litellm-master-key",
        description="Virtual bearer token for local LiteLLM proxy authentication."
    )
    SERVER_PORT: int = Field(
        default=8080,
        description="Default HTTP listening port for the FastAPI ChatGPT web application."
    )

    # =========================================================================
    # 3. Model Routing Defaults
    # =========================================================================
    PRIMARY_MODEL: str = Field(
        default="groq/qwen/qwen3.8-27b",
        description="Default model used for general-purpose chat and agent planning."
    )
    FALLBACK_MODEL: str = Field(
        default="openrouter/inclusionai/ling-3.0-flash-fin:free",
        description="Emergency fallback model if the primary provider hits rate limits (429) or outages (503)."
    )
    REASONING_MODEL: str = Field(
        default="groq/groq/compound",
        description="High-reasoning model used for multi-step agent planning and synthesis."
    )
    FAST_GUARD_MODEL: str = Field(
        default="groq/meta-llama/llama-prompt-guard-2-86m",
        description="Sub-200ms safety classifier model running on Groq LPUs for NeMo Guardrails."
    )

    # =========================================================================
    # 4. Langfuse Cloud Observability
    # =========================================================================
    LANGFUSE_PUBLIC_KEY: str = Field(
        default="",
        description="Langfuse project public key (starts with pk-lf-)."
    )
    LANGFUSE_SECRET_KEY: str = Field(
        default="",
        description="Langfuse project secret key (starts with sk-lf-)."
    )
    LANGFUSE_HOST: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse server host endpoint URL (cloud or self-hosted)."
    )
    LANGFUSE_ENABLED: bool = Field(
        default=True,
        description="Global master switch to enable/disable Langfuse telemetry syncing."
    )

    # =========================================================================
    # 5. NeMo Guardrails Safety Settings
    # =========================================================================
    GUARDRAILS_CONFIG_PATH: str = Field(
        default="./config/nemoguardrails",
        description="Filesystem directory containing Colang rails (.co) and config.yml."
    )
    GUARDRAILS_ENABLED: bool = Field(
        default=True,
        description="Global master switch to enforce input jailbreak filtering and output PII redaction."
    )

    # =========================================================================
    # 6. LangGraph Agent Runtime Controls
    # =========================================================================
    MAX_RESEARCH_STEPS: int = Field(
        default=4,
        description="Maximum tool execution loop iterations to prevent runaway agent costs."
    )
    ENABLE_LIVE_WEB_SEARCH: bool = Field(
        default=True,
        description="Enables live internet querying via DuckDuckGo search tool."
    )
    SIMULATION_FALLBACK: bool = Field(
        default=True,
        description="Enables deterministic fallback synthesis if external tool networks time out."
    )


# Exported settings singleton
settings = Settings()
