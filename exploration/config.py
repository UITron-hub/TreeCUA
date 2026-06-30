# openrlhf/agent/exploration/config.py
"""
Unified configuration for exploration agents and post-processing.

All sensitive values (API keys, tokens, internal URLs) are loaded from
environment variables. Nothing is hardcoded.

Usage:
    from exploration.config import get_api_config, get_path_config

    api_cfg = get_api_config()
    path_cfg = get_path_config()
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# --- Environment variable names ---
ENV_LLM_API_URL = "LLM_API_URL"
ENV_LLM_API_KEY = "LLM_API_KEY"
ENV_LLM_MODEL = "LLM_MODEL"
ENV_LLM_MAX_TOKENS = "LLM_MAX_TOKENS"

ENV_VERIFIER_API_URL = "VERIFIER_API_URL"
ENV_VERIFIER_API_KEY = "VERIFIER_API_KEY"
ENV_VERIFIER_MODEL = "VERIFIER_MODEL"
ENV_VERIFIER_MAX_TOKENS = "VERIFIER_MAX_TOKENS"

ENV_POSTPROCESS_API_URL = "POSTPROCESS_API_URL"
ENV_POSTPROCESS_API_KEY = "POSTPROCESS_API_KEY"
ENV_POSTPROCESS_MODEL = "POSTPROCESS_MODEL"

ENV_WORLD_KNOWLEDGE_PATH = "WORLD_KNOWLEDGE_PATH"
ENV_RESOURCE_DIR = "RESOURCE_DIR"


@dataclass
class ApiConfig:
    """Configuration for an LLM API endpoint."""
    api_url: str
    api_key: str
    model: str
    max_tokens: int = 4096


@dataclass
class PathConfig:
    """Configuration for data/resource paths."""
    world_knowledge_path: str = ""
    resource_dir: str = ""


def _get_env(key: str, default: str = "") -> str:
    """Get an environment variable, raising an error if missing and no default."""
    value = os.environ.get(key, default)
    return value


def get_api_config(
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> ApiConfig:
    """
    Get API configuration for the main exploration LLM.

    Precedence: explicit arguments > environment variables.
    """
    return ApiConfig(
        api_url=api_url or _get_env(ENV_LLM_API_URL),
        api_key=api_key or _get_env(ENV_LLM_API_KEY),
        model=model or _get_env(ENV_LLM_MODEL),
        max_tokens=max_tokens or int(_get_env(ENV_LLM_MAX_TOKENS, "4096")),
    )


def get_verifier_config(
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> ApiConfig:
    """
    Get API configuration for the verification agent (OpenAI-compatible format).

    Does NOT fall back to LLM_API_* because those are Anthropic-native format
    and are incompatible with the OpenAI format used by the verifier.
    """
    return ApiConfig(
        api_url=api_url or _get_env(ENV_VERIFIER_API_URL),
        api_key=api_key or _get_env(ENV_VERIFIER_API_KEY),
        model=model or _get_env(ENV_VERIFIER_MODEL, "gpt-4o-mini"),
        max_tokens=max_tokens or int(_get_env(ENV_VERIFIER_MAX_TOKENS, "512")),
    )


def get_postprocess_config(
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> ApiConfig:
    """
    Get API configuration for post-processing agents: trajectory scoring,
    overall summarization, stage breakdown, and reason synthesis
    (OpenAI-compatible format).

    Does NOT fall back to LLM_API_* because those are Anthropic-native format
    and are incompatible with the OpenAI format used by post-processing.
    """
    return ApiConfig(
        api_url=api_url or _get_env(ENV_POSTPROCESS_API_URL),
        api_key=api_key or _get_env(ENV_POSTPROCESS_API_KEY),
        model=model or _get_env(ENV_POSTPROCESS_MODEL, "gpt-4o-mini"),
        max_tokens=1024,
    )


def get_path_config(
    world_knowledge_path: Optional[str] = None,
    resource_dir: Optional[str] = None,
) -> PathConfig:
    """
    Get path configuration for data resources.
    """
    return PathConfig(
        world_knowledge_path=world_knowledge_path or _get_env(ENV_WORLD_KNOWLEDGE_PATH),
        resource_dir=resource_dir or _get_env(ENV_RESOURCE_DIR),
    )
