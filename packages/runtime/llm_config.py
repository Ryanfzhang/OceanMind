"""Small shared configuration reader for OpenAI-compatible services."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1-mini"
LEGACY_MODEL = "claude-sonnet-4-5-20250929"
AGENT_DEFAULT_BASE_URL = "https://api.deepseek.com"
AGENT_DEFAULT_MODEL = "deepseek-flash"
PLACEHOLDERS = {
    "your_openai_api_key_here",
    "your_openai_compatible_api_key_here",
    "your_api_key_here",
}


def load_config_value(env_var, project_root=None, default=None):
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    path = root / ".env"
    if path.exists():
        match = re.search(
            rf'^{re.escape(env_var)}="?([^"\n]+)"?$',
            path.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        if match:
            return match.group(1)
    return os.getenv(env_var) or default


def _valid(value):
    return bool(value and value.strip() and value.strip().lower() not in PLACEHOLDERS)


def _has_openai_key(project_root):
    return _valid(load_config_value("OPENAI_API_KEY", project_root))


def _has_legacy_config(project_root):
    return any(
        _valid(load_config_value(name, project_root))
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL")
    )


def load_llm_api_key(project_root=None):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        value = load_config_value(name, project_root)
        if _valid(value):
            return value
    raise ValueError("No LLM API key found. Checked: OPENAI_API_KEY, ANTHROPIC_API_KEY.")


def load_llm_base_url(project_root=None, default=DEFAULT_BASE_URL):
    if _has_openai_key(project_root):
        return load_config_value("OPENAI_BASE_URL", project_root, default) or default
    return (
        load_config_value("ANTHROPIC_BASE_URL", project_root)
        or load_config_value("OPENAI_BASE_URL", project_root)
        or default
    )


def load_model_name(
    role_env_var=None,
    *,
    project_root=None,
    default=DEFAULT_MODEL,
    legacy_default=LEGACY_MODEL,
):
    role = load_config_value(role_env_var, project_root) if role_env_var else None
    if _has_openai_key(project_root):
        return role or load_config_value("OPENAI_MODEL", project_root) or default
    if _has_legacy_config(project_root):
        return role if _valid(role) and "claude" in role.lower() else legacy_default or default
    model = load_config_value("OPENAI_MODEL", project_root)
    return role if _valid(role) else model if _valid(model) else default


@dataclass(frozen=True)
class AgentModelConfig:
    base_url: str
    api_key: str = field(repr=False)
    model: str = AGENT_DEFAULT_MODEL


def load_agent_model_config(project_root=None) -> AgentModelConfig:
    """Read the agent's OpenAI-compatible endpoint without changing search settings."""
    api_key = load_config_value("AGENT_API_KEY", project_root) or load_config_value(
        "OPENAI_API_KEY", project_root
    )
    if not _valid(api_key):
        raise ValueError("Agent API key is missing; set AGENT_API_KEY or OPENAI_API_KEY.")

    base_url = load_config_value("AGENT_BASE_URL", project_root) or load_config_value(
        "OPENAI_BASE_URL", project_root, AGENT_DEFAULT_BASE_URL
    )
    model = (load_config_value("AGENT_MODEL", project_root)
             or load_config_value("OPENAI_MODEL", project_root)
             or AGENT_DEFAULT_MODEL)
    return AgentModelConfig(
        base_url=base_url.strip().rstrip("/"),
        api_key=api_key.strip(),
        model=model.strip(),
    )
