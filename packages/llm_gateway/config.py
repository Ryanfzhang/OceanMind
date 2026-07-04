"""
LLM gateway configuration helpers.
"""

import os
import re
from pathlib import Path
from typing import Iterable, Optional


OPENAI_API_KEY_ENV_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
OPENAI_BASE_URL_ENV_VARS = ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL")
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
LEGACY_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"


def _load_runtime_config_value(
    env_var: str,
    project_root: Optional[str] = None,
) -> Optional[str]:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    path = root / ".env"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        match = re.search(
            rf'^{re.escape(env_var)}="?([^"\n]+)"?$',
            text,
            flags=re.MULTILINE,
        )
        if match:
            return match.group(1)
    return os.getenv(env_var) or None


def _is_placeholder_value(value: Optional[str]) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    if not normalized:
        return True
    return normalized in {
        "your_openai_api_key_here",
        "your_openai_compatible_api_key_here",
        "your_api_key_here",
    }


def has_valid_openai_runtime_config(project_root: Optional[str] = None) -> bool:
    value = _load_runtime_config_value("OPENAI_API_KEY", project_root=project_root)
    return isinstance(value, str) and not _is_placeholder_value(value)


def load_config_value(
    env_var: str,
    project_root: Optional[str] = None,
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Load a config value from `.env` or the environment.
    """
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    for filename in (".env",):
        path = root / filename
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        match = re.search(
            rf'^{re.escape(env_var)}="?([^"\n]+)"?$',
            text,
            flags=re.MULTILINE,
        )
        if match:
            return match.group(1)

    value = os.getenv(env_var)
    if value:
        return value

    return default


def load_api_key(
    env_var: str,
    project_root: Optional[str] = None,
) -> str:
    """
    Load an API key from `.env` or the environment.
    """
    value = load_config_value(env_var, project_root=project_root)
    if value:
        return value

    raise ValueError(f"{env_var} was not found in the environment or .env.")


def load_first_config_value(
    env_vars: Iterable[str],
    *,
    project_root: Optional[str] = None,
    default: Optional[str] = None,
) -> Optional[str]:
    for env_var in env_vars:
        value = load_config_value(env_var, project_root=project_root)
        if value:
            return value
    return default


def load_llm_api_key(project_root: Optional[str] = None) -> str:
    openai_key = load_config_value("OPENAI_API_KEY", project_root=project_root)
    if openai_key and not _is_placeholder_value(openai_key):
        return openai_key

    legacy_key = load_config_value("ANTHROPIC_API_KEY", project_root=project_root)
    if legacy_key and not _is_placeholder_value(legacy_key):
        return legacy_key

    joined = ", ".join(OPENAI_API_KEY_ENV_VARS)
    raise ValueError(f"No LLM API key found. Checked: {joined}.")


def load_llm_base_url(
    project_root: Optional[str] = None,
    default: str = DEFAULT_OPENAI_BASE_URL,
) -> str:
    if has_valid_openai_runtime_config(project_root=project_root):
        return load_config_value(
            "OPENAI_BASE_URL",
            project_root=project_root,
            default=default,
        ) or default

    legacy_base_url = load_config_value("ANTHROPIC_BASE_URL", project_root=project_root)
    if legacy_base_url:
        return legacy_base_url

    return load_first_config_value(
        OPENAI_BASE_URL_ENV_VARS,
        project_root=project_root,
        default=default,
    ) or default


def load_model_name(
    role_env_var: Optional[str] = None,
    *,
    project_root: Optional[str] = None,
    default: str = DEFAULT_OPENAI_MODEL,
    legacy_default: Optional[str] = LEGACY_ANTHROPIC_MODEL,
) -> str:
    if has_valid_openai_runtime_config(project_root=project_root):
        env_vars = [role_env_var] if role_env_var else []
        env_vars.append("OPENAI_MODEL")
        return load_first_config_value(env_vars, project_root=project_root, default=default) or default

    if should_use_legacy_anthropic_config(project_root=project_root):
        if role_env_var:
            role_value = load_config_value(role_env_var, project_root=project_root)
            if (
                role_value
                and not _is_placeholder_value(role_value)
                and "claude" in role_value.lower()
            ):
                return role_value
        return legacy_default or default

    if role_env_var:
        role_value = load_config_value(role_env_var, project_root=project_root)
        if role_value and not _is_placeholder_value(role_value):
            return role_value
    openai_model = load_config_value("OPENAI_MODEL", project_root=project_root)
    if openai_model and not _is_placeholder_value(openai_model):
        return openai_model
    return default


def should_use_legacy_anthropic_config(project_root: Optional[str] = None) -> bool:
    if has_valid_openai_runtime_config(project_root=project_root):
        return False
    return any(
        value and not _is_placeholder_value(value)
        for value in (
            _load_runtime_config_value("ANTHROPIC_API_KEY", project_root=project_root),
            _load_runtime_config_value("ANTHROPIC_BASE_URL", project_root=project_root),
        )
    )
