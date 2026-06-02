import os
from pathlib import Path

import yaml

_config: dict | None = None


def _find_config_path() -> Path:
    env_path = os.environ.get("M3EXAM_CONFIG")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    candidates = [
        Path(__file__).parent / "config.yaml",
        Path(__file__).parent.parent / "config" / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "config.yaml not found. Searched:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\nAlternatively, set the M3EXAM_CONFIG environment variable."
    )


def get_config() -> dict:
    global _config
    if _config is None:
        path = _find_config_path()
        with open(path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f) or {}
    return _config


def cfg(*keys, default=None):
    d = get_config()
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def reload_config() -> dict:
    global _config
    _config = None
    return get_config()
