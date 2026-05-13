"""
Config and API key management.
Stored in ~/.portfolio_tracker/config.json
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR    = Path.home() / ".portfolio_tracker"
CONFIG_FILE = DATA_DIR / "config.json"

PROVIDERS = ("polygon", "finnhub")
PLAID_KEYS = ("plaid-client-id", "plaid-secret", "plaid-env")


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_config(cfg: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def get_key(name: str) -> str | None:
    return load_config().get(name)


def set_key(name: str, value: str) -> None:
    cfg = load_config()
    cfg[name] = value
    save_config(cfg)


def configured_providers() -> list[str]:
    """Return ordered list of providers that have API keys configured."""
    cfg = load_config()
    available = []
    if cfg.get("polygon"):
        available.append("polygon")
    if cfg.get("finnhub"):
        available.append("finnhub")
    available.append("yfinance")   # always available as final fallback
    return available
