from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PATHS = (
    ROOT / "local" / "zh-cn" / "nodes.json",
)


def _load_locale() -> dict[str, Any]:
    for path in LOCAL_PATHS:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


LOCALE = _load_locale()


def category(name: str, fallback: str) -> str:
    return LOCALE.get("category_names", {}).get(name, fallback)


def display_name(class_name: str, fallback: str) -> str:
    return LOCALE.get("node_display_names", {}).get(class_name, fallback)


def description(class_name: str, fallback: str) -> str:
    return LOCALE.get("descriptions", {}).get(class_name, fallback)


def return_names(class_name: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    names = LOCALE.get("return_names", {}).get(class_name)
    if isinstance(names, list) and len(names) == len(fallback):
        return tuple(str(name) for name in names)
    return fallback


def ui(section: str, name: str, fallback: str | None = None) -> dict[str, str]:
    all_ui = LOCALE.get("ui", {})
    item = all_ui.get(section, {}).get(name, {}) or all_ui.get("common", {}).get(name, {})
    display = item.get("display_name") or item.get("name") or fallback or name
    tooltip = item.get("tooltip", "")
    return {"display_name": display, "tooltip": tooltip}
