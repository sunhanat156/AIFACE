"""Central config.json reader/writer shared by every module.

Keeping this in one place means the Settings page, the camera and the
notifier all see the same values without each re-implementing file IO.
"""
import json
import os
import threading

from backend.paths import APP_DIR

CONFIG_PATH = os.path.join(APP_DIR, "config.json")

DEFAULTS = {
    "line_token": "",
    "tolerance": 0.42,
    "log_cooldown": 10,
    "unknown_cooldown": 30,
    "camera_index": 0,
    "recognize_every_n_frames": 5,
}

_lock = threading.Lock()


def load_config():
    """Return the saved config merged over the defaults."""
    config = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                config.update(saved)
        except Exception as e:
            print(f"[WARN] config.json unreadable ({e}); using defaults.")
    return config


def save_config(updates):
    """Merge `updates` into config.json and return the full new config."""
    with _lock:
        config = load_config()
        config.update(updates or {})
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        return config


def get(key, default=None):
    return load_config().get(key, DEFAULTS.get(key, default))
