"""Optional push notification on a successful match.

NOTE: LINE Notify (notify-api.line.me) was shut down by LINE on 2025-03-31, so
this will not deliver anymore even with a valid old token. The code is kept as
the notification hook — point NOTIFY_URL at LINE Messaging API, Discord or
Telegram when you want notifications back. With an empty token it is a no-op
and costs nothing.
"""
import os

import requests

from backend.core import config_store

NOTIFY_URL = "https://notify-api.line.me/api/notify"


def get_line_token():
    return config_store.get("line_token", "")


def send_line_notify(message, image_path=None):
    token = get_line_token()
    if not token:
        return

    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}

    try:
        if image_path and os.path.exists(image_path):
            # `with` so the snapshot file handle is always released — the old
            # version leaked one per notification.
            with open(image_path, "rb") as img:
                requests.post(NOTIFY_URL, headers=headers, data=data,
                              files={"imageFile": img}, timeout=10)
        else:
            requests.post(NOTIFY_URL, headers=headers, data=data, timeout=10)
    except Exception as e:
        print(f"[ERROR] Notify: {e}")
