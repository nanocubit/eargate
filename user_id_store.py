"""Персистентное хранение Yandex user_id между рестартами."""
import json
import logging
import pathlib

logger = logging.getLogger("user_store")

_USER_FILE = pathlib.Path("models/user_id.json")


def load_user_id() -> str | None:
    if _USER_FILE.exists():
        try:
            return json.loads(_USER_FILE.read_text()).get("user_id")
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("user_id.json повреждён (%s) — игнорирую", e)
    return None


def save_user_id(user_id: str):
    try:
        _USER_FILE.parent.mkdir(exist_ok=True)
        _USER_FILE.write_text(json.dumps({"user_id": user_id}))
        logger.info("user_id сохранён: %s", user_id)
    except OSError as e:
        logger.error("Не удалось сохранить user_id: %s", e)


def clear_user_id():
    if _USER_FILE.exists():
        _USER_FILE.unlink()
        logger.info("user_id удалён")
