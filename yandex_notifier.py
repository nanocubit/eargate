"""Отправка событий в Яндекс + локальный импульс реле.

Критический путь (реле) выполняется локально и синхронно.
Некритический путь (Яндекс) обёрнут в try/except.
"""
import logging
import threading
import time

import requests

import siren_config as cfg
from relay_control import pulse as relay_pulse
from siren_config import (
    DEVICES,
    OAUTH_TOKEN,
    RELAY_PULSE_SECONDS,
    SENSOR_RESET_DELAY,
    YANDEX_NOTIFY_URL,
)

logger = logging.getLogger("notifier")


def _send_callback(devices_payload: list[dict]):
    if cfg.USER_ID is None:
        logger.error("USER_ID не установлен — событие не отправлено")
        return

    body = {
        "ts": time.time(),
        "payload": {
            "user_id": cfg.USER_ID,
            "devices": devices_payload,
        },
    }

    try:
        resp = requests.post(
            YANDEX_NOTIFY_URL,
            headers={
                "Authorization": f"OAuth {OAUTH_TOKEN}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=10,
        )
        if resp.status_code != 202:
            logger.error("Яндекс ответил %d: %s", resp.status_code, resp.text)
    except requests.RequestException as e:
        logger.error("Ошибка сети при отправке в Яндекс: %s", e)


def _send_motion(device_id: str, value: str):
    _send_callback([{
        "id": device_id,
        "properties": [{
            "type": "devices.properties.event",
            "state": {"instance": "motion", "value": value},
        }],
    }])


def _reset_sensor_later(device_id: str):
    threading.Timer(
        SENSOR_RESET_DELAY,
        _send_motion,
        args=(device_id, "not_detected"),
    ).start()


def send_siren_event(siren_class: str, direction: str, confidence: float):
    """
    Критический путь: реле щёлкает ЛОКАЛЬНО.
    Некритический путь: оповещение Яндекса — в try/except.
    """
    device_key = f"{siren_class}_{direction}"
    device = DEVICES.get(device_key)
    if not device:
        logger.error("Нет устройства для ключа %s", device_key)
        return

    # Критический путь — реле
    relay_pulse(duration=RELAY_PULSE_SECONDS)

    # Некритический путь — Яндекс
    try:
        _send_callback([{
            "id": device["id"],
            "properties": [{
                "type": "devices.properties.event",
                "state": {"instance": "motion", "value": "detected"},
            }],
        }])
        _reset_sensor_later(device["id"])
    except Exception:
        logger.exception("Ошибка оповещения (реле уже сработало)")

    logger.info("→ %s | conf=%.2f", device["name"], confidence)
