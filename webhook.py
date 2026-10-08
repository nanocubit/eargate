"""Provider API webhook для Яндекс УДЯ + /health."""
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

from siren_config import DEVICES, BARRIER_RELAY_DEVICE_ID, RELAY_PULSE_SECONDS
import siren_config as cfg
from user_id_store import save_user_id, clear_user_id
from relay_control import pulse as relay_pulse

logger = logging.getLogger("webhook")

_detector_threads: list = []


def set_detector_threads(threads: list):
    global _detector_threads
    _detector_threads = threads


def _sensor_spec(device: dict) -> dict:
    return {
        "id": device["id"],
        "name": device["name"],
        "description": "датчик сирены",
        "room": "шлагбаум",
        "type": "devices.types.sensor",
        "capabilities": [],
        "properties": [{
            "type": "devices.properties.event",
            "retrievable": False,
            "reportable": True,
            "parameters": {
                "instance": "motion",
                "events": [
                    {"value": "detected"},
                    {"value": "not_detected"},
                ],
            },
        }],
        "device_info": {"manufacturer": "EarGate", "model": "siren-sensor-v1"},
    }


def _relay_spec() -> dict:
    return {
        "id": BARRIER_RELAY_DEVICE_ID,
        "name": "Шлагбаум",
        "description": "реле управления шлагбаумом",
        "room": "шлагбаум",
        "type": "devices.types.switch",
        "capabilities": [{
            "type": "devices.capabilities.on_off",
            "retrievable": True,
            "reportable": False,
        }],
        "properties": [],
        "device_info": {"manufacturer": "EarGate", "model": "relay-v1"},
    }


class WebhookHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Заменяем стандартный stderr-лог на debug
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_GET(self):
        if self.path == "/v1.0":
            self._send(200, {})
            return
        if self.path == "/health":
            alive = bool(_detector_threads) and all(
                t.is_alive() for t in _detector_threads
            )
            self._send(200 if alive else 503, {
                "status": "ok" if alive else "degraded",
                "threads": [
                    {"name": t.name, "alive": t.is_alive()}
                    for t in _detector_threads
                ],
                "user_id_bound": cfg.USER_ID is not None,
            })
            return

    def do_POST(self):
        body = self._body()
        rid = self.headers.get("X-Request-Id", "unknown")

        # ── Discovery ──
        if self.path == "/v1.0/user/devices":
            user_id = body.get("payload", {}).get("user_id")
            if user_id:
                cfg.USER_ID = user_id
                save_user_id(user_id)
                logger.info("Навык привязан, user_id=%s", user_id)

            devices = [_sensor_spec(d) for d in DEVICES.values()]
            devices.append(_relay_spec())
            self._send(200, {
                "request_id": rid,
                "payload": {
                    "user_id": user_id or cfg.USER_ID or "",
                    "devices": devices,
                },
            })
            return

        # ── Query ──
        if self.path == "/v1.0/user/devices/query":
            devices = []
            for d in body.get("payload", {}).get("devices", []):
                if d["id"] == BARRIER_RELAY_DEVICE_ID:
                    devices.append({
                        "id": d["id"],
                        "capabilities": [{
                            "type": "devices.capabilities.on_off",
                            "state": {"instance": "on", "value": False},
                        }],
                    })
                else:
                    devices.append({"id": d["id"], "properties": []})
            self._send(200, {
                "request_id": rid,
                "payload": {"devices": devices},
            })
            return

        # ── Action ──
        if self.path == "/v1.0/user/devices/action":
            resp = []
            for d in body.get("payload", {}).get("devices", []):
                if d["id"] == BARRIER_RELAY_DEVICE_ID:
                    target_value = False
                    for c in d.get("capabilities", []):
                        if c.get("type") == "devices.capabilities.on_off":
                            target_value = c.get("state", {}).get("value", False)

                    if target_value:
                        relay_pulse(duration=RELAY_PULSE_SECONDS)

                    resp.append({
                        "id": d["id"],
                        "capabilities": [{
                            "type": "devices.capabilities.on_off",
                            "state": {
                                "instance": "on",
                                "value": target_value,
                                "action_result": {"status": "DONE"},
                            },
                        }],
                    })
                else:
                    resp.append({
                        "id": d["id"],
                        "error_code": "NOT_SUPPORTED",
                    })
            self._send(200, {
                "request_id": rid,
                "payload": {"devices": resp},
            })
            return

        # ── Unlink ──
        if self.path == "/v1.0/user/unlink":
            cfg.USER_ID = None
            clear_user_id()
            self._send(200, {"request_id": rid})
            return

        self._send(404, {"error": "unknown endpoint"})


def run_webhook(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), WebhookHandler)
    logger.info("Вебхук :%d", port)
    server.serve_forever()
