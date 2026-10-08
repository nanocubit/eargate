"""Все настройки EarGate. Значения можно переопределить через env."""
import os

# ─── Аудио ───
SAMPLE_RATE = 22050
CHUNK_DURATION = 3.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)   # 66150
HOP_DURATION = 0.5
HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION)        # 11025

# ─── Энергетический фильтр ───
ENERGY_THRESHOLD = 0.01
SIREN_FREQ_MIN = 150    # ГОСТ Р 50574-2019
SIREN_FREQ_MAX = 2000

# ─── MFCC ───
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH_MFCC = 512
N_MELS = 128
N_FEATURES = 48

# ─── Классификатор ───
MODEL_PATH = "models/siren_classifier.pkl"
SCALER_PATH = "models/siren_scaler.pkl"
CONFIDENCE_THRESHOLD = 0.65

# ─── Защита от ложных срабатываний ───
CONSECUTIVE_HITS = 2         # подряд одинаковых классов
PROBA_HISTORY_LEN = 3        # кадров для скользящего среднего
MIN_SIREN_DURATION = 0.8     # секунд непрерывного сигнала (клаксон = 0.2–0.5)
COOLDOWN_SECONDS = 30        # не повторять оповещение

# ─── Сброс датчика и реле ───
SENSOR_RESET_DELAY = 5       # сек до not_detected
RELAY_PULSE_SECONDS = 2.0    # длительность импульса на реле

# ─── Логи ───
DETECTION_LOG_DIR = "logs/detections"
MAX_LOG_AGE_DAYS = 7
MAX_LOG_SIZE_MB = 500

# ─── Классы ───
SIREN_CLASSES = {
    0: "fire_truck",
    1: "ambulance",
    2: "police_car",
    3: "none",
}
SIREN_LABELS_RU = {
    "fire_truck": "пожарная машина",
    "ambulance": "скорая помощь",
    "police_car": "полиция",
    "none": "—",
}

# ─── Источник аудио ───
AUDIO_SOURCE = os.environ.get("AUDIO_SOURCE", "microphone")
RTSP_AUDIO_URL_ENTRY = os.environ.get("RTSP_AUDIO_URL_ENTRY", "")
RTSP_AUDIO_URL_EXIT = os.environ.get("RTSP_AUDIO_URL_EXIT", "")
MIC_ENTRY_DEVICE_INDEX = int(os.environ.get("MIC_ENTRY_DEVICE_INDEX", "0"))
MIC_EXIT_DEVICE_INDEX = int(os.environ.get("MIC_EXIT_DEVICE_INDEX", "1"))

# ─── Яндекс ───
SKILL_ID = os.environ.get("YANDEX_SKILL_ID", "")
OAUTH_TOKEN = os.environ.get("YANDEX_OAUTH_TOKEN", "")
USER_ID: str | None = None  # восстанавливается из user_id_store

# ─── Виртуальные устройства ───
DEVICES = {
    "fire_truck_entry":   {"id": "sensor-fire-entry",      "name": "Пожарная — въезд"},
    "ambulance_entry":    {"id": "sensor-ambulance-entry", "name": "Скорая — въезд"},
    "police_car_entry":   {"id": "sensor-police-entry",    "name": "Полиция — въезд"},
    "fire_truck_exit":    {"id": "sensor-fire-exit",       "name": "Пожарная — выезд"},
    "ambulance_exit":     {"id": "sensor-ambulance-exit",  "name": "Скорая — выезд"},
    "police_car_exit":    {"id": "sensor-police-exit",     "name": "Полиция — выезд"},
}
BARRIER_RELAY_DEVICE_ID = "relay-barrier"

YANDEX_NOTIFY_URL = (
    f"https://dialogs.yandex.net/api/v1/skills/{SKILL_ID}/callback/state"
)

# ─── Вебхук ───
WEBHOOK_PORT = int(os.environ.get("WEBHOOK_PORT", "8080"))
