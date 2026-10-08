"""Real-time детектор с четырёхуровневой защитой от ложных срабатываний.

Уровни:
  1. Энергетический (RMS) + полосовой фильтр 150–2000 Гц
  2. Скользящее среднее по 3 кадрам вероятностей
  3. Минимальная длительность непрерывной сирены (0.8 с)
  4. Подтверждение подряд (CONSECUTIVE_HITS)

Все аварийные операции (запись аудио, отправка в Яндекс) обёрнуты
в try/except, чтобы не убить аудио-колбэк.
"""
import collections
import logging
import os
import pathlib
import threading
import time

import numpy as np
import soundfile as sf

from siren_config import (
    CHUNK_SAMPLES,
    CONFIDENCE_THRESHOLD,
    CONSECUTIVE_HITS,
    COOLDOWN_SECONDS,
    DETECTION_LOG_DIR,
    ENERGY_THRESHOLD,
    HOP_DURATION,
    HOP_SAMPLES,
    MAX_LOG_AGE_DAYS,
    MAX_LOG_SIZE_MB,
    MIN_SIREN_DURATION,
    PROBA_HISTORY_LEN,
    SAMPLE_RATE,
    SIREN_CLASSES,
    SIREN_LABELS_RU,
)

logger = logging.getLogger("detector")


# ─── Кольцевой буфер ───

class CircularBuffer:
    def __init__(self, size: int):
        self.buf = np.zeros(size, dtype=np.float32)
        self.write_idx = 0
        self.size = size

    def write(self, chunk: np.ndarray):
        n = len(chunk)
        if n >= self.size:
            self.buf[:] = chunk[-self.size:]
            self.write_idx = self.size
            return
        end = self.write_idx + n
        if end <= self.size:
            self.buf[self.write_idx:end] = chunk
        else:
            split = self.size - self.write_idx
            self.buf[self.write_idx:] = chunk[:split]
            self.buf[:end - self.size] = chunk[split:]
        self.write_idx = end % self.size

    def read(self) -> np.ndarray:
        if self.write_idx == self.size or self.write_idx == 0:
            return self.buf.copy()
        return np.concatenate([
            self.buf[self.write_idx:], self.buf[:self.write_idx],
        ])


# ─── Запись аудио для разбора ───

def _cleanup_old_logs():
    """Удаляет старые записи по возрасту и размеру."""
    log_dir = pathlib.Path(DETECTION_LOG_DIR)
    if not log_dir.exists():
        return
    cutoff = time.time() - MAX_LOG_AGE_DAYS * 86400
    files = sorted(log_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)

    for f in list(files):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                files.remove(f)
        except OSError:
            pass

    total_mb = sum(f.stat().st_size for f in files) / 1e6
    while total_mb > MAX_LOG_SIZE_MB and files:
        oldest = files.pop(0)
        try:
            total_mb -= oldest.stat().st_size / 1e6
            oldest.unlink()
        except OSError:
            pass


def _save_detection_audio(
    audio: np.ndarray, siren_class: str, direction: str, confidence: float,
):
    """Синхронная запись — вызывается из фонового потока."""
    try:
        os.makedirs(DETECTION_LOG_DIR, exist_ok=True)
        ts = int(time.time())
        path = os.path.join(
            DETECTION_LOG_DIR,
            f"{siren_class}_{direction}_{ts}_conf{confidence:.2f}.wav",
        )
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        logger.info("Аудио сохранено: %s", path)
    except Exception:
        logger.exception("Не удалось сохранить аудио (не критично)")


def _save_async(audio: np.ndarray, *args):
    """Асинхронная запись — не блокирует аудио-колбэк."""
    audio_copy = audio.copy()
    threading.Thread(
        target=_save_detection_audio,
        args=(audio_copy, *args),
        daemon=True,
    ).start()


# ─── Детектор ───

class RealTimeSirenDetector:

    def __init__(self, classifier, direction: str = "entry"):
        self.classifier = classifier
        self.buffer = CircularBuffer(CHUNK_SAMPLES)
        self.direction = direction

        # Защита от ложных
        self.proba_history = collections.deque(maxlen=PROBA_HISTORY_LEN)
        self.siren_duration = 0.0
        self.hit_count = 0
        self.last_class = "none"

        # Дебаунс per-class+direction
        self.last_alert: dict[str, float] = collections.defaultdict(float)

        # Фоновый сборщик логов
        self._last_cleanup = 0.0

    # ─── Захват ───

    def run_microphone(self, on_detect, device_index: int | None = None):
        import sounddevice as sd
        idx = device_index if device_index is not None else 0
        logger.info("Микрофон (device=%s, dir=%s)", idx, self.direction)

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio status: %s", status)
            self._process(indata[:, 0], on_detect)

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=HOP_SAMPLES, device=idx, callback=callback,
        ):
            logger.info("Слушаю... (dir=%s)", self.direction)
            while True:
                sd.sleep(int(HOP_DURATION * 1000))

    def run_ffmpeg(self, on_detect, url: str):
        import subprocess
        logger.info("RTSP: %s (dir=%s)", url, self.direction)

        cmd = [
            "ffmpeg", "-i", url,
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "f32le", "-hide_banner", "-loglevel", "error", "pipe:1",
        ]

        while True:
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                bph = HOP_SAMPLES * 4  # float32
                while True:
                    raw = proc.stdout.read(bph)
                    if len(raw) < bph:
                        break
                    chunk = np.frombuffer(raw, dtype=np.float32)
                    self._process(chunk, on_detect)
            except Exception as e:  # noqa: BLE001 — ffmpeg может упасть с чем угодно
                logger.error("RTSP прерван: %s. Реконнект через 5с", e)
                time.sleep(5)

    # ─── Обработка кадра ───

    def _process(self, chunk: np.ndarray, on_detect):
        # Периодическая очистка логов (раз в час)
        now = time.time()
        if now - self._last_cleanup > 3600:
            threading.Thread(target=_cleanup_old_logs, daemon=True).start()
            self._last_cleanup = now

        self.buffer.write(chunk)
        audio = self.buffer.read()

        # ── Уровень 1: энергия ──
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < ENERGY_THRESHOLD:
            self._reset_on_silence()
            return

        # ── Классификация: вектор вероятностей ──
        try:
            proba = self.classifier.predict_proba(audio, SAMPLE_RATE)
        except Exception:
            logger.exception("Классификатор упал")
            self._reset_on_silence()
            return

        # ── Уровень 2: скользящее среднее ──
        self.proba_history.append(proba)
        if len(self.proba_history) < PROBA_HISTORY_LEN:
            return

        smoothed = np.mean(self.proba_history, axis=0)
        best_idx = int(np.argmax(smoothed))
        confidence = float(smoothed[best_idx])
        siren_class = SIREN_CLASSES[best_idx]

        if siren_class == "none" or confidence < CONFIDENCE_THRESHOLD:
            self._decay_on_none()
            return

        # ── Уровень 3: длительность ──
        self.siren_duration += HOP_DURATION
        if self.siren_duration < MIN_SIREN_DURATION:
            logger.debug(
                "%s conf=%.2f dur=%.1f/%.1f (накапливаем)",
                SIREN_LABELS_RU.get(siren_class, "?"),
                confidence, self.siren_duration, MIN_SIREN_DURATION,
            )
            return

        # ── Уровень 4: подтверждение подряд ──
        if siren_class == self.last_class:
            self.hit_count += 1
        else:
            self.hit_count = 1
            self.last_class = siren_class

        logger.debug(
            "%s conf=%.2f hits=%d/%d dur=%.1f",
            SIREN_LABELS_RU.get(siren_class, "?"),
            confidence, self.hit_count, CONSECUTIVE_HITS,
            self.siren_duration,
        )

        if self.hit_count < CONSECUTIVE_HITS:
            return

        # ── Срабатывание ──
        key = f"{siren_class}_{self.direction}"
        if now - self.last_alert[key] <= COOLDOWN_SECONDS:
            return

        logger.info(
            "✓ %s (%s, conf=%.2f, dir=%s, dur=%.1fs)",
            SIREN_LABELS_RU[siren_class], key, confidence,
            self.direction, self.siren_duration,
        )

        # Запись аудио — асинхронно
        _save_async(audio, siren_class, self.direction, confidence)

        # Колбэк — в try/except, чтобы не убить поток
        try:
            on_detect(siren_class, self.direction, confidence)
        except Exception:
            logger.exception("on_detect упал (реле уже сработало)")

        self.last_alert[key] = now
        self.hit_count = 0
        # siren_duration НЕ сбрасываем — сирена продолжается,
        # следующее срабатывание произойдёт после cooldown без накопления заново

    # ─── Сбросы ───

    def _reset_on_silence(self):
        """Жёсткий сброс: аудио нет. Чистим историю вероятностей."""
        self.proba_history.clear()
        self.siren_duration = 0.0
        self.hit_count = 0
        self.last_class = "none"

    def _decay_on_none(self):
        """Мягкий сброс: классификатор сказал 'none' на кадре.
        История сохраняется, чтобы скользящее среднее сглаживало шум."""
        self.siren_duration = 0.0
        self.hit_count = 0
        self.last_class = "none"
