"""Точка входа EarGate."""
import logging
import sys
import threading

from siren_config import (
    AUDIO_SOURCE,
    MIC_ENTRY_DEVICE_INDEX, MIC_EXIT_DEVICE_INDEX,
    RTSP_AUDIO_URL_ENTRY, RTSP_AUDIO_URL_EXIT,
    WEBHOOK_PORT,
)
import siren_config as cfg
from siren_classifier import SirenClassifier
from siren_detector import RealTimeSirenDetector
from yandex_notifier import send_siren_event
from webhook import run_webhook, set_detector_threads
from user_id_store import load_user_id
from relay_control import cleanup as relay_cleanup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("main")


def on_detect(siren_class: str, direction: str, confidence: float):
    send_siren_event(siren_class, direction, confidence)


def train_model():
    from siren_synthesizer import generate_dataset
    logger.info("Генерация синтетического датасета...")
    X, y = generate_dataset(n_samples_per_class=300)
    logger.info("Датасет: %d образцов, %d признаков", X.shape[0], X.shape[1])

    clf = SirenClassifier()
    clf.train(X, y)
    logger.info("Обучение завершено. Модель сохранена в models/")


def _safe_run(target, *args, name=""):
    try:
        target(*args)
    except Exception:
        logger.exception("Поток %s упал", name)


def run_detector():
    classifier = SirenClassifier()
    classifier.load()

    det_entry = RealTimeSirenDetector(classifier, direction="entry")
    det_exit = RealTimeSirenDetector(classifier, direction="exit")

    threads = []

    if AUDIO_SOURCE == "microphone":
        t1 = threading.Thread(
            target=_safe_run,
            args=(det_entry.run_microphone, on_detect, MIC_ENTRY_DEVICE_INDEX),
            kwargs={"name": "entry"},
            daemon=True, name="entry",
        )
        t2 = threading.Thread(
            target=_safe_run,
            args=(det_exit.run_microphone, on_detect, MIC_EXIT_DEVICE_INDEX),
            kwargs={"name": "exit"},
            daemon=True, name="exit",
        )
        threads = [t1, t2]

    elif AUDIO_SOURCE == "rtsp":
        t1 = threading.Thread(
            target=_safe_run,
            args=(det_entry.run_ffmpeg, on_detect, RTSP_AUDIO_URL_ENTRY),
            kwargs={"name": "entry"},
            daemon=True, name="entry",
        )
        threads = [t1]
        if RTSP_AUDIO_URL_EXIT:
            t2 = threading.Thread(
                target=_safe_run,
                args=(det_exit.run_ffmpeg, on_detect, RTSP_AUDIO_URL_EXIT),
                kwargs={"name": "exit"},
                daemon=True, name="exit",
            )
            threads.append(t2)
        else:
            logger.warning(
                "RTSP_AUDIO_URL_EXIT не задан — только въезд"
            )
    else:
        logger.error("Неизвестный AUDIO_SOURCE: %s", AUDIO_SOURCE)
        sys.exit(1)

    for t in threads:
        t.start()

    set_detector_threads(threads)

    for t in threads:
        t.join()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "train":
        train_model()
    elif mode == "run":
        cfg.USER_ID = load_user_id()
        if cfg.USER_ID is None:
            logger.warning(
                "USER_ID не найден. Привяжите навык в «Доме с Алисой» — "
                "оповещения заработают. Реле работает уже сейчас."
            )
        else:
            logger.info("USER_ID восстановлен из хранилища")

        webhook_thread = threading.Thread(
            target=run_webhook, args=(WEBHOOK_PORT,),
            daemon=True, name="webhook",
        )
        webhook_thread.start()
        logger.info("Вебхук слушает :%d", WEBHOOK_PORT)

        try:
            run_detector()
        finally:
            relay_cleanup()
            logger.info("Завершение работы")
    else:
        print("Использование: python main.py [train|run]")
        sys.exit(1)


if __name__ == "__main__":
    main()
