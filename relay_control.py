"""Неблокирующее управление реле через GPIO.

Замыкание — мгновенно под Lock. Размыкание — через Timer в отдельном потоке.
Это позволяет вызывать pulse() из аудио-колбэка sounddevice без переполнения.
"""
import logging
import threading

logger = logging.getLogger("relay")

_RELAY_PIN = 17
_relay_lock = threading.Lock()
_release_timer: threading.Timer | None = None
_impulse_gen = 0
_HAS_GPIO = False

try:
    from RPi import GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(_RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH)
    _HAS_GPIO = True
    logger.info("GPIO инициализирован: пин %d (active-low)", _RELAY_PIN)
except (ImportError, RuntimeError) as e:
    logger.warning("GPIO недоступен (%s) — реле в режиме заглушки", e)


def pulse(duration: float = 2.0):
    """
    Неблокирующий импульс реле.
    Если предыдущий импульс активен — продлевается счётчик генерации,
    старый Timer размыкает только если его поколение актуально.
    """
    global _release_timer, _impulse_gen

    if not _HAS_GPIO:
        logger.info("[stub] реле: импульс %.1f сек", duration)
        return

    with _relay_lock:
        _impulse_gen += 1
        gen = _impulse_gen
        GPIO.output(_RELAY_PIN, GPIO.LOW)

    if _release_timer and _release_timer.is_alive():
        _release_timer.cancel()

    def _release():
        with _relay_lock:
            if _impulse_gen == gen:
                GPIO.output(_RELAY_PIN, GPIO.HIGH)

    _release_timer = threading.Timer(duration, _release)
    _release_timer.daemon = True
    _release_timer.start()
    logger.info("реле: импульс %.1f сек (gen=%d)", duration, gen)


def cleanup():
    if _release_timer and _release_timer.is_alive():
        _release_timer.cancel()
    if _HAS_GPIO:
        with _relay_lock:
            GPIO.output(_RELAY_PIN, GPIO.HIGH)
        GPIO.cleanup()
        logger.info("GPIO очищен")
