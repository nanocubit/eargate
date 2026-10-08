"""Тесты кольцевого буфера детектора."""
import numpy as np

from siren_config import CHUNK_SAMPLES
from siren_detector import CircularBuffer


def test_initial_buffer_is_zeroed():
    buf = CircularBuffer(100)
    assert buf.read().shape == (100,)
    assert np.all(buf.read() == 0)


def test_write_single_chunk():
    buf = CircularBuffer(100)
    chunk = np.arange(10, dtype=np.float32)
    buf.write(chunk)
    data = buf.read()
    # Последние записанные значения — в конце
    np.testing.assert_allclose(data[-10:], chunk)


def test_write_wraps_around():
    buf = CircularBuffer(10)
    buf.write(np.arange(7, dtype=np.float32))          # idx 0..6
    buf.write(np.arange(10, 15, dtype=np.float32))     # idx 7..9, 0..1
    data = buf.read()
    # Старейшие значения (0, 1) перезаписаны новыми (13, 14)
    np.testing.assert_allclose(
        data, [2, 3, 4, 5, 6, 10, 11, 12, 13, 14],
    )


def test_write_chunk_larger_than_buffer():
    """Чанк больше буфера: остаётся только хвост."""
    buf = CircularBuffer(10)
    big = np.arange(25, dtype=np.float32)
    buf.write(big)
    np.testing.assert_allclose(buf.read(), big[-10:])


def test_full_cycle_realistic_size():
    """Реалистичный сценарий: буфер 3 сек, запись чанками по 0.5 сек.

    12 чанков = 6 сек аудио, буфер вмещает ровно 6 чанков (3 сек),
    поэтому в нём остаются значения 6..11, по одному чанку на значение.
    """
    buf = CircularBuffer(CHUNK_SAMPLES)
    hop = CHUNK_SAMPLES // 6
    for i in range(12):
        chunk = np.full(hop, float(i), dtype=np.float32)
        buf.write(chunk)
    data = buf.read()
    expected = np.concatenate([
        np.full(hop, float(i), dtype=np.float32) for i in range(6, 12)
    ])
    np.testing.assert_allclose(data, expected)
