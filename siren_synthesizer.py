"""Синтетический датасет с доплером, реверберацией, искажениями, шумом."""
import numpy as np

from siren_config import CHUNK_DURATION, SAMPLE_RATE
from siren_features import extract_features

SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION)


# ─── Базовые генераторы ───

def generate_wail(freq_min, freq_max, cycle_sec, duration_sec, sr=SAMPLE_RATE):
    """Косинусоидальная модуляция частоты."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    phase = (t % cycle_sec) / cycle_sec
    freq = freq_min + (freq_max - freq_min) * 0.5 * (1 - np.cos(2 * np.pi * phase))
    return np.sin(2 * np.pi * np.cumsum(freq) / sr)


def generate_yelp(freq_min, freq_max, cycle_sec, duration_sec, sr=SAMPLE_RATE):
    """Пилообразная модуляция — резкий yelp."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    phase = (t % cycle_sec) / cycle_sec
    freq = freq_min + (freq_max - freq_min) * phase
    return np.sin(2 * np.pi * np.cumsum(freq) / sr)


def generate_hilo(freq_low, freq_high, switch_sec, duration_sec, sr=SAMPLE_RATE):
    """Чередование двух фиксированных тонов."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    freq = np.where((t // switch_sec) % 2 == 0, freq_high, freq_low)
    return np.sin(2 * np.pi * np.cumsum(freq) / sr)


def generate_airhorn(freq, duration_sec, sr=SAMPLE_RATE):
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    signal = (
        np.sin(2 * np.pi * freq * t)
        + 0.5 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.25 * np.sin(2 * np.pi * freq * 3 * t)
    )
    env = np.minimum(t / 0.05, 1.0) * np.minimum((duration_sec - t) / 0.1, 1.0)
    return signal * env


# ─── Аугментации ───

def add_doppler(signal, sr=SAMPLE_RATE, speed_mps=15, min_dist=10.0):
    """Доплер-эффект через интерполяцию: источник проходит мимо."""
    c = 343.0
    n = len(signal)
    t = np.arange(n) / sr
    x = speed_mps * t
    r = np.sqrt(x ** 2 + min_dist ** 2)
    v_r = (x * speed_mps) / r
    factor = c / (c - v_r)
    idx = np.cumsum(factor) - 1
    idx = np.clip(idx, 0, n - 1)
    return np.interp(idx, np.arange(n), signal)


def _pink_noise(n, sr=SAMPLE_RATE):
    """Честный 1/f шум через фильтрацию в частотной области."""
    white = np.random.randn(n)
    f = np.fft.rfftfreq(n, 1 / sr)
    f[0] = 1
    pink = np.fft.irfft(np.fft.rfft(white) / np.sqrt(f), n=n)
    return pink / (np.std(pink) + 1e-10)


def add_traffic_noise(signal, sr=SAMPLE_RATE, snr_db=10):
    """Розовый шум + низкочастотный гул мотора."""
    pink = _pink_noise(len(signal), sr)
    t = np.linspace(0, len(signal) / sr, len(signal))
    engine = np.sin(2 * np.pi * np.random.uniform(80, 120) * t) * 0.3
    noise = pink * 0.7 + engine * 0.3
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(noise_power) / (np.std(noise) + 1e-10)
    return signal + noise


def add_reverb(signal, sr=SAMPLE_RATE, rt60=0.3):
    length = int(sr * rt60)
    impulse = np.zeros(length)
    impulse[0] = 1.0
    impulse *= np.exp(-np.linspace(0, 6, length))
    return np.convolve(signal, impulse, mode="full")[:len(signal)]


def add_distortion(signal, drive=1.5):
    return np.tanh(signal * drive)


def add_noise(signal, snr_db=15):
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    return signal + np.random.randn(len(signal)) * np.sqrt(noise_power)


def augment(signal, sr=SAMPLE_RATE):
    signal = add_distortion(signal, drive=np.random.uniform(1.2, 2.0))
    if np.random.random() > 0.3:
        signal = add_doppler(signal, sr, speed_mps=np.random.uniform(5, 20))
    if np.random.random() > 0.3:
        signal = add_reverb(signal, sr, rt60=np.random.uniform(0.1, 0.5))
    signal = add_traffic_noise(signal, sr, snr_db=np.random.uniform(3, 20))
    signal = add_noise(signal, snr_db=np.random.uniform(5, 25))
    peak = np.max(np.abs(signal))
    if peak > 1.0:
        signal = signal / peak
    return signal.astype(np.float32)


def generate_dataset(n_samples_per_class=300):
    X, y = [], []
    duration = CHUNK_DURATION

    for _ in range(n_samples_per_class):
        # Пожарная: wail + иногда airhorn
        signal = generate_wail(
            np.random.uniform(150, 300), np.random.uniform(600, 1000),
            np.random.uniform(2.5, 4.0), duration,
        )
        if np.random.random() > 0.5:
            horn = generate_airhorn(
                np.random.uniform(200, 400), np.random.uniform(0.5, 1.5),
            )
            offset = np.random.randint(0, len(signal) - len(horn))
            signal[offset:offset + len(horn)] += horn
        signal = augment(signal)
        feats, _ = extract_features(signal)
        X.append(feats); y.append(0)

        # Скорая: hi-lo + иногда wail
        signal = generate_hilo(
            np.random.uniform(500, 700), np.random.uniform(1000, 1500),
            np.random.uniform(0.3, 0.7), duration,
        )
        if np.random.random() > 0.6:
            wail = generate_wail(
                np.random.uniform(400, 600), np.random.uniform(800, 1200),
                np.random.uniform(1.5, 2.5), duration,
            )
            signal = 0.5 * signal + 0.5 * wail
        signal = augment(signal)
        feats, _ = extract_features(signal)
        X.append(feats); y.append(1)

        # Полиция: yelp + иногда phaser
        signal = generate_yelp(
            np.random.uniform(500, 800), np.random.uniform(1200, 2000),
            np.random.uniform(0.3, 0.8), duration,
        )
        if np.random.random() > 0.5:
            phaser = generate_yelp(
                np.random.uniform(800, 1200), np.random.uniform(1500, 2000),
                np.random.uniform(0.1, 0.3), duration,
            )
            signal = 0.6 * signal + 0.4 * phaser
        signal = augment(signal)
        feats, _ = extract_features(signal)
        X.append(feats); y.append(2)

        # Не сирена: бытовой шум + иногда клаксон
        snr = np.random.choice([0.001, 0.005, 0.01, 0.02])
        signal = np.random.randn(SAMPLES_PER_CHUNK).astype(np.float32) * snr
        if np.random.random() > 0.7:
            horn = generate_airhorn(
                np.random.uniform(1800, 3550), np.random.uniform(0.2, 0.5),
            )
            offset = np.random.randint(0, len(signal) - len(horn))
            signal[offset:offset + len(horn)] += horn * 0.3
        signal = augment(signal)
        feats, _ = extract_features(signal)
        X.append(feats); y.append(3)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int32),
    )


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    print("Генерация синтетического датасета...")
    X, y = generate_dataset(n_samples_per_class=300)
    print(f"Датасет: {X.shape[0]} образцов, {X.shape[1]} признаков")
    print(f"Распределение классов: {np.bincount(y)}")
    np.save("models/X_siren.npy", X)
    np.save("models/y_siren.npy", y)
    print("Сохранено в models/")
