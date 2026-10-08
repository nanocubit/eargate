"""Сбор реального датасета: скачивание с YouTube + нарезка на 3-секундные слайсы.

Запуск:
  1. Заполнить YOUTUBE_URLS ниже
  2. python scripts/collect_dataset.py
  3. Полученные wav попадут в dataset/raw/<class>/
  4. Затем build_features() → X, y для дообучения

Требования: yt-dlp, ffmpeg в PATH.
"""
import logging
import os
import subprocess

import librosa
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset")

OUTPUT_DIR = "dataset/raw"
SLICE_DURATION = 3.0
SR = 22050
MIN_RMS = 0.005

YOUTUBE_URLS = {
    "fire_truck": [
        # "https://youtube.com/watch?v=...",
    ],
    "ambulance": [],
    "police_car": [],
    "none": [],
}


def download_and_slice(url: str, label: str, idx: int) -> int:
    """Скачивает аудио с YouTube, нарезает на 3-секундные куски."""
    out_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out_dir, exist_ok=True)

    tmp_webm = os.path.join(OUTPUT_DIR, f"tmp_{label}_{idx}.webm")
    tmp_wav = tmp_webm.replace(".webm", ".wav")

    try:
        subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "wav",
             "-o", tmp_webm, url],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-i", tmp_webm, "-ar", str(SR),
             "-ac", "1", "-y", tmp_wav],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("Ошибка скачивания %s: %s", url, e)
        return 0

    audio, _ = librosa.load(tmp_wav, sr=SR, mono=True)
    slice_samples = int(SLICE_DURATION * SR)
    n_slices = len(audio) // slice_samples
    saved = 0

    for s in range(n_slices):
        chunk = audio[s * slice_samples:(s + 1) * slice_samples]
        if np.sqrt(np.mean(chunk ** 2)) < MIN_RMS:
            continue
        out_path = os.path.join(out_dir, f"{label}_{idx}_{s:04d}.wav")
        sf.write(out_path, chunk, SR, subtype="PCM_16")
        saved += 1

    for f in (tmp_webm, tmp_wav):
        if os.path.exists(f):
            os.remove(f)

    logger.info("%s [%d]: сохранено %d слайсов", label, idx, saved)
    return saved


def download_all():
    total = 0
    for label, urls in YOUTUBE_URLS.items():
        for i, url in enumerate(urls):
            total += download_and_slice(url, label, i)
    logger.info("Всего сохранено: %d слайсов", total)


def build_features():
    """Собирает X, y из dataset/raw/. Для дообучения модели."""
    from siren_features import extract_features
    from siren_config import SIREN_CLASSES

    X, y = [], []
    label_map = {v: k for k, v in SIREN_CLASSES.items()}

    for label_name, label_idx in label_map.items():
        dir_path = os.path.join(OUTPUT_DIR, label_name)
        if not os.path.isdir(dir_path):
            continue
        files = [f for f in os.listdir(dir_path) if f.endswith(".wav")]
        logger.info("%s: %d файлов", label_name, len(files))

        for f in files:
            audio, _ = librosa.load(
                os.path.join(dir_path, f), sr=SR, mono=True,
            )
            feats, _ = extract_features(audio)
            X.append(feats)
            y.append(label_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    logger.info("Итого: %d образцов", X.shape[0])
    return X, y


if __name__ == "__main__":
    download_all()
    X, y = build_features()
    if X.size > 0:
        os.makedirs("models", exist_ok=True)
        np.save("models/X_real.npy", X)
        np.save("models/y_real.npy", y)
        logger.info("Сохранено в models/X_real.npy, y_real.npy")
        logger.info(
            "Для дообучения: "
            "X = np.vstack([np.load('models/X_siren.npy'), X_real]); "
            "y = np.concatenate([np.load('models/y_siren.npy'), y_real]); "
            "clf.train(X, y)"
        )
