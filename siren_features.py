"""Извлечение 48 признаков из аудиофрагмента. Один STFT, ранний выход."""
import librosa
import numpy as np

from siren_config import (
    HOP_LENGTH_MFCC,
    N_FEATURES,
    N_FFT,
    N_MELS,
    N_MFCC,
    SAMPLE_RATE,
    SIREN_FREQ_MAX,
    SIREN_FREQ_MIN,
)


def extract_features(
    audio: np.ndarray, sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, bool]:
    """
    Возвращает (features[48], in_siren_band).

    Если энергия в полосе 150–2000 Гц < 30% от общей — возвращает
    нулевой вектор и False, не тратя CPU на MFCC.
    """
    audio = audio.astype(np.float32)

    # Абсолютный RMS — ДО нормализации, чтобы не терять громкость
    abs_rms = float(np.sqrt(np.mean(audio ** 2)))

    # Нормализация для тембральных признаков
    norm_audio = audio.copy()
    peak = np.max(np.abs(norm_audio))
    if peak > 1e-10:
        norm_audio = norm_audio / peak

    # Один STFT на все признаки
    mag = np.abs(librosa.stft(
        norm_audio, n_fft=N_FFT, hop_length=HOP_LENGTH_MFCC,
    ))
    power = mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # Полосовой фильтр
    siren_mask = (freqs >= SIREN_FREQ_MIN) & (freqs <= SIREN_FREQ_MAX)
    siren_energy = np.mean(mag[siren_mask, :]) if siren_mask.any() else 0.0
    total_energy = float(np.mean(mag)) + 1e-10
    siren_energy_ratio = float(siren_energy / total_energy)
    in_siren_band = siren_energy_ratio > 0.3

    if not in_siren_band:
        return np.zeros(N_FEATURES, dtype=np.float32), False

    # MFCC через mel-спектрограмму из power
    mel_S = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=N_MELS)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_S), n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(mfcc_delta, axis=1)

    centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))
    modulation_rate = float(np.mean(np.abs(np.diff(centroid[0]))))
    centroid_ac = _autocorr_periodicity(centroid[0])

    zcr = librosa.feature.zero_crossing_rate(
        norm_audio, frame_length=N_FFT, hop_length=HOP_LENGTH_MFCC,
    )
    zcr_mean = float(np.mean(zcr))

    flatness = librosa.feature.spectral_flatness(S=mag)
    flatness_mean = float(np.mean(flatness))

    rolloff = librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.85)
    rolloff_mean = float(np.mean(rolloff))

    features = np.concatenate([
        mfcc_mean,                       # 13
        mfcc_std,                        # 13
        delta_mean,                      # 13
        [centroid_mean, centroid_std],   # 2
        [zcr_mean],                      # 1
        [abs_rms],                       # 1
        [flatness_mean],                 # 1
        [rolloff_mean],                  # 1
        [modulation_rate],               # 1
        [siren_energy_ratio],            # 1
        [centroid_ac],                   # 1
    ]).astype(np.float32)

    if features.shape[0] != N_FEATURES:
        raise ValueError(
            f"Ожидалось {N_FEATURES} признаков, получено {features.shape[0]}"
        )

    return features, True


def _autocorr_periodicity(signal: np.ndarray) -> float:
    """Нормализованная автокорреляция — периодичность wail-цикла."""
    if len(signal) < 4:
        return 0.0
    sig = signal - np.mean(signal)
    ac = np.correlate(sig, sig, mode="full")
    ac = ac[len(ac) // 2:]
    if ac[0] < 1e-10:
        return 0.0
    ac_norm = ac / ac[0]
    for i in range(2, min(len(ac_norm) - 1, 50)):
        if (ac_norm[i] > 0.3
                and ac_norm[i] > ac_norm[i - 1]
                and ac_norm[i] > ac_norm[i + 1]):
            return float(ac_norm[i])
    return 0.0
