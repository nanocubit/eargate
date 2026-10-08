"""Тесты извлечения признаков из аудио."""
import numpy as np
import pytest

from siren_config import N_FEATURES, SAMPLE_RATE
from siren_features import extract_features


def _tone(freq: float, duration: float = 3.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Синусоидальный тон заданной частоты."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_silence_returns_zero_features_and_out_of_band():
    """Тишина: энергия в полосе сирен < 30% → нулевой вектор, False."""
    audio = (np.random.randn(SAMPLE_RATE * 3) * 0.001).astype(np.float32)
    features, in_band = extract_features(audio)
    assert in_band is False
    assert features.shape == (N_FEATURES,)
    assert np.all(features == 0)


def test_high_frequency_tone_rejected():
    """Тон 5 кГц — вне полосы 150–2000 Гц → ранний выход."""
    audio = _tone(5000)
    features, in_band = extract_features(audio)
    assert in_band is False
    assert np.all(features == 0)


def test_siren_band_tone_returns_full_features():
    """Тон 800 Гц — в полосе сирен → вектор из 48 признаков, True."""
    audio = _tone(800)
    features, in_band = extract_features(audio)
    assert in_band is True
    assert features.shape == (N_FEATURES,)
    assert features.dtype == np.float32
    assert not np.all(features == 0)


def test_features_finite_for_noisy_signal():
    """Признаки конечны (нет NaN/inf) на зашумлённом сигнале."""
    rng = np.random.default_rng(42)
    audio = _tone(1000) + rng.normal(0, 0.01, SAMPLE_RATE * 3)
    features, _ = extract_features(audio.astype(np.float32))
    assert np.all(np.isfinite(features))


def test_features_deterministic():
    """Извлечение признаков детерминировано."""
    audio = _tone(600)
    f1, b1 = extract_features(audio)
    f2, b2 = extract_features(audio)
    assert b1 == b2
    np.testing.assert_allclose(f1, f2)


def test_abs_rms_present_in_features():
    """RMS попадает в признаки (26-й элемент после MFCC-блоков 13+13)."""
    audio = _tone(800) * 0.5
    features, in_band = extract_features(audio)
    assert in_band is True
    expected_rms = float(np.sqrt(np.mean(audio ** 2)))
    # abs_rms — 40-й элемент по индексации (13+13+13+2+1)
    assert features[39] == pytest.approx(expected_rms, rel=1e-4)
