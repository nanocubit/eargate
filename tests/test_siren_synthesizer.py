"""Тесты генераторов синтетических сигналов и аугментаций."""
import numpy as np

from siren_synthesizer import (
    generate_wail, generate_yelp, generate_hilo, generate_airhorn,
    add_doppler, add_reverb, add_distortion, add_noise,
    add_traffic_noise, augment,
)
from siren_config import SAMPLE_RATE


def _assert_same_length(signal: np.ndarray, original: np.ndarray):
    assert signal.shape == original.shape


def _assert_finite(signal: np.ndarray):
    assert np.all(np.isfinite(signal))


def test_generate_wail_length_and_finite():
    signal = generate_wail(200, 800, 3.0, 3.0)
    assert len(signal) == int(SAMPLE_RATE * 3.0)
    _assert_finite(signal)


def test_generate_yelp_length_and_finite():
    signal = generate_yelp(600, 1500, 0.5, 3.0)
    assert len(signal) == int(SAMPLE_RATE * 3.0)
    _assert_finite(signal)


def test_generate_hilo_length_and_finite():
    signal = generate_hilo(600, 1200, 0.5, 3.0)
    assert len(signal) == int(SAMPLE_RATE * 3.0)
    _assert_finite(signal)


def test_generate_airhorn_envelope():
    """Airhorn затухает на краях (env=0 в начале и конце)."""
    signal = generate_airhorn(300, 1.0)
    assert len(signal) == SAMPLE_RATE
    _assert_finite(signal)
    assert abs(signal[0]) < 1e-6
    assert abs(signal[-1]) < 1e-6


def test_add_doppler_preserves_length():
    signal = generate_wail(200, 800, 3.0, 3.0)
    shifted = add_doppler(signal, speed_mps=15)
    _assert_same_length(shifted, signal)
    _assert_finite(shifted)


def test_add_reverb_preserves_length():
    signal = generate_wail(200, 800, 3.0, 3.0)
    reverbed = add_reverb(signal, rt60=0.3)
    _assert_same_length(reverbed, signal)
    _assert_finite(reverbed)


def test_add_distortion_bounds_output():
    """tanh ограничивает амплитуду [-1, 1]."""
    signal = generate_wail(200, 800, 3.0, 3.0) * 10
    distorted = add_distortion(signal, drive=2.0)
    assert np.max(np.abs(distorted)) <= 1.0
    _assert_finite(distorted)


def test_add_noise_increases_energy():
    signal = generate_wail(200, 800, 3.0, 3.0)
    noisy = add_noise(signal, snr_db=20)
    _assert_same_length(noisy, signal)
    _assert_finite(noisy)
    assert np.mean(noisy ** 2) > np.mean(signal ** 2)


def test_add_traffic_noise_changes_signal():
    signal = generate_wail(200, 800, 3.0, 3.0)
    noisy = add_traffic_noise(signal, snr_db=10)
    _assert_same_length(noisy, signal)
    _assert_finite(noisy)
    assert not np.allclose(noisy, signal)


def test_augment_returns_finite_float32():
    np.random.seed(42)
    signal = generate_wail(200, 800, 3.0, 3.0)
    augmented = augment(signal)
    assert augmented.dtype == np.float32
    _assert_finite(augmented)
