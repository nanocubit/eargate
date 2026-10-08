"""Тесты конфигурации проекта."""
from siren_config import (
    N_FEATURES, N_MFCC, SIREN_CLASSES, CHUNK_SAMPLES, SAMPLE_RATE,
    CHUNK_DURATION, HOP_SAMPLES, HOP_DURATION,
)


def test_features_count_matches_mfcc_blocks():
    """48 признаков = 13 MFCC + 13 std + 13 delta + 7 прочих."""
    assert N_MFCC == 13
    assert N_FEATURES == 48


def test_siren_classes_mapping():
    assert SIREN_CLASSES == {
        0: "fire_truck",
        1: "ambulance",
        2: "police_car",
        3: "none",
    }


def test_audio_constants_consistent():
    assert CHUNK_SAMPLES == int(SAMPLE_RATE * CHUNK_DURATION)
    assert HOP_SAMPLES == int(SAMPLE_RATE * HOP_DURATION)
