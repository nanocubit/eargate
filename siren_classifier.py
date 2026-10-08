"""SVM-классификатор сирен с калибровкой вероятностей."""
import logging

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from siren_config import (
    CONFIDENCE_THRESHOLD,
    MODEL_PATH,
    N_FEATURES,
    SCALER_PATH,
    SIREN_CLASSES,
)
from siren_features import extract_features

logger = logging.getLogger("classifier")


class SirenClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None

    def load(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        logger.info("Модель загружена (%d признаков)", N_FEATURES)

    def save(self):
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

    def predict_proba(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Полный вектор вероятностей [fire, ambulance, police, none]."""
        features, in_band = extract_features(audio, sr)
        if not in_band:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        return self.model.predict_proba(features_scaled)[0]

    def predict(self, audio: np.ndarray, sr: int) -> tuple[str, float]:
        proba = self.predict_proba(audio, sr)
        best_idx = int(np.argmax(proba))
        best_class = SIREN_CLASSES[best_idx]
        confidence = float(proba[best_idx])
        if confidence < CONFIDENCE_THRESHOLD:
            return "none", confidence
        return best_class, confidence

    def train(self, X: np.ndarray, y: np.ndarray):
        if X.shape[1] != N_FEATURES:
            raise ValueError(
                f"X имеет {X.shape[1]} признаков, ожидалось {N_FEATURES}"
            )

        min_class_size = int(np.bincount(y).min())
        logger.info("Минимальный размер класса: %d", min_class_size)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        base = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)

        if min_class_size >= 3:
            self.model = CalibratedClassifierCV(base, cv=3)
        elif min_class_size == 2:
            self.model = CalibratedClassifierCV(base, cv=2)
        else:
            logger.warning(
                "Класс с 1 образцом — калибровка отключена, "
                "использую SVC(probability=True)"
            )
            self.model = SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=42,
            )

        self.model.fit(X_train_s, y_train)

        y_pred = self.model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        labels = sorted(set(y))
        target_names = [SIREN_CLASSES[i] for i in labels]
        logger.info("Точность: %.4f", acc)
        logger.info(
            "\n%s",
            classification_report(y_test, y_pred, target_names=target_names),
        )

        self.save()
