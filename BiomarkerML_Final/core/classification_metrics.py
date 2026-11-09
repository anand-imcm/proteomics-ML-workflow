from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix


def per_class_sensitivity_specificity_from_confusion_matrix(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-vs-rest metrics, using NaN for undefined class denominators."""
    cm = np.asarray(matrix)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        return np.array([], dtype=float), np.array([], dtype=float)

    sensitivities = np.full(cm.shape[0], np.nan, dtype=float)
    specificities = np.full(cm.shape[0], np.nan, dtype=float)
    total = float(cm.sum())
    for index in range(cm.shape[0]):
        tp = float(cm[index, index])
        fn = float(cm[index, :].sum() - tp)
        fp = float(cm[:, index].sum() - tp)
        tn = total - tp - fn - fp
        if tp + fn:
            sensitivities[index] = tp / (tp + fn)
        if tn + fp:
            specificities[index] = tn / (tn + fp)
    return sensitivities, specificities


def sensitivity_specificity_from_confusion_matrix(matrix: np.ndarray) -> tuple[float, float]:
    """Return binary or macro one-vs-rest sensitivity and specificity."""
    cm = np.asarray(matrix)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.shape[0] < 2:
        return float("nan"), float("nan")

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity_denominator = tp + fn
        specificity_denominator = tn + fp
        sensitivity = tp / sensitivity_denominator if sensitivity_denominator else np.nan
        specificity = tn / specificity_denominator if specificity_denominator else np.nan
        return float(sensitivity), float(specificity)

    sensitivities, specificities = per_class_sensitivity_specificity_from_confusion_matrix(cm)
    defined_sensitivities = sensitivities[~np.isnan(sensitivities)]
    defined_specificities = specificities[~np.isnan(specificities)]
    sensitivity = np.mean(defined_sensitivities) if defined_sensitivities.size else np.nan
    specificity = np.mean(defined_specificities) if defined_specificities.size else np.nan
    return float(sensitivity), float(specificity)


def classification_sensitivity_specificity(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    labels: Sequence[object],
) -> tuple[float, float]:
    """Compute metrics using an explicit, stable class order."""
    ordered_labels = list(labels)
    if len(ordered_labels) < 2:
        return float("nan"), float("nan")
    matrix = confusion_matrix(y_true, y_pred, labels=ordered_labels)
    return sensitivity_specificity_from_confusion_matrix(matrix)
