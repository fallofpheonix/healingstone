"""Break surface classification model.

Trains a lightweight random-forest classifier to distinguish break surfaces
from exterior surfaces, using the geometric features (curvature, normal
variance, roughness) computed by the feature-extraction stage.

Usage
-----
From code::

    from healingstone.ml_models.surface_model import (
        train_break_surface_classifier,
        predict_break_surface,
    )

    bundle = train_break_surface_classifier(features_list)
    break_mask = predict_break_surface(bundle, features)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)

# Number of estimators for the random forest classifier.
_N_ESTIMATORS = 64
# Random state for reproducibility; controlled externally via seed.
_RANDOM_STATE = 42


@dataclass
class BreakSurfaceClassifier:
    """Fitted break-surface classifier and its feature scaler."""

    classifier: RandomForestClassifier
    scaler: StandardScaler
    feature_names: List[str] = field(default_factory=lambda: ["curvature", "normal_var", "roughness"])
    train_accuracy: float = 0.0


@dataclass
class SurfaceModelBundle:
    """Trained surface classifier together with training diagnostics."""

    model: BreakSurfaceClassifier
    n_train_samples: int
    n_positive: int
    n_negative: int


def _build_feature_matrix(
    curvature: np.ndarray,
    normal_var: np.ndarray,
    roughness: np.ndarray,
) -> np.ndarray:
    """Stack per-point geometric features into an (N, 3) feature matrix."""
    return np.column_stack([
        curvature.ravel().astype(np.float32),
        normal_var.ravel().astype(np.float32),
        roughness.ravel().astype(np.float32),
    ])


def train_break_surface_classifier(
    features_list: list,
    seed: int = _RANDOM_STATE,
    n_estimators: int = _N_ESTIMATORS,
) -> SurfaceModelBundle:
    """Train a random-forest break-surface classifier from a list of FeatureBundle objects.

    Positive class  = points where ``break_mask == True``
    Negative class  = points where ``break_mask == False``

    Parameters
    ----------
    features_list:
        Iterable of ``healingstone.features.FeatureBundle`` instances.
    seed:
        Random seed for the classifier (passed through from the pipeline seed).
    n_estimators:
        Number of trees in the random forest.

    Returns
    -------
    SurfaceModelBundle
        Contains the fitted :class:`BreakSurfaceClassifier` and training stats.
    """
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for fb in features_list:
        X = _build_feature_matrix(fb.curvature, fb.normal_var, fb.roughness)
        y = fb.break_mask.astype(np.int32).ravel()
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise ValueError("No feature bundles provided for training.")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    n_positive = int(np.sum(y_all == 1))
    n_negative = int(np.sum(y_all == 0))
    LOG.info(
        "Training break-surface classifier: %d samples (%d positive, %d negative)",
        len(y_all),
        n_positive,
        n_negative,
    )

    if n_positive == 0 or n_negative == 0:
        LOG.warning(
            "Degenerate training set (all one class). Returning a trivial classifier."
        )
        dominant_class = int(n_positive > n_negative)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        clf.fit(X_scaled, y_all)
        model = BreakSurfaceClassifier(
            classifier=clf,
            scaler=scaler,
            train_accuracy=float(dominant_class),
        )
        return SurfaceModelBundle(
            model=model,
            n_train_samples=len(y_all),
            n_positive=n_positive,
            n_negative=n_negative,
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    clf.fit(X_scaled, y_all)

    preds = clf.predict(X_scaled)
    accuracy = float(np.mean(preds == y_all))
    LOG.info("Break-surface classifier training accuracy: %.4f", accuracy)

    model = BreakSurfaceClassifier(
        classifier=clf,
        scaler=scaler,
        train_accuracy=accuracy,
    )
    return SurfaceModelBundle(
        model=model,
        n_train_samples=len(y_all),
        n_positive=n_positive,
        n_negative=n_negative,
    )


def predict_break_surface(
    bundle: BreakSurfaceClassifier,
    curvature: np.ndarray,
    normal_var: np.ndarray,
    roughness: np.ndarray,
) -> np.ndarray:
    """Predict break-surface membership for a single fragment.

    Parameters
    ----------
    bundle:
        Fitted :class:`BreakSurfaceClassifier`.
    curvature, normal_var, roughness:
        Per-point geometric features (1-D arrays of shape ``(N,)``).

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(N,)`` where ``True`` means break surface.
    """
    X = _build_feature_matrix(curvature, normal_var, roughness)
    X_scaled = bundle.scaler.transform(X)
    preds = bundle.classifier.predict(X_scaled)
    return preds.astype(bool)


def save_surface_model(
    bundle: SurfaceModelBundle,
    output_dir: Path,
) -> Path:
    """Persist the surface model bundle to disk using numpy/joblib serialisation.

    Parameters
    ----------
    bundle:
        Trained :class:`SurfaceModelBundle`.
    output_dir:
        Directory in which to write ``surface_model.npz``.

    Returns
    -------
    Path
        Path to the saved file.
    """
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "surface_model.joblib"
    joblib.dump(bundle, str(model_path))
    LOG.info("Saved surface model to %s", model_path)
    return model_path


def load_surface_model(model_path: Path) -> Optional[SurfaceModelBundle]:
    """Load a previously saved surface model.

    Returns ``None`` if the file does not exist.
    """
    import joblib

    if not model_path.exists():
        LOG.warning("Surface model not found at %s", model_path)
        return None
    bundle: SurfaceModelBundle = joblib.load(str(model_path))
    LOG.info("Loaded surface model from %s", model_path)
    return bundle
