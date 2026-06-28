"""
Code Smell Detector
===================
Detects long-method code smell in Python functions using a trained ML model.

Uses the same feature engineering and SVM model approach from
model_synthetic.ipynb — 12 safe features with synthetic augmentation.

The model is trained by `train_smell_model.py` which replicates the
notebook's methodology: Gaussian noise + SMOTE augmentation, then SVM
classification with ~95% CV accuracy.
"""

import re
import os
import sys
import warnings
import numpy as np
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent.parent  # app/
MODELS_DIR   = SCRIPT_DIR / "models"
SMELL_MODEL  = MODELS_DIR / "smell_svm.pkl"
SMELL_SCALER = MODELS_DIR / "smell_scaler.pkl"

# ── Safe features (must match model_synthetic.ipynb exactly) ─────────────────
SAFE_FEATURES = [
    "num_lines",
    "code_avg_line_len",
    "code_max_line_len",
    "code_indent_depth",
    "code_has_loop",
    "code_has_conditional",
    "code_has_return",
    "code_has_try",
    "code_num_returns",
    "code_num_ifs",
    "code_num_loops",
    "code_comment_lines",
]


def extract_code_features(code_str: str) -> dict:
    """
    Extract the 12 safe features from a Python function code string.
    Mirrors the logic in model.ipynb and model_synthetic.ipynb.
    """
    code = str(code_str)
    lines = code.replace("\\n", "\n").split("\n")
    nonempty = [l for l in lines if l.strip()]

    num_lines = float(len(nonempty))
    line_lengths = [len(l) for l in nonempty]

    features = {
        "num_lines":             num_lines,
        "code_avg_line_len":     float(np.mean(line_lengths)) if line_lengths else 0.0,
        "code_max_line_len":     float(max(line_lengths)) if line_lengths else 0.0,
        "code_indent_depth":     float(max(len(l) - len(l.lstrip()) for l in nonempty)) if nonempty else 0.0,
        "code_has_loop":         int(any(kw in code for kw in ["for ", "while "])),
        "code_has_conditional":  int(any(kw in code for kw in ["if ", "elif ", "else:"])),
        "code_has_return":       int("return" in code),
        "code_has_try":          int("try:" in code or "except" in code),
        "code_num_returns":      float(code.count("return")),
        "code_num_ifs":          float(code.count("if ")),
        "code_num_loops":        float(code.count("for ") + code.count("while ")),
        "code_comment_lines":    float(sum(1 for l in lines if l.strip().startswith("#"))),
    }
    return features


def extract_features_vector(code_str: str) -> np.ndarray:
    """Return a (1, 12) numpy array for model inference."""
    feats = extract_code_features(code_str)
    return np.array([[feats[f] for f in SAFE_FEATURES]], dtype=float)


# ── Global model cache ───────────────────────────────────────────────────────
_model = None
_scaler = None


def _load_model():
    global _model, _scaler
    if _model is not None:
        return _model, _scaler

    if not SMELL_MODEL.exists() or not SMELL_SCALER.exists():
        print("[code_smell_detector] Trained model not found. Run 'python src/train_smell_model.py' first.",
              file=sys.stderr)
        sys.exit(1)

    _model = joblib.load(str(SMELL_MODEL))
    _scaler = joblib.load(str(SMELL_SCALER))
    return _model, _scaler


from typing import TypedDict

LONG_METHOD_THRESHOLD = 20


class SmellPrediction(TypedDict):
    is_bad_smell: bool
    probability: float
    num_lines: int
    is_long_method: bool


def _load_resources() -> tuple:
    """Load the trained model and its scaler."""
    return _load_model()


def _extract_feature_vector(code: str):
    """Extract the feature vector from source code."""
    return extract_features_vector(code)


def _count_nonempty_lines(code: str) -> int:
    """Count non‑empty lines in the provided source code."""
    lines = code.replace("\\n", "\n").split("\n")
    return sum(1 for line in lines if line.strip())


def _build_result(prediction: int, probability: float, line_count: int) -> SmellPrediction:
    """Construct the result dictionary from prediction data."""
    return {
        "is_bad_smell": prediction == 1,
        "probability": probability,
        "num_lines": line_count,
        "is_long_method": line_count > LONG_METHOD_THRESHOLD,
    }


def predict_smell(code_str: str) -> SmellPrediction:
    """Predict whether a Python function exhibits the long‑method code smell.

    Args:
        code_str: Source code of the function to analyze.

    Returns:
        A dictionary with the following keys:
            is_bad_smell: True if the model predicts a long‑method smell.
            probability: Model's probability for the positive class (0‑1).
            num_lines: Number of non‑empty lines in the function.
            is_long_method: Heuristic flag indicating if the function exceeds
                the long‑method threshold.

    Raises:
        Any exception raised by the underlying model or feature extraction.
    """
    model, scaler = _load_resources()
    feature_vector = _extract_feature_vector(code_str)
    scaled_vector = scaler.transform(feature_vector)

    probability = float(model.predict_proba(scaled_vector)[0, 1])
    prediction = int(model.predict(scaled_vector)[0])

    line_count = _count_nonempty_lines(code_str)

    return _build_result(prediction, probability, line_count)


from typing import List, Dict, Any
import numpy as np

LONG_METHOD_THRESHOLD = 20


def _prepare_features(codes: List[str]) -> np.ndarray:
    """Extract and stack feature vectors for a list of code snippets."""
    feature_vectors = [extract_features_vector(code) for code in codes]
    return np.vstack(feature_vectors)


def _predict(
    model,
    scaler,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale features and obtain predictions and probabilities."""
    scaled = scaler.transform(features)
    probabilities = model.predict_proba(scaled)[:, 1]
    predictions = model.predict(scaled)
    return predictions, probabilities


def _build_result(
    code: str,
    prediction: int,
    probability: float,
) -> Dict[str, Any]:
    """Create a result dictionary for a single code snippet."""
    nonempty_lines = [
        line for line in code.replace("\\n", "\n").split("\n") if line.strip()
    ]
    return {
        "is_bad_smell": prediction == 1,
        "probability": float(probability),
        "num_lines": len(nonempty_lines),
        "is_long_method": len(nonempty_lines) > LONG_METHOD_THRESHOLD,
    }


def predict_smell_batch(codes: List[str]) -> List[Dict[str, Any]]:
    """Predict code smell for a batch of code snippets.

    Args:
        codes: A list of source code strings to evaluate.

    Returns:
        A list of dictionaries, each containing:
            - is_bad_smell (bool): Whether the snippet exhibits the smell.
            - probability (float): Model confidence for the positive class.
            - num_lines (int): Number of non‑empty lines in the snippet.
            - is_long_method (bool): True if the snippet exceeds
              LONG_METHOD_THRESHOLD lines.

    Raises:
        Any exception raised by the underlying model or scaler loading.
    """
    model, scaler = _load_model()
    features = _prepare_features(codes)
    predictions, probabilities = _predict(model, scaler, features)

    results = [
        _build_result(code, pred, prob)
        for code, pred, prob in zip(codes, predictions, probabilities)
    ]
    return results
