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


from typing import Dict, Union

LONG_METHOD_THRESHOLD = 20


def _count_nonempty_lines(code: str) -> int:
    """Return the number of non‑empty lines in *code*."""
    return sum(1 for line in code.replace("\\n", "\n").split("\n") if line.strip())


def predict_smell(code_str: str) -> Dict[str, Union[bool, float, int]]:
    """Predict whether a Python function exhibits the long‑method code smell.

    Args:
        code_str: Source code of the function to analyze.

    Returns:
        A dictionary with the following keys:
            - is_bad_smell (bool): True if the model predicts a long method.
            - probability (float): Model confidence for the positive class (0‑1).
            - num_lines (int): Count of non‑empty lines in the function.
            - is_long_method (bool): Heuristic flag, True if *num_lines* exceeds
              :data:`LONG_METHOD_THRESHOLD`.

    Raises:
        Any exception raised by the underlying model loading or prediction
        utilities will propagate to the caller.
    """
    model, scaler = _load_model()
    features = extract_features_vector(code_str)
    scaled_features = scaler.transform(features)

    probability = float(model.predict_proba(scaled_features)[0, 1])
    prediction = int(model.predict(scaled_features)[0])

    num_lines = _count_nonempty_lines(code_str)

    return {
        "is_bad_smell": prediction == 1,
        "probability": probability,
        "num_lines": num_lines,
        "is_long_method": num_lines > LONG_METHOD_THRESHOLD,
    }


import numpy as np
from typing import List, Dict, Any, Tuple

LONG_METHOD_THRESHOLD = 20


def _load_model_and_scaler() -> Tuple[Any, Any]:
    """Load the trained model and its scaler."""
    return _load_model()


def _extract_feature_matrix(codes: List[str]) -> np.ndarray:
    """Convert source codes to a feature matrix."""
    feature_vectors = [extract_features_vector(code) for code in codes]
    return np.vstack(feature_vectors)


def _nonempty_line_count(code: str) -> int:
    """Count non‑empty lines in a source code string."""
    lines = code.replace("\\n", "\n").split("\n")
    return len([line for line in lines if line.strip()])


def _build_result_entry(
    pred: int,
    prob: float,
    line_count: int,
) -> Dict[str, Any]:
    """Create a dictionary with prediction details for a single code snippet."""
    return {
        "is_bad_smell": pred == 1,
        "probability": prob,
        "num_lines": line_count,
        "is_long_method": line_count > LONG_METHOD_THRESHOLD,
    }


def predict_smell_batch(codes: List[str]) -> List[Dict[str, Any]]:
    """Predict code smell for a batch of source code strings.

    Args:
        codes: List of source code strings to evaluate.

    Returns:
        A list of dictionaries, each containing:
            - is_bad_smell (bool): Whether the model predicts a smell.
            - probability (float): Probability of the positive class.
            - num_lines (int): Number of non‑empty lines in the code.
            - is_long_method (bool): True if the code exceeds
              LONG_METHOD_THRESHOLD lines.

    Raises:
        Any exception raised by the underlying model or scaler.
    """
    model, scaler = _load_model_and_scaler()
    feature_matrix = _extract_feature_matrix(codes)
    scaled_features = scaler.transform(feature_matrix)

    probabilities = model.predict_proba(scaled_features)[:, 1]
    predictions = model.predict(scaled_features)

    results: List[Dict[str, Any]] = []
    for idx, code in enumerate(codes):
        line_count = _nonempty_line_count(code)
        entry = _build_result_entry(
            pred=int(predictions[idx]),
            prob=float(probabilities[idx]),
            line_count=line_count,
        )
        results.append(entry)

    return results
