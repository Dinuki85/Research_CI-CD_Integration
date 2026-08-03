#!/usr/bin/env python3
"""
Train Smell Detection Model
============================
Trains an SVM classifier to detect long-method code smell in Python functions.

Replicates the approach from codeworks/model_synthetic.ipynb:
  1. Load dataset.csv from codeworks/
  2. Extract 12 safe features (avoids data leakage)
  3. Gaussian noise augmentation (4 copies, sigma=0.25, num_lines_noise=12)
  4. SMOTE oversampling for class balance
  5. Train SVM with standardization
  6. Save model and scaler to app/models/

Usage:
    python src/train_smell_model.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
CODEWORKS_CSV = Path(__file__).resolve().parent.parent.parent / "codeworks" / "dataset.csv"
MODELS_DIR    = Path(__file__).resolve().parent.parent / "models"

# ── Feature definitions (must match code_smell_detector.py) ─────────────────
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

BINARY_COLS = [
    "code_has_loop",
    "code_has_conditional",
    "code_has_return",
    "code_has_try",
]

SEED = 42
rng = np.random.default_rng(SEED)


def extract_code_features(code):
    """Same as code_smell_detector.extract_code_features."""
    code = str(code)
    lines = code.split("\n")
    nonempty = [l for l in lines if l.strip()]
    line_lengths = [len(l) for l in nonempty]

    return pd.Series({
        "num_lines":             len(nonempty),
        "code_avg_line_len":     float(np.mean(line_lengths)) if line_lengths else 0.0,
        "code_max_line_len":     float(max(line_lengths)) if line_lengths else 0.0,
        "code_indent_depth":     float(max(len(l) - len(l.lstrip()) for l in nonempty)) if nonempty else 0.0,
        "code_has_loop":         int(any(kw in code for kw in ["for ", "while "])),
        "code_has_conditional":  int(any(kw in code for kw in ["if ", "elif ", "else:"])),
        "code_has_return":       int("return" in code),
        "code_has_try":          int("try:" in code or "except" in code),
        "code_num_returns":      code.count("return"),
        "code_num_ifs":          code.count("if "),
        "code_num_loops":        code.count("for ") + code.count("while "),
        "code_comment_lines":    sum(1 for l in lines if l.strip().startswith("#")),
    })


import numpy as np
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Any

SEED = 42  # Assuming a default seed if not defined elsewhere

SHORT_LABEL = "No"
LONG_LABEL = "Yes"
PERTURBATION_ITERATIONS = 3


def _get_short_templates() -> List[str]:
    """Return a list of short function code templates."""
    return [
        '"""Get the value."""\nreturn self._value',
        '"""Set the name."""\nself._name = name',
        'return sum(data) / len(data)',
        'return [x for x in items if x > 0]',
        'return a + b',
        '"""Check if valid."""\nreturn value is not None and value > 0',
        '"""Format the string."""\nreturn f"{prefix}: {msg}"',
        'result = []\nfor i in range(n):\n    result.append(i * 2)\nreturn result',
        'total = 0\nfor x in numbers:\n    total += x\nreturn total / len(numbers)',
        'count = 0\nfor item in collection:\n    if item.active:\n        count += 1\nreturn count',
        'def wrapper(func):\n    """Simple decorator."""\n    return func',
        'if condition:\n    return option_a\nreturn option_b',
        'if x < 0:\n    return -1\nelif x == 0:\n    return 0\nreturn 1',
        'try:\n    return int(value)\nexcept ValueError:\n    return 0',
    ]


def _get_long_templates() -> List[str]:
    """Return a list of long function code templates."""
    return [
        '"""Process data with full validation pipeline."""\n'
        'errors = []\n'
        'results = []\n'
        'for idx, item in enumerate(data):\n'
        '    try:\n'
        '        if not isinstance(item, dict):\n'
        '            errors.append(f"Item {idx} is not a dict")\n'
        '            continue\n'
        '        required = ["id", "name", "value"]\n'
        '        missing = [f for f in required if f not in item]\n'
        '        if missing:\n'
        '            errors.append(f"Item {idx} missing: {missing}")\n'
        '            continue\n'
        '        transformed = {\n'
        '            "id": item["id"],\n'
        '            "name": item["name"].strip(),\n'
        '            "value": float(item["value"]),\n'
        '            "processed_at": datetime.now().isoformat(),\n'
        '        }\n'
        '        results.append(transformed)\n'
        '    except Exception as e:\n'
        '        errors.append(f"Item {idx} error: {e}")\n'
        'return {"results": results, "errors": errors, "count": len(results)}',
        '"""Parse and validate configuration from a file."""\n'
        'config = {}\n'
        "with open(filepath, 'r') as f:\n"
        '    for line in f:\n'
        '        line = line.strip()\n'
        '        if not line or line.startswith("#"):\n'
        '            continue\n'
        '        if "=" not in line:\n'
        '            continue\n'
        '        key, value = line.split("=", 1)\n'
        '        key = key.strip()\n'
        '        value = value.strip()\n'
        '        if value.lower() in ("true", "yes", "1"):\n'
        '            config[key] = True\n'
        '        elif value.lower() in ("false", "no", "0"):\n'
        '            config[key] = False\n'
        '        else:\n'
        '            try:\n'
        '                config[key] = int(value)\n'
        '            except ValueError:\n'
        '                try:\n'
        '                    config[key] = float(value)\n'
        '                except ValueError:\n'
        '                    config[key] = value\n'
        'required = ["host", "port", "debug"]\n'
        'missing = [k for k in required if k not in config]\n'
        'if missing:\n'
        '    raise ValueError(f"Missing required config: {missing}")\n'
        'config.setdefault("timeout", 30)\n'
        'config.setdefault("retries", 3)\n'
        'return config',
        '"""Process items in batches with retry logic."""\n'
        'results = []\n'
        'failed = []\n'
        'for i in range(0, len(items), batch_size):\n'
        '    batch = items[i:i + batch_size]\n'
        '    for attempt in range(max_retries):\n'
        '        try:\n'
        '            batch_results = processor.process_batch(batch)\n'
        '            results.extend(batch_results)\n'
        '            break\n'
        '        except ConnectionError as e:\n'
        '            if attempt == max_retries - 1:\n'
        '                failed.extend(batch)\n'
        '            time.sleep(2 ** attempt)\n'
        '        except Exception as e:\n'
        '            failed.append({"batch": batch, "error": str(e)})\n'
        '            break\n'
        'summary = {\n'
        '    "total": len(items),\n'
        '    "processed": len(results),\n'
        '    "failed": len(failed),\n'
        '    "success_rate": len(results) / len(items) * 100,\n'
        '}\n'
        'return results, failed, summary',
    ]


def _extract_features(codes: List[str]) -> List[Dict[str, Any]]:
    """Extract feature dictionaries for a list of code strings."""
    return [extract_code_features(code) for code in codes]


def _augment_codes(
    base_templates: List[str],
    label: str,
    iterations: int,
    suffix: str,
) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Create augmented code samples, their features, and labels."""
    codes: List[str] = []
    features: List[Dict[str, Any]] = []
    labels: List[str] = []

    for template in base_templates:
        for _ in range(iterations):
            augmented = f"{template}\n    {suffix}"
            codes.append(augmented)
            features.append(extract_code_features(augmented))
            labels.append(label)

    return codes, features, labels


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic training dataset when dataset.csv is missing.

    Returns:
        pd.DataFrame: DataFrame containing extracted features and a
            'long_method' label indicating short (No) or long (Yes) code.

    Raises:
        RuntimeError: If feature extraction fails for any generated code.
    """
    print(
        "  codeworks/dataset.csv not found — generating synthetic training data"
    )
    rng = np.random.default_rng(SEED)  # noqa: F841 – kept for reproducibility

    short_templates = _get_short_templates()
    long_templates = _get_long_templates()

    short_codes = short_templates.copy()
    long_codes = long_templates.copy()

    short_features = _extract_features(short_codes)
    long_features = _extract_features(long_codes)

    short_labels = [SHORT_LABEL] * len(short_codes)
    long_labels = [LONG_LABEL] * len(long_codes)

    short_aug, short_aug_feats, short_aug_labels = _augment_codes(
        short_templates,
        SHORT_LABEL,
        PERTURBATION_ITERATIONS,
        "# extra comment\n    pass",
    )
    long_aug, long_aug_feats, long_aug_labels = _augment_codes(
        long_templates,
        LONG_LABEL,
        PERTURBATION_ITERATIONS,
        "# additional logging\n    logger.info(\"Processed batch\")\n"
        "    metrics.record(count=len(results))",
    )

    all_codes = short_codes + long_codes + short_aug + long_aug
    all_features = short_features + long_features + short_aug_feats + long_aug_feats
    all_labels = short_labels + long_labels + short_aug_labels + long_aug_labels

    synthetic_df = pd.DataFrame(all_features)
    synthetic_df["long_method"] = all_labels

    print(
        f"  Generated {len(synthetic_df)} synthetic training examples "
        f"(No={all_labels.count(SHORT_LABEL)}, "
        f"Yes={all_labels.count(LONG_LABEL)})"
    )
    return synthetic_df


import pathlib
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# External constants / utilities (assumed to be defined elsewhere)
CODEWORKS_CSV: pathlib.Path
SAFE_FEATURES: List[str]
BINARY_COLS: List[str]
SEED: int
MODELS_DIR: pathlib.Path
rng: np.random.Generator
extract_code_features: callable
generate_synthetic_dataset: callable

# Magic numbers replaced with named constants
SIGMA: float = 0.25
NUM_LINES_NOISE_STD: float = 12.0
N_NOISE_COPIES: int = 4


def _load_dataset() -> pd.DataFrame:
    """Load the dataset from CSV or generate a synthetic one.

    Returns:
        pd.DataFrame: Combined feature and label dataframe.
    """
    if CODEWORKS_CSV.exists():
        df_raw = pd.read_csv(str(CODEWORKS_CSV), encoding="latin-1")
        df_raw = df_raw[["code_section", "Approximate Number of Lines", "Long Method"]].copy()
        df_raw.columns = ["code_section", "num_lines", "long_method"]
        feats = df_raw["code_section"].apply(extract_code_features)
        df_all = pd.concat([feats, df_raw[["long_method"]]], axis=1)
    else:
        df_all = generate_synthetic_dataset()
    return df_all


def _prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Encode labels and extract feature matrix.

    Args:
        df: Dataframe containing features and the target column.

    Returns:
        Tuple of (X, y, feature_names).
    """
    le = LabelEncoder()
    y = le.fit_transform(df["long_method"])
    X = df[SAFE_FEATURES].copy()
    return X.values, y, SAFE_FEATURES


def _split_data(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform stratified train/test split.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    return train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )


def _augment_training_data(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Create noisy copies of the training data and apply SMOTE.

    Args:
        X_train: Original training features.
        y_train: Original training labels.
        feature_names: List of feature column names.

    Returns:
        Augmented feature matrix and label vector.
    """
    feature_stds = pd.DataFrame(X_train, columns=feature_names).std()
    binary_idx = [feature_names.index(col) for col in BINARY_COLS]
    num_lines_idx = feature_names.index("num_lines")

    noise_copies: List[np.ndarray] = []
    label_copies: List[np.ndarray] = []

    for _ in range(N_NOISE_COPIES):
        noise = rng.normal(
            loc=0.0,
            scale=(SIGMA * feature_stds).values,
            size=X_train.shape,
        )
        noise[:, num_lines_idx] = rng.normal(
            0, NUM_LINES_NOISE_STD, size=len(X_train)
        )
        X_noisy = X_train + noise
        X_noisy[:, binary_idx] = np.clip(np.round(X_noisy[:, binary_idx]), 0, 1)
        X_noisy = np.clip(X_noisy, 0, None)
        noise_copies.append(X_noisy)
        label_copies.append(y_train.copy())

    X_augmented = np.vstack([X_train] + noise_copies)
    y_augmented = np.concatenate([y_train] + label_copies)

    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_augmented, y_augmented)
    return X_resampled, y_resampled


def _train_pipeline(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Fit a scaling + SVM pipeline on the provided data.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        Trained sklearn Pipeline.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=1.0, probability=True, random_state=SEED)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def _cross_validate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Run stratified 5‑fold cross‑validation and return mean accuracy and F1.

    Args:
        pipeline: Trained pipeline.
        X: Feature matrix.
        y: Target vector.

    Returns:
        Tuple of (mean_accuracy, mean_f1_score).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
    return acc_scores.mean(), f1_scores.mean()


def _evaluate(pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate the pipeline on a held‑out test set.

    Args:
        pipeline: Trained pipeline.
        X_test: Test feature matrix.
        y_test: Test target vector.

    Returns:
        Test accuracy.
    """
    return pipeline.score(X_test, y_test)


def _save_model(pipeline: Pipeline) -> None:
    """Persist the trained SVM and scaler to disk.

    Args:
        pipeline: Trained pipeline containing 'svm' and 'scaler' steps.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["svm"], str(MODELS_DIR / "smell_svm.pkl"))
    joblib.dump(pipeline.named_steps["scaler"], str(MODELS_DIR / "smell_scaler.pkl"))


def main() -> None:
    """Orchestrate training of the Long‑Method Code Smell detector.

    This function loads data, extracts features, augments the training set,
    trains an SVM model, evaluates it, and saves the resulting artifacts.
    """
    print("=" * 60)
    print("  Training Long-Method Code Smell Detector")
    print("=" * 60)

    df = _load_dataset()
    print(f"\nLoaded {len(df)} rows from dataset")

    X, y, feature_names = _prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = _split_data(X, y)
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    X_aug, y_aug = _augment_training_data(X_train, y_train, feature_names)
    print(f"Augmented train shape: {X_aug.shape}, class distribution: {np.bincount(y_aug)}")

    pipeline = _train_pipeline(X_aug, y_aug)

    mean_acc, mean_f1 = _cross_validate(pipeline, X_aug, y_aug)
    print(f"\nCV Accuracy: {mean_acc:.4f}")
    print(f"CV F1 Score: {mean_f1:.4f}")

    test_acc = _evaluate(pipeline, X_test, y_test)
    print(f"Test Accuracy (real data): {test_acc:.4f}")

    _save_model(pipeline)
    print(f"\n✅ Model saved to: {MODELS_DIR / 'smell_svm.pkl'}")
    print(f"✅ Scaler saved to: {MODELS_DIR / 'smell_scaler.pkl'}")


if __name__ == "__main__":
    main()
