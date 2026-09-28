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
from datetime import datetime
import time
from typing import List, Tuple

SEED = 42
SHORT_LABEL = "No"
LONG_LABEL = "Yes"
PERTURBATION_ITERATIONS = 3
SHORT_COMMENT = "\n    # extra comment\n    pass"
LONG_COMMENT = (
    "\n    # additional logging\n"
    "    logger.info(f\"Processed batch\")\n"
    "    metrics.record(count=len(results))"
)


def _generate_codes(templates: List[str]) -> List[str]:
    """Return a copy of the provided template strings."""
    return list(templates)


def _extract_features(codes: List[str]) -> List[dict]:
    """Extract code features for each code snippet."""
    return [extract_code_features(code) for code in codes]


def _create_labels(count: int, label: str) -> List[str]:
    """Create a list of identical labels."""
    return [label] * count


def _perturb_codes(
    base_codes: List[str],
    comment: str,
    label: str,
    iterations: int,
) -> Tuple[List[str], List[dict], List[str]]:
    """Generate perturbed code snippets, their features, and labels."""
    perturbed_codes: List[str] = []
    perturbed_features: List[dict] = []
    perturbed_labels: List[str] = []

    for _ in range(iterations):
        for code in base_codes:
            new_code = code + comment
            perturbed_codes.append(new_code)
            perturbed_features.append(extract_code_features(new_code))
            perturbed_labels.append(label)

    return perturbed_codes, perturbed_features, perturbed_labels


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic training dataset when ``codeworks/dataset.csv`` is missing.

    The dataset consists of short (label ``No``) and long (label ``Yes``) function
    templates, optionally perturbed to increase variety. Each entry contains
    extracted code features and a ``long_method`` label.

    Returns:
        pandas.DataFrame: DataFrame with feature columns and a ``long_method`` label.
    """
    print("  codeworks/dataset.csv not found — generating synthetic training data")
    np.random.default_rng(SEED)  # rng is retained for side‑effects if any

    # Short function templates (label = "No")
    short_templates = [
        "\"\"\"Get the value.\"\"\"\nreturn self._value",
        "\"\"\"Set the name.\"\"\"\nself._name = name",
        "return sum(data) / len(data)",
        "return [x for x in items if x > 0]",
        "return a + b",
        "\"\"\"Check if valid.\"\"\"\nreturn value is not None and value > 0",
        "\"\"\"Format the string.\"\"\"\nreturn f\"{prefix}: {msg}\"",
        "result = []\nfor i in range(n):\n    result.append(i * 2)\nreturn result",
        "total = 0\nfor x in numbers:\n    total += x\nreturn total / len(numbers)",
        "count = 0\nfor item in collection:\n    if item.active:\n        count += 1\nreturn count",
        "def wrapper(func):\n    \"\"\"Simple decorator.\"\"\"\n    return func",
        "if condition:\n    return option_a\nreturn option_b",
        "if x < 0:\n    return -1\nelif x == 0:\n    return 0\nreturn 1",
        "try:\n    return int(value)\nexcept ValueError:\n    return 0",
    ]

    # Long function templates (label = "Yes")
    long_templates = [
        "\"\"\"Process data with full validation pipeline.\"\"\"\n"
        "errors = []\n"
        "results = []\n"
        "for idx, item in enumerate(data):\n"
        "    try:\n"
        "        if not isinstance(item, dict):\n"
        "            errors.append(f\"Item {idx} is not a dict\")\n"
        "            continue\n"
        "        required = ['id', 'name', 'value']\n"
        "        missing = [f for f in required if f not in item]\n"
        "        if missing:\n"
        "            errors.append(f\"Item {idx} missing: {missing}\")\n"
        "            continue\n"
        "        transformed = {\n"
        "            'id': item['id'],\n"
        "            'name': item['name'].strip(),\n"
        "            'value': float(item['value']),\n"
        "            'processed_at': datetime.now().isoformat(),\n"
        "        }\n"
        "        results.append(transformed)\n"
        "    except Exception as e:\n"
        "        errors.append(f\"Item {idx} error: {e}\")\n"
        "return {'results': results, 'errors': errors, 'count': len(results)}",

        "\"\"\"Parse and validate configuration from a file.\"\"\"\n"
        "config = {}\n"
        "with open(filepath, 'r') as f:\n"
        "    for line in f:\n"
        "        line = line.strip()\n"
        "        if not line or line.startswith('#'):\n"
        "            continue\n"
        "        if '=' not in line:\n"
        "            continue\n"
        "        key, value = line.split('=', 1)\n"
        "        key = key.strip()\n"
        "        value = value.strip()\n"
        "        if value.lower() in ('true', 'yes', '1'):\n"
        "            config[key] = True\n"
        "        elif value.lower() in ('false', 'no', '0'):\n"
        "            config[key] = False\n"
        "        else:\n"
        "            try:\n"
        "                config[key] = int(value)\n"
        "            except ValueError:\n"
        "                try:\n"
        "                    config[key] = float(value)\n"
        "                except ValueError:\n"
        "                    config[key] = value\n"
        "required = ['host', 'port', 'debug']\n"
        "missing = [k for k in required if k not in config]\n"
        "if missing:\n"
        "    raise ValueError(f\"Missing required config: {missing}\")\n"
        "config.setdefault('timeout', 30)\n"
        "config.setdefault('retries', 3)\n"
        "return config",

        "\"\"\"Process items in batches with retry logic.\"\"\"\n"
        "results = []\n"
        "failed = []\n"
        "for i in range(0, len(items), batch_size):\n"
        "    batch = items[i:i + batch_size]\n"
        "    for attempt in range(max_retries):\n"
        "        try:\n"
        "            batch_results = processor.process_batch(batch)\n"
        "            results.extend(batch_results)\n"
        "            break\n"
        "        except ConnectionError as e:\n"
        "            if attempt == max_retries - 1:\n"
        "                failed.extend(batch)\n"
        "            time.sleep(2 ** attempt)\n"
        "        except Exception as e:\n"
        "            failed.append({'batch': batch, 'error': str(e)})\n"
        "            break\n"
        "summary = {\n"
        "    'total': len(items),\n"
        "    'processed': len(results),\n"
        "    'failed': len(failed),\n"
        "    'success_rate': len(results) / len(items) * 100,\n"
        "}\n"
        "return results, failed, summary",
    ]

    # Base codes and features
    short_codes = _generate_codes(short_templates)
    long_codes = _generate_codes(long_templates)

    short_features = _extract_features(short_codes)
    long_features = _extract_features(long_codes)

    short_labels = _create_labels(len(short_codes), SHORT_LABEL)
    long_labels = _create_labels(len(long_codes), LONG_LABEL)

    # Perturb short and long templates
    perturbed_short_codes, perturbed_short_feats, perturbed_short_labels = _perturb_codes(
        short_templates, SHORT_COMMENT, SHORT_LABEL, PERTURBATION_ITERATIONS
    )
    perturbed_long_codes, perturbed_long_feats, perturbed_long_labels = _perturb_codes(
        long_templates, LONG_COMMENT, LONG_LABEL, PERTURBATION_ITERATIONS
    )

    # Combine all data
    all_features = short_features + long_features + perturbed_short_feats + perturbed_long_feats
    all_labels = short_labels + long_labels + perturbed_short_labels + perturbed_long_labels

    synthetic_df = pd.DataFrame(all_features)
    synthetic_df["long_method"] = all_labels

    print(
        f"  Generated {len(synthetic_df)} synthetic training examples "
        f"(No={all_labels.count(SHORT_LABEL)}, Yes={all_labels.count(LONG_LABEL)})"
    )
    return synthetic_df


import pathlib
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Constants
SIGMA = 0.25
NUM_LINES_NOISE_STD = 12.0
N_NOISE_COPIES = 4

def _load_dataset() -> pd.DataFrame:
    """Load the dataset from CSV or generate a synthetic one.

    Returns:
        pd.DataFrame: Combined feature and label DataFrame.
    """
    if CODEWORKS_CSV.exists():
        df_raw = pd.read_csv(str(CODEWORKS_CSV), encoding="latin-1")
        df_raw = df_raw[
            ["code_section", "Approximate Number of Lines", "Long Method"]
        ].copy()
        df_raw.columns = ["code_section", "num_lines", "long_method"]
        print(f"\nLoaded {len(df_raw)} rows from dataset.csv")
        feats = df_raw["code_section"].apply(extract_code_features)
        df_all = pd.concat([feats, df_raw[["long_method"]]], axis=1)
    else:
        df_all = generate_synthetic_dataset()
    return df_all


def _encode_labels(df: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder]:
    """Encode the target column into numeric labels.

    Args:
        df: DataFrame containing a ``long_method`` column.

    Returns:
        Tuple of label array and fitted ``LabelEncoder``.
    """
    le = LabelEncoder()
    y = le.fit_transform(df["long_method"])
    print(f"Class distribution: No={sum(y==0)}, Yes={sum(y==1)}")
    return y, le


def _split_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform a stratified train‑test split.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        Train‑test splits for features and labels.
    """
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )


def _augment_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_stds: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy copies of the training set and balance with SMOTE.

    Args:
        X_train: Original training features.
        y_train: Original training labels.
        feature_stds: Standard deviation of each feature column.

    Returns:
        Augmented feature matrix and label vector.
    """
    binary_idx = [SAFE_FEATURES.index(c) for c in BINARY_COLS]
    num_lines_idx = SAFE_FEATURES.index("num_lines")

    noise_copies = []
    label_copies = []

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
        X_noisy[:, binary_idx] = np.clip(
            np.round(X_noisy[:, binary_idx]), 0, 1
        )
        X_noisy = np.clip(X_noisy, 0, None)
        noise_copies.append(X_noisy)
        label_copies.append(y_train.copy())

    X_aug = np.vstack([X_train] + noise_copies)
    y_aug = np.concatenate([y_train] + label_copies)

    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X_aug, y_aug)

    print(
        f"Augmented train: {X_balanced.shape}, "
        f"classes: {np.bincount(y_balanced)}"
    )
    return X_balanced, y_balanced


def _train_pipeline(
    X: np.ndarray, y: np.ndarray
) -> Pipeline:
    """Fit a scaling + SVM pipeline on the provided data.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        Fitted ``Pipeline`` instance.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=1.0, probability=True, random_state=SEED)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def _cross_validate(
    pipeline: Pipeline, X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run stratified 5‑fold cross‑validation for accuracy and F1.

    Args:
        pipeline: Fitted pipeline to evaluate.
        X: Feature matrix.
        y: Target vector.

    Returns:
        Tuple of accuracy scores and F1 scores.
    """
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )
    acc = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy"
    )
    f1 = cross_val_score(
        pipeline, X, y, cv=cv, scoring="f1"
    )
    return acc, f1


def _evaluate(
    pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Compute test accuracy on real data.

    Args:
        pipeline: Trained pipeline.
        X_test: Test feature matrix.
        y_test: Test label vector.

    Returns:
        Accuracy as a float.
    """
    return pipeline.score(X_test, y_test)


def _save_model(pipeline: Pipeline) -> None:
    """Persist the SVM and scaler objects to disk.

    Args:
        pipeline: Trained pipeline containing ``svm`` and ``scaler`` steps.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    svm_path = MODELS_DIR / "smell_svm.pkl"
    scaler_path = MODELS_DIR / "smell_scaler.pkl"
    joblib.dump(pipeline.named_steps["svm"], str(svm_path))
    joblib.dump(pipeline.named_steps["scaler"], str(scaler_path))
    print(f"\n✅ Model saved to: {svm_path}")
    print(f"✅ Scaler saved to: {scaler_path}")


def main() -> None:
    """Execute the full training pipeline for the Long‑Method detector."""
    print("=" * 60)
    print("  Training Long-Method Code Smell Detector")
    print("=" * 60)

    df_all = _load_dataset()

    y, _ = _encode_labels(df_all)

    X = df_all[SAFE_FEATURES].copy()
    print(f"Feature matrix: {X.shape}")

    (
        X_train_real,
        X_test_real,
        y_train_real,
        y_test_real,
    ) = _split_data(X.values, y)
    print(f"\nTrain: {len(X_train_real)}, Test: {len(X_test_real)}")

    feature_stds = X.std()
    X_train_aug, y_train_aug = _augment_training_data(
        X_train_real, y_train_real, feature_stds
    )

    pipeline = _train_pipeline(X_train_aug, y_train_aug)

    cv_acc, cv_f1 = _cross_validate(pipeline, X_train_aug, y_train_aug)
    print(f"\nCV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"CV F1 Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    test_acc = _evaluate(pipeline, X_test_real, y_test_real)
    print(f"Test Accuracy (real data): {test_acc:.4f}")

    _save_model(pipeline)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
