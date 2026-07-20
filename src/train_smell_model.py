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


import time
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


def _extract_features(
    templates: List[str],
) -> Tuple[List[Dict], List[str]]:
    """Extract features from code templates and assign a label.

    Args:
        templates: List of code snippets.

    Returns:
        A tuple containing a list of feature dictionaries and a list of labels.
    """
    features = [extract_code_features(t) for t in templates]
    return features, []


def _perturb_templates(
    base_templates: List[str],
    label: str,
    repetitions: int,
    suffix: str,
) -> Tuple[List[str], List[Dict], List[str]]:
    """Create perturbed variants of the given templates.

    Args:
        base_templates: Original code snippets.
        label: Label to assign to each variant.
        repetitions: How many times to repeat the perturbation loop.
        suffix: Code to append for perturbation.

    Returns:
        Tuple of (augmented_codes, augmented_features, augmented_labels).
    """
    augmented_codes: List[str] = []
    augmented_features: List[Dict] = []
    augmented_labels: List[str] = []

    for _ in range(repetitions):
        for template in base_templates:
            variant = f"{template}\n    # extra comment\n    pass" if label == "No" else f"{template}\n    # additional logging\n    logger.info(\"Processed batch\")\n    metrics.record(count=len(results))"
            augmented_codes.append(variant)
            augmented_features.append(extract_code_features(variant))
            augmented_labels.append(label)

    return augmented_codes, augmented_features, augmented_labels


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic training dataset when dataset.csv is unavailable.

    The function creates short (label="No") and long (label="Yes") code
    snippets, extracts their features, adds perturbed variants, and returns a
    pandas DataFrame ready for model training.

    Returns:
        DataFrame containing feature columns and a ``long_method`` column with
        labels "Yes" or "No".
    """
    print("  codeworks/dataset.csv not found — generating synthetic training data")
    rng = np.random.default_rng(SEED)

    short_templates: List[str] = [
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

    long_templates: List[str] = [
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

    short_features, _ = _extract_features(short_templates)
    short_labels = ["No"] * len(short_features)

    long_features, _ = _extract_features(long_templates)
    long_labels = ["Yes"] * len(long_features)

    short_variants, short_variant_feats, short_variant_labels = _perturb_templates(
        short_templates, "No", repetitions=3, suffix=""
    )
    short_features.extend(short_variant_feats)
    short_labels.extend(short_variant_labels)

    long_variants, long_variant_feats, long_variant_labels = _perturb_templates(
        long_templates, "Yes", repetitions=3, suffix=""
    )
    long_features.extend(long_variant_feats)
    long_labels.extend(long_variant_labels)

    all_features = short_features + long_features
    all_labels = short_labels + long_labels

    synthetic_df = pd.DataFrame(all_features)
    synthetic_df["long_method"] = all_labels

    print(
        f"  Generated {len(synthetic_df)} synthetic training examples "
        f"(No={all_labels.count('No')}, Yes={all_labels.count('Yes')})"
    )
    return synthetic_df


import pathlib
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

def _print_header() -> None:
    """Print a formatted header for the training run."""
    line = "=" * 60
    print(line)
    print("  Training Long-Method Code Smell Detector")
    print(line)


def _load_dataset() -> pd.DataFrame:
    """Load the dataset from disk or generate a synthetic one.

    Returns:
        pd.DataFrame: Combined feature and label dataframe.
    """
    if CODEWORKS_CSV.exists():
        raw_df = pd.read_csv(str(CODEWORKS_CSV), encoding="latin-1")
        raw_df = raw_df[
            ["code_section", "Approximate Number of Lines", "Long Method"]
        ].copy()
        raw_df.columns = ["code_section", "num_lines", "long_method"]
        print(f"\nLoaded {len(raw_df)} rows from dataset.csv")
        feature_df = raw_df["code_section"].apply(extract_code_features)
        combined_df = pd.concat([feature_df, raw_df[["long_method"]]], axis=1)
    else:
        combined_df = generate_synthetic_dataset()
    return combined_df


def _encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Encode the target column using LabelEncoder.

    Args:
        df: Dataframe containing a 'long_method' column.

    Returns:
        Tuple of feature dataframe, label array, and fitted encoder.
    """
    encoder = LabelEncoder()
    labels = encoder.fit_transform(df["long_method"])
    print(f"Class distribution: No={sum(labels==0)}, Yes={sum(labels==1)}")
    features = df[SAFE_FEATURES].copy()
    print(f"Feature matrix: {features.shape}")
    return features, labels, encoder


def _split_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform a stratified train‑test split.

    Args:
        X: Feature matrix.
        y: Label vector.

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
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy copies of the training data and apply SMOTE.

    Args:
        X_train: Original training feature matrix.
        y_train: Original training label vector.

    Returns:
        Tuple of augmented feature matrix and label vector.
    """
    feature_stds = pd.DataFrame(X_train, columns=SAFE_FEATURES).std()
    binary_indices = [
        SAFE_FEATURES.index(col) for col in BINARY_COLS
    ]
    num_lines_idx = SAFE_FEATURES.index("num_lines")

    noisy_features = []
    noisy_labels = []

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
        X_noisy[:, binary_indices] = np.clip(
            np.round(X_noisy[:, binary_indices]), 0, 1
        )
        X_noisy = np.clip(X_noisy, 0, None)
        noisy_features.append(X_noisy)
        noisy_labels.append(y_train.copy())

    X_aug = np.vstack([X_train] + noisy_features)
    y_aug = np.concatenate([y_train] + noisy_labels)

    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_aug, y_aug)

    print(
        f"Augmented train: {X_resampled.shape}, "
        f"classes: {np.bincount(y_resampled)}"
    )
    return X_resampled, y_resampled


def _train_pipeline(
    X_train: np.ndarray, y_train: np.ndarray
) -> Pipeline:
    """Fit a scaling + SVM pipeline on the training data.

    Args:
        X_train: Feature matrix.
        y_train: Label vector.

    Returns:
        Trained scikit‑learn Pipeline.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=1.0, probability=True, random_state=SEED)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def _cross_validate(
    pipeline: Pipeline, X: np.ndarray, y: np.ndarray
) -> None:
    """Run stratified 5‑fold cross‑validation and print metrics.

    Args:
        pipeline: Trained pipeline to evaluate.
        X: Feature matrix.
        y: Label vector.
    """
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )
    acc_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy"
    )
    f1_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="f1"
    )
    print(f"\nCV Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"CV F1 Score:  {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")


def _evaluate_test_set(
    pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """Evaluate the pipeline on the held‑out test set.

    Args:
        pipeline: Trained pipeline.
        X_test: Test feature matrix.
        y_test: Test label vector.
    """
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test Accuracy (real data): {accuracy:.4f}")


def _save_model(pipeline: Pipeline) -> None:
    """Persist the trained SVM and scaler objects to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        pipeline.named_steps["svm"],
        str(MODELS_DIR / "smell_svm.pkl"),
    )
    joblib.dump(
        pipeline.named_steps["scaler"],
        str(MODELS_DIR / "smell_scaler.pkl"),
    )
    print(f"\n✅ Model saved to: {MODELS_DIR / 'smell_svm.pkl'}")
    print(f"✅ Scaler saved to: {MODELS_DIR / 'smell_scaler.pkl'}")


def run_training_pipeline() -> None:
    """Execute the full training pipeline for the long‑method detector.

    The function loads data, augments it, trains an SVM model,
    evaluates via cross‑validation and a held‑out test set,
    and finally saves the trained model and scaler.
    """
    _print_header()

    dataset = _load_dataset()
    features_df, labels, _ = _encode_labels(dataset)

    X_matrix = features_df.values
    (
        X_train_real,
        X_test_real,
        y_train_real,
        y_test_real,
    ) = _split_data(X_matrix, labels)
    print(f"\nTrain: {len(X_train_real)}, Test: {len(X_test_real)}")

    X_train_aug, y_train_aug = _augment_training_data(
        X_train_real, y_train_real
    )

    model_pipeline = _train_pipeline(X_train_aug, y_train_aug)

    _cross_validate(model_pipeline, X_train_aug, y_train_aug)
    _evaluate_test_set(model_pipeline, X_test_real, y_test_real)
    _save_model(model_pipeline)


if __name__ == "__main__":
    run_training_pipeline()


if __name__ == "__main__":
    main()
