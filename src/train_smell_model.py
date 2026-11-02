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


import datetime
import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# Constants
SEED: int = 42
NUM_PERTURBATIONS: int = 3
EXTRA_COMMENT: str = "# extra comment"
LONG_EXTRA_COMMENT: str = "# additional logging"
LOGGER_INFO_TEMPLATE: str = 'logger.info(f"Processed batch")'
METRICS_RECORD_TEMPLATE: str = "metrics.record(count=len(results))"
LABEL_NO: str = "No"
LABEL_YES: str = "Yes"


def _generate_short_templates() -> List[str]:
    """Return a list of short function code templates."""
    return [
        '"""Get the value."""\nreturn self._value',
        '"""Set the name."""\nself._name = name',
        "return sum(data) / len(data)",
        "return [x for x in items if x > 0]",
        "return a + b",
        '"""Check if valid."""\nreturn value is not None and value > 0',
        '"""Format the string."""\nreturn f"{prefix}: {msg}"',
        "result = []\nfor i in range(n):\n    result.append(i * 2)\nreturn result",
        "total = 0\nfor x in numbers:\n    total += x\nreturn total / len(numbers)",
        "count = 0\nfor item in collection:\n    if item.active:\n        count += 1\nreturn count",
        "def wrapper(func):\n    \"\"\"Simple decorator.\"\"\"\n    return func",
        "if condition:\n    return option_a\nreturn option_b",
        "if x < 0:\n    return -1\nelif x == 0:\n    return 0\nreturn 1",
        "try:\n    return int(value)\nexcept ValueError:\n    return 0",
    ]


def _generate_long_templates() -> List[str]:
    """Return a list of long function code templates."""
    return [
        '"""Process data with full validation pipeline."""\n'
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
        "            'processed_at': datetime.datetime.now().isoformat(),\n"
        "        }\n"
        "        results.append(transformed)\n"
        "    except Exception as e:\n"
        "        errors.append(f\"Item {idx} error: {e}\")\n"
        "return {'results': results, 'errors': errors, 'count': len(results)}",
        '"""Parse and validate configuration from a file."""\n'
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
        '"""Process items in batches with retry logic."""\n'
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


def _extract_features(codes: List[str]) -> List[Dict]:
    """Extract feature dictionaries from a list of code strings."""
    return [extract_code_features(code) for code in codes]


def _perturb_short_templates(
    templates: List[str], repetitions: int
) -> Tuple[List[str], List[Dict], List[str]]:
    """Create perturbed short function variants."""
    codes: List[str] = []
    feats: List[Dict] = []
    labels: List[str] = []
    for _ in range(repetitions):
        for tmpl in templates:
            perturbed = f"{tmpl}\n    {EXTRA_COMMENT}\n    pass"
            codes.append(perturbed)
            feats.append(extract_code_features(perturbed))
            labels.append(LABEL_NO)
    return codes, feats, labels


def _perturb_long_templates(
    templates: List[str], repetitions: int
) -> Tuple[List[str], List[Dict], List[str]]:
    """Create extended long function variants."""
    codes: List[str] = []
    feats: List[Dict] = []
    labels: List[str] = []
    for _ in range(repetitions):
        for tmpl in templates:
            extended = (
                f"{tmpl}\n    {LONG_EXTRA_COMMENT}\n    {LOGGER_INFO_TEMPLATE}\n"
                f"    {METRICS_RECORD_TEMPLATE}"
            )
            codes.append(extended)
            feats.append(extract_code_features(extended))
            labels.append(LABEL_YES)
    return codes, feats, labels


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic training dataset when ``codeworks/dataset.csv`` is missing.

    The function creates short (label ``No``) and long (label ``Yes``) code
    snippets, extracts their features, and adds perturbed variants to enlarge
    the dataset.

    Returns:
        pandas.DataFrame: A DataFrame containing feature columns and a
        ``long_method`` column with labels ``"No"`` or ``"Yes"``.

    Raises:
        RuntimeError: If feature extraction fails for any generated snippet.
    """
    print("  codeworks/dataset.csv not found — generating synthetic training data")
    rng = np.random.default_rng(SEED)  # noqa: F841 – retained for potential future use

    short_templates = _generate_short_templates()
    long_templates = _generate_long_templates()

    # Base short and long codes
    short_codes = list(short_templates)
    long_codes = list(long_templates)

    # Extract base features
    short_feats = _extract_features(short_codes)
    long_feats = _extract_features(long_codes)

    # Labels for base data
    short_labels = [LABEL_NO] * len(short_codes)
    long_labels = [LABEL_YES] * len(long_codes)

    # Perturb and extend templates
    perturbed_short_codes, perturbed_short_feats, perturbed_short_labels = _perturb_short_templates(
        short_templates, NUM_PERTURBATIONS
    )
    perturbed_long_codes, perturbed_long_feats, perturbed_long_labels = _perturb_long_templates(
        long_templates, NUM_PERTURBATIONS
    )

    # Combine all data
    all_codes = short_codes + long_codes + perturbed_short_codes + perturbed_long_codes
    all_feats = short_feats + long_feats + perturbed_short_feats + perturbed_long_feats
    all_labels = short_labels + long_labels + perturbed_short_labels + perturbed_long_labels

    synthetic_df = pd.DataFrame(all_feats)
    synthetic_df["long_method"] = all_labels

    print(
        f"  Generated {len(synthetic_df)} synthetic training examples "
        f"(No={all_labels.count(LABEL_NO)}, Yes={all_labels.count(LABEL_YES)})"
    )
    return synthetic_df


import pathlib
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Constants
SIGMA: float = 0.25
NUM_LINES_NOISE_STD: float = 12.0
N_NOISE_COPIES: int = 4

def load_dataset() -> pd.DataFrame:
    """Load the dataset from CSV or generate a synthetic one.

    Returns:
        pd.DataFrame: Combined feature and target dataframe.
    """
    if CODEWORKS_CSV.exists():
        df_raw = pd.read_csv(str(CODEWORKS_CSV), encoding="latin-1")
        df_raw = df_raw[["code_section", "Approximate Number of Lines", "Long Method"]].copy()
        df_raw.columns = ["code_section", "num_lines", "long_method"]
        print(f"\nLoaded {len(df_raw)} rows from dataset.csv")
        feats = df_raw["code_section"].apply(extract_code_features)
        df_all = pd.concat([feats, df_raw[["long_method"]]], axis=1)
    else:
        df_all = generate_synthetic_dataset()
    return df_all


def encode_target(df: pd.DataFrame) -> Tuple[np.ndarray, LabelEncoder]:
    """Encode the target column to numeric labels.

    Args:
        df: Dataframe containing a 'long_method' column.

    Returns:
        Tuple of encoded target array and the fitted LabelEncoder.
    """
    le = LabelEncoder()
    y = le.fit_transform(df["long_method"])
    print(f"Class distribution: No={sum(y==0)}, Yes={sum(y==1)}")
    return y, le


def split_data(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform a stratified train/test split.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        X_train, X_test, y_train, y_test split arrays.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def augment_training_data(
    X_train: np.ndarray, y_train: np.ndarray, feature_stds: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """Create noisy copies of the training data and apply SMOTE.

    Args:
        X_train: Original training feature matrix.
        y_train: Original training target vector.
        feature_stds: Standard deviation of each feature column.

    Returns:
        Tuple of augmented feature matrix and target vector.
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
        X_noisy[:, binary_idx] = np.clip(np.round(X_noisy[:, binary_idx]), 0, 1)
        X_noisy = np.clip(X_noisy, 0, None)
        noise_copies.append(X_noisy)
        label_copies.append(y_train.copy())

    X_augmented = np.vstack([X_train] + noise_copies)
    y_augmented = np.concatenate([y_train] + label_copies)

    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_augmented, y_augmented)

    print(
        f"Augmented train: {X_resampled.shape}, classes: {np.bincount(y_resampled)}"
    )
    return X_resampled, y_resampled


def build_pipeline() -> Pipeline:
    """Construct the preprocessing and classification pipeline.

    Returns:
        Configured sklearn Pipeline.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=1.0, probability=True, random_state=SEED)),
        ]
    )


def cross_validate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    """Run stratified 5‑fold cross‑validation and print metrics.

    Args:
        pipeline: Trained sklearn Pipeline.
        X: Feature matrix.
        y: Target vector.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
    print(f"\nCV Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"CV F1 Score:  {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")


def evaluate_test_set(pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Evaluate the trained model on the real test set.

    Args:
        pipeline: Trained sklearn Pipeline.
        X_test: Test feature matrix.
        y_test: Test target vector.
    """
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test Accuracy (real data): {accuracy:.4f}")


def save_artifacts(pipeline: Pipeline) -> None:
    """Persist the trained model and scaler to disk.

    Args:
        pipeline: Trained sklearn Pipeline.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["svm"], str(MODELS_DIR / "smell_svm.pkl"))
    joblib.dump(pipeline.named_steps["scaler"], str(MODELS_DIR / "smell_scaler.pkl"))
    print(f"\n✅ Model saved to: {MODELS_DIR / 'smell_svm.pkl'}")
    print(f"✅ Scaler saved to: {MODELS_DIR / 'smell_scaler.pkl'}")


def main() -> None:
    """Train and evaluate the Long‑Method code smell detector."""
    print("=" * 60)
    print("  Training Long-Method Code Smell Detector")
    print("=" * 60)

    df_all = load_dataset()
    y, _ = encode_target(df_all)

    X = df_all[SAFE_FEATURES].copy()
    print(f"Feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = split_data(X.values, y)

    feature_stds = X.std()
    X_train_aug, y_train_aug = augment_training_data(
        X_train, y_train, feature_stds
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train_aug, y_train_aug)

    cross_validate(pipeline, X_train_aug, y_train_aug)
    evaluate_test_set(pipeline, X_test, y_test)
    save_artifacts(pipeline)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
