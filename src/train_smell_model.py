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
from typing import List, Dict, Any, Tuple, Callable

import numpy as np
import pandas as pd
from datetime import datetime

# Constants
PERTURBATION_COUNT = 3
SHORT_LABEL = "No"
LONG_LABEL = "Yes"
EXTRA_COMMENT = "# extra comment"
LONG_EXTRA_COMMENT = "# additional logging"
LOGGER_INFO = "logger.info(f\"Processed batch\")"
METRICS_RECORD = "metrics.record(count=len(results))"


def _extract_features(codes: List[str]) -> List[Dict[str, Any]]:
    """Extract feature dictionaries for a list of code strings."""
    return [extract_code_features(code) for code in codes]


def _perturb_short(code: str) -> str:
    """Add a harmless comment and a pass statement to a short function."""
    return f"{code}\n    {EXTRA_COMMENT}\n    pass"


def _extend_long(code: str) -> str:
    """Append logging and metric collection to a long function."""
    return (
        f"{code}\n    {LONG_EXTRA_COMMENT}\n    {LOGGER_INFO}\n"
        f"    {METRICS_RECORD}"
    )


def _generate_variations(
    templates: List[str],
    label: str,
    perturb_func: Callable[[str], str],
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Create original and perturbed code samples with features and labels."""
    original_codes = templates.copy()
    original_features = _extract_features(original_codes)
    original_labels = [label] * len(original_codes)

    perturbed_codes: List[str] = []
    perturbed_features: List[Dict[str, Any]] = []
    perturbed_labels: List[str] = []

    for _ in range(PERTURBATION_COUNT):
        for tmpl in templates:
            perturbed = perturb_func(tmpl)
            perturbed_codes.append(perturbed)
            perturbed_features.append(extract_code_features(perturbed))
            perturbed_labels.append(label)

    all_codes = original_codes + perturbed_codes
    all_features = original_features + perturbed_features
    all_labels = original_labels + perturbed_labels
    return all_codes, all_features, all_labels


def _print_summary(labels: List[str]) -> None:
    """Print a summary of generated synthetic examples."""
    total = len(labels)
    no_count = labels.count(SHORT_LABEL)
    yes_count = labels.count(LONG_LABEL)
    print(
        f"  Generated {total} synthetic training examples "
        f"(No={no_count}, Yes={yes_count})"
    )


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic training dataset when `codeworks/dataset.csv` is missing.

    The dataset consists of short (label "No") and long (label "Yes") function
    templates, optionally perturbed to increase variety. Features are extracted
    via `extract_code_features`.

    Returns:
        pd.DataFrame: DataFrame with feature columns and a `long_method` label.

    Raises:
        Any exception raised by `extract_code_features` or pandas operations.
    """
    print(
        "  codeworks/dataset.csv not found — generating synthetic training data"
    )
    rng = np.random.default_rng(SEED)  # noqa: F841 – kept for compatibility

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

    short_codes, short_feats, short_labels = _generate_variations(
        short_templates, SHORT_LABEL, _perturb_short
    )
    long_codes, long_feats, long_labels = _generate_variations(
        long_templates, LONG_LABEL, _extend_long
    )

    all_features = short_feats + long_feats
    all_labels = short_labels + long_labels

    synthetic_df = pd.DataFrame(all_features)
    synthetic_df["long_method"] = all_labels
    _print_summary(all_labels)
    return synthetic_df


from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import joblib

# Constants
TEST_SIZE: float = 0.20
SIGMA: float = 0.25
NUM_LINES_NOISE_STD: float = 12.0
N_NOISE_COPIES: int = 4

def load_dataset() -> pd.DataFrame:
    """Load dataset from CSV or generate a synthetic one.

    Returns:
        pd.DataFrame: Combined feature and target data.
    """
    if CODEWORKS_CSV.exists():
        raw_df = pd.read_csv(str(CODEWORKS_CSV), encoding="latin-1")
        raw_df = raw_df[
            ["code_section", "Approximate Number of Lines", "Long Method"]
        ].copy()
        raw_df.columns = ["code_section", "num_lines", "long_method"]
        print(f"\nLoaded {len(raw_df)} rows from dataset.csv")
        features = raw_df["code_section"].apply(extract_code_features)
        dataset = pd.concat([features, raw_df[["long_method"]]], axis=1)
    else:
        dataset = generate_synthetic_dataset()
    return dataset


def prepare_features(
    dataset: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Encode target and extract feature matrix.

    Args:
        dataset: DataFrame containing features and target column.

    Returns:
        Tuple of feature matrix, encoded target vector, and label encoder.
    """
    le = LabelEncoder()
    y = le.fit_transform(dataset["long_method"])
    print(f"Class distribution: No={sum(y==0)}, Yes={sum(y==1)}")
    X = dataset[SAFE_FEATURES].to_numpy()
    print(f"Feature matrix: {X.shape}")
    return X, y, le


def split_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform stratified train‑test split.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        Train and test splits for features and target.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def augment_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_stds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy copies of training data and balance with SMOTE.

    Args:
        X_train: Original training features.
        y_train: Original training targets.
        feature_stds: Standard deviation per feature column.

    Returns:
        Resampled feature matrix and target vector.
    """
    binary_idx = [SAFE_FEATURES.index(col) for col in BINARY_COLS]
    num_lines_idx = SAFE_FEATURES.index("num_lines")
    noisy_features: list[np.ndarray] = []
    noisy_labels: list[np.ndarray] = []

    for _ in range(N_NOISE_COPIES):
        noise = rng.normal(
            loc=0.0,
            scale=(SIGMA * feature_stds),
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


def train_classifier(
    X: np.ndarray, y: np.ndarray
) -> Pipeline:
    """Train an SVM classifier wrapped in a scaling pipeline.

    Args:
        X: Training feature matrix.
        y: Training target vector.

    Returns:
        Fitted sklearn Pipeline.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=1.0, probability=True, random_state=SEED)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def cross_validate(
    pipeline: Pipeline, X: np.ndarray, y: np.ndarray
) -> None:
    """Run 5‑fold stratified cross‑validation and print metrics.

    Args:
        pipeline: Trained pipeline to evaluate.
        X: Feature matrix.
        y: Target vector.
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
    print(
        f"\nCV Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}"
    )
    print(
        f"CV F1 Score:  {f1_scores.mean():.4f} ± {f1_scores.std():.4f}"
    )


def evaluate_on_test(
    pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """Evaluate the model on the real test set.

    Args:
        pipeline: Trained pipeline.
        X_test: Test feature matrix.
        y_test: Test target vector.
    """
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test Accuracy (real data): {accuracy:.4f}")


def save_artifacts(pipeline: Pipeline) -> None:
    """Persist the trained SVM and scaler to disk.

    Args:
        pipeline: Trained pipeline containing SVM and scaler.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    svm_path = MODELS_DIR / "smell_svm.pkl"
    scaler_path = MODELS_DIR / "smell_scaler.pkl"
    joblib.dump(pipeline.named_steps["svm"], str(svm_path))
    joblib.dump(pipeline.named_steps["scaler"], str(scaler_path))
    print(f"\n✅ Model saved to: {svm_path}")
    print(f"✅ Scaler saved to: {scaler_path}")


def main() -> None:
    """Execute the training pipeline for the Long‑Method code smell detector."""
    print("=" * 60)
    print("  Training Long-Method Code Smell Detector")
    print("=" * 60)

    dataset = load_dataset()
    X, y, _ = prepare_features(dataset)
    X_train, X_test, y_train, y_test = split_data(X, y)
    feature_stds = np.std(X_train, axis=0)
    X_aug, y_aug = augment_training_data(
        X_train, y_train, feature_stds
    )
    model = train_classifier(X_aug, y_aug)
    cross_validate(model, X_aug, y_aug)
    evaluate_on_test(model, X_test, y_test)
    save_artifacts(model)


if __name__ == "__main__":
    main()
