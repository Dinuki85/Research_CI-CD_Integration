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


import pandas as pd
import numpy as np
from datetime import datetime
import time
from typing import List, Dict, Any

SEED = 42  # example seed, replace with actual if defined elsewhere
SHORT_LABEL = "No"
LONG_LABEL = "Yes"
PERTURBATION_ITERATIONS = 3
SHORT_SUFFIX = "\n    # extra comment\n    pass"
LONG_SUFFIX = (
    "\n    # additional logging\n"
    "    logger.info(f\"Processed batch\")\n"
    "    metrics.record(count=len(results))"
)


def _generate_templates() -> tuple[list[str], list[str]]:
    """Create short and long function templates.

    Returns:
        A tuple containing two lists:
        - short function templates (label = No)
        - long function templates (label = Yes)
    """
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

    return short_templates, long_templates


def _extract_features(codes: List[str]) -> List[Dict[str, Any]]:
    """Extract feature dictionaries from source code strings.

    Args:
        codes: List of source code snippets.

    Returns:
        List of feature dictionaries produced by ``extract_code_features``.
    """
    return [extract_code_features(code) for code in codes]


def _perturb_templates(
    templates: List[str],
    suffix: str,
    label: str,
) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Create perturbed versions of templates.

    Args:
        templates: Original code templates.
        suffix: Code to append for perturbation.
        label: Label associated with the templates.

    Returns:
        Tuple of (new_codes, new_features, new_labels).
    """
    new_codes: List[str] = []
    new_features: List[Dict[str, Any]] = []
    new_labels: List[str] = []

    for _ in range(PERTURBATION_ITERATIONS):
        for tmpl in templates:
            perturbed = tmpl + suffix
            new_codes.append(perturbed)
            new_features.append(extract_code_features(perturbed))
            new_labels.append(label)

    return new_codes, new_features, new_labels


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic training dataset when ``codeworks/dataset.csv`` is missing.

    The function creates short (label ``No``) and long (label ``Yes``) Python
    function snippets, extracts their code features, and expands the set with
    slight perturbations. The resulting ``pandas.DataFrame`` contains one row
    per example and a ``long_method`` column with the corresponding label.

    Returns:
        DataFrame with feature columns and a ``long_method`` label column.
    """
    print(
        "  codeworks/dataset.csv not found — generating synthetic training data"
    )
    rng = np.random.default_rng(SEED)  # noqa: F841 – kept for reproducibility

    short_templates, long_templates = _generate_templates()

    # Base short examples
    short_codes = list(short_templates)
    short_features = _extract_features(short_codes)
    short_labels = [SHORT_LABEL] * len(short_codes)

    # Base long examples
    long_codes = list(long_templates)
    long_features = _extract_features(long_codes)
    long_labels = [LONG_LABEL] * len(long_codes)

    # Perturb short and long templates
    perturbed_short, perturbed_short_feats, perturbed_short_labels = _perturb_templates(
        short_templates, SHORT_SUFFIX, SHORT_LABEL
    )
    perturbed_long, perturbed_long_feats, perturbed_long_labels = _perturb_templates(
        long_templates, LONG_SUFFIX, LONG_LABEL
    )

    # Combine all data
    all_features = (
        short_features
        + long_features
        + perturbed_short_feats
        + perturbed_long_feats
    )
    all_labels = (
        short_labels
        + long_labels
        + perturbed_short_labels
        + perturbed_long_labels
    )

    synthetic_df = pd.DataFrame(all_features)
    synthetic_df["long_method"] = all_labels

    print(
        f"  Generated {len(synthetic_df)} synthetic training examples "
        f"(No={all_labels.count(SHORT_LABEL)}, "
        f"Yes={all_labels.count(LONG_LABEL)})"
    )
    return synthetic_df


import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from imblearn.over_sampling import SMOTE

# Constants
SEED: int = 42
TEST_SIZE: float = 0.20
SIGMA: float = 0.25
NUM_LINES_NOISE_STD: float = 12.0
N_NOISE_COPIES: int = 4
COLUMNS_FEATURES: list[str] = SAFE_FEATURES  # type: ignore[name-defined]
BINARY_COLUMNS: list[str] = BINARY_COLS  # type: ignore[name-defined]
DATASET_PATH: Path = CODEWORKS_CSV  # type: ignore[name-defined]
MODELS_DIRECTORY: Path = MODELS_DIR  # type: ignore[name-defined]


def load_dataset() -> pd.DataFrame:
    """Load the dataset from CSV or generate a synthetic one.

    Returns:
        pd.DataFrame: Combined feature and target dataframe.
    """
    if DATASET_PATH.exists():
        df_raw = pd.read_csv(str(DATASET_PATH), encoding="latin-1")
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


def encode_target(df: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder]:
    """Encode the target column.

    Args:
        df: Dataframe containing a 'long_method' column.

    Returns:
        Tuple of encoded target array and the fitted LabelEncoder.
    """
    le = LabelEncoder()
    y = le.fit_transform(df["long_method"])
    print(f"Class distribution: No={sum(y==0)}, Yes={sum(y==1)}")
    return y, le


def augment_training_data(
    X_train: np.ndarray,
) -> np.ndarray:
    """Create noisy copies of the training data for augmentation.

    Args:
        X_train: Original training feature matrix.

    Returns:
        Augmented feature matrix before SMOTE.
    """
    feature_stds = pd.DataFrame(X_train, columns=COLUMNS_FEATURES).std()
    binary_idx = [COLUMNS_FEATURES.index(col) for col in BINARY_COLUMNS]
    num_lines_idx = COLUMNS_FEATURES.index("num_lines")

    noise_copies = []
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
    return np.vstack([X_train] + noise_copies)


def balance_dataset(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to balance classes.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        Balanced feature matrix and target vector.
    """
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    return smote.fit_resample(X, y)


def train_pipeline(
    X: np.ndarray, y: np.ndarray
) -> Pipeline:
    """Train an SVM pipeline on the provided data.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        Trained scikit-learn Pipeline.
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
    """Perform stratified 5‑fold cross‑validation and print metrics.

    Args:
        pipeline: Trained pipeline.
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
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Evaluate the pipeline on the real test set and print accuracy.

    Args:
        pipeline: Trained pipeline.
        X_test: Test feature matrix.
        y_test: Test target vector.
    """
    acc = pipeline.score(X_test, y_test)
    print(f"Test Accuracy (real data): {acc:.4f}")


def save_artifacts(pipeline: Pipeline) -> None:
    """Persist the trained SVM model and scaler to disk.

    Args:
        pipeline: Trained pipeline containing 'svm' and 'scaler' steps.
    """
    MODELS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        pipeline.named_steps["svm"],
        str(MODELS_DIRECTORY / "smell_svm.pkl"),
    )
    joblib.dump(
        pipeline.named_steps["scaler"],
        str(MODELS_DIRECTORY / "smell_scaler.pkl"),
    )
    print(f"\n✅ Model saved to: {MODELS_DIRECTORY / 'smell_svm.pkl'}")
    print(f"✅ Scaler saved to: {MODELS_DIRECTORY / 'smell_scaler.pkl'}")


def main() -> None:
    """Entry point for training the Long‑Method Code Smell Detector."""
    print("=" * 60)
    print("  Training Long-Method Code Smell Detector")
    print("=" * 60)

    df_all = load_dataset()
    y, _ = encode_target(df_all)

    X = df_all[COLUMNS_FEATURES].copy()
    print(f"Feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    X_augmented = augment_training_data(X_train)
    y_augmented = np.concatenate(
        [y_train] + [y_train.copy() for _ in range(N_NOISE_COPIES)]
    )
    X_balanced, y_balanced = balance_dataset(
        X_augmented, y_augmented
    )
    print(
        f"Augmented train: {X_balanced.shape}, classes: {np.bincount(y_balanced)}"
    )

    pipeline = train_pipeline(X_balanced, y_balanced)
    cross_validate(pipeline, X_balanced, y_balanced)
    evaluate_on_test(pipeline, X_test, y_test)
    save_artifacts(pipeline)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
