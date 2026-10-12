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
from typing import List, Dict, Any, Tuple

# Number of times to generate perturbed variants for each template
_PERTURBATION_REPEATS = 3


def _get_template_lists() -> Tuple[List[str], List[str]]:
    """Return short and long function templates used for synthetic data."""
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
    """Extract feature dictionaries from a list of code strings."""
    return [extract_code_features(code) for code in codes]


def _perturb_templates(
    base_templates: List[str],
    label: str,
    extra_snippet: str,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Create perturbed variants of templates and compute their features."""
    codes: List[str] = []
    features: List[Dict[str, Any]] = []
    labels: List[str] = []

    for _ in range(_PERTURBATION_REPEATS):
        for tmpl in base_templates:
            perturbed = f"{tmpl}\n    # extra comment\n    pass{extra_snippet}"
            codes.append(perturbed)
            features.append(extract_code_features(perturbed))
            labels.append(label)

    return codes, features, labels


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic dataset of short and long Python functions.

    Returns:
        pd.DataFrame: Dataframe where each row represents extracted code
        features and a ``long_method`` column indicating the presence of a
        long‑method smell (\"Yes\" or \"No\").
    """
    print(
        "  codeworks/dataset.csv not found — generating synthetic training data"
    )
    rng = np.random.default_rng(SEED)  # noqa: F841 – kept for parity with original

    short_templates, long_templates = _get_template_lists()

    # Base short/long codes and their features
    short_codes = short_templates.copy()
    long_codes = long_templates.copy()
    short_features = _extract_features(short_codes)
    long_features = _extract_features(long_codes)
    short_labels = ["No"] * len(short_codes)
    long_labels = ["Yes"] * len(long_codes)

    # Perturb short templates
    extra_short = "\n    pass"
    pert_codes, pert_feats, pert_labels = _perturb_templates(
        short_templates, "No", extra_short
    )
    short_codes.extend(pert_codes)
    short_features.extend(pert_feats)
    short_labels.extend(pert_labels)

    # Perturb long templates with additional logging snippet
    extra_long = (
        "\n    # additional logging\n"
        "    logger.info(f\"Processed batch\")\n"
        "    metrics.record(count=len(results))"
    )
    pert_codes, pert_feats, pert_labels = _perturb_templates(
        long_templates, "Yes", extra_long
    )
    long_codes.extend(pert_codes)
    long_features.extend(pert_feats)
    long_labels.extend(pert_labels)

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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from imblearn.over_sampling import SMOTE

# Magic numbers moved to named constants
SIGMA = 0.25
NUM_LINES_NOISE_STD = 12.0
N_NOISE_COPIES = 4

def _load_dataset() -> pd.DataFrame:
    """Load the dataset from CSV or generate a synthetic one.

    Returns:
        pd.DataFrame: Combined feature and target DataFrame.
    """
    if CODEWORKS_CSV.exists():
        df_raw = pd.read_csv(str(CODEWORKS_CSV), encoding="latin-1")
        df_raw = df_raw[
            ["code_section", "Approximate Number of Lines", "Long Method"]
        ].copy()
        df_raw.columns = ["code_section", "num_lines", "long_method"]
        print(f"\nLoaded {len(df_raw)} rows from dataset.csv")
        feats = df_raw["code_section"].apply(extract_code_features)
        return pd.concat([feats, df_raw[["long_method"]]], axis=1)
    return generate_synthetic_dataset()


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Encode target and extract feature matrix.

    Args:
        df: DataFrame containing features and target column.

    Returns:
        Tuple of (feature matrix, encoded target array).
    """
    le = LabelEncoder()
    y = le.fit_transform(df["long_method"])  # No=0, Yes=1
    print(f"Class distribution: No={sum(y==0)}, Yes={sum(y==1)}")
    X = df[SAFE_FEATURES].copy()
    print(f"Feature matrix: {X.shape}")
    return X.values, y


def _augment_training_data(
    X_train: np.ndarray, y_train: np.ndarray, feature_stds: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy copies of the training set and apply SMOTE.

    Args:
        X_train: Original training feature matrix.
        y_train: Original training labels.
        feature_stds: Standard deviation of each feature column.

    Returns:
        Augmented feature matrix and label array.
    """
    binary_idx = [SAFE_FEATURES.index(col) for col in BINARY_COLS]
    num_lines_idx = SAFE_FEATURES.index("num_lines")

    noisy_X = []
    noisy_y = []
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
        noisy_X.append(X_noisy)
        noisy_y.append(y_train.copy())

    X_aug = np.vstack([X_train] + noisy_X)
    y_aug = np.concatenate([y_train] + noisy_y)

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
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        Trained sklearn Pipeline.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=1.0, probability=True, random_state=SEED)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def _cross_validate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    """Run stratified 5‑fold cross‑validation and print scores.

    Args:
        pipeline: Trained pipeline to evaluate.
        X: Feature matrix.
        y: Labels.
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


def _evaluate(pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Evaluate the pipeline on a held‑out test set."""
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test Accuracy (real data): {accuracy:.4f}")


def _save_model(pipeline: Pipeline) -> None:
    """Persist the trained SVM and scaler to disk."""
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


def main() -> None:
    """Entry point for training the Long‑Method code smell detector."""
    print("=" * 60)
    print("  Training Long-Method Code Smell Detector")
    print("=" * 60)

    df = _load_dataset()
    X, y = _prepare_features(df)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    feature_stds = pd.DataFrame(X, columns=SAFE_FEATURES).std()
    X_aug, y_aug = _augment_training_data(
        X_train, y_train, feature_stds
    )

    pipeline = _train_pipeline(X_aug, y_aug)
    _cross_validate(pipeline, X_aug, y_aug)
    _evaluate(pipeline, X_test, y_test)
    _save_model(pipeline)


if __name__ == "__main__":
    main()
