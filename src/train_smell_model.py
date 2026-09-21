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

# Constants
NUM_PERTURBATIONS = 3
EXTRA_COMMENT = "# extra comment"
ADDITIONAL_LOGGING = "# additional logging"


def _extract_features(codes: List[str]) -> List[dict]:
    """Extract code features for a list of code snippets."""
    return [extract_code_features(code) for code in codes]


def _generate_code_variations(
    templates: List[str],
    label: str,
    perturb: bool = False,
) -> Tuple[List[str], List[dict], List[str]]:
    """Create code snippets, their features, and labels.

    Args:
        templates: List of code template strings.
        label: Label to assign to each generated snippet.
        perturb: Whether to create perturbed variations.

    Returns:
        A tuple containing the list of code snippets, their feature dicts,
        and the corresponding labels.
    """
    codes: List[str] = []
    labels: List[str] = []

    for tmpl in templates:
        codes.append(tmpl)
        labels.append(label)

    if perturb:
        for _ in range(NUM_PERTURBATIONS):
            for tmpl in templates:
                padded = f"{tmpl}\n    {EXTRA_COMMENT}\n    pass"
                codes.append(padded)
                labels.append(label)

    features = _extract_features(codes)
    return codes, features, labels


def _extend_long_codes(
    templates: List[str],
    label: str,
) -> Tuple[List[str], List[dict], List[str]]:
    """Create long code snippets with additional logging extensions.

    Args:
        templates: List of long code template strings.
        label: Label to assign to each generated snippet.

    Returns:
        A tuple containing the list of extended code snippets,
        their feature dicts, and the corresponding labels.
    """
    codes: List[str] = []
    labels: List[str] = []

    for tmpl in templates:
        codes.append(tmpl)
        labels.append(label)

    for _ in range(NUM_PERTURBATIONS):
        for tmpl in templates:
            extended = (
                f"{tmpl}\n    {ADDITIONAL_LOGGING}\n"
                "    logger.info(f\"Processed batch\")\n"
                "    metrics.record(count=len(results))"
            )
            codes.append(extended)
            labels.append(label)

    features = _extract_features(codes)
    return codes, features, labels


def _create_dataframe(
    short_features: List[dict],
    short_labels: List[str],
    long_features: List[dict],
    long_labels: List[str],
) -> pd.DataFrame:
    """Combine short and long features into a single DataFrame.

    Args:
        short_features: Feature dictionaries for short functions.
        short_labels: Corresponding labels for short functions.
        long_features: Feature dictionaries for long functions.
        long_labels: Corresponding labels for long functions.

    Returns:
        DataFrame with features and a 'long_method' label column.
    """
    all_features = short_features + long_features
    all_labels = short_labels + long_labels
    df = pd.DataFrame(all_features)
    df["long_method"] = all_labels
    return df


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic training dataset when dataset.csv is missing.

    Returns:
        DataFrame containing synthetic code features and long_method labels.
    """
    print(
        "  codeworks/dataset.csv not found — generating synthetic training data"
    )
    rng = np.random.default_rng(SEED)

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

    _, short_features, short_labels = _generate_code_variations(
        short_templates, "No", perturb=True
    )
    _, long_features, long_labels = _extend_long_codes(
        long_templates, "Yes"
    )

    synthetic_df = _create_dataframe(
        short_features, short_labels, long_features, long_labels
    )
    print(
        f"  Generated {len(synthetic_df)} synthetic training examples "
        f"(No={short_labels.count('No')}, Yes={long_labels.count('Yes')})"
    )
    return synthetic_df


import pathlib
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Constants
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


def _prepare_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """Select safe features for model training.

    Args:
        df_all: Dataframe containing all features and the target column.

    Returns:
        pd.DataFrame: Feature matrix.
    """
    return df_all[SAFE_FEATURES].copy()


def _augment_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_stds: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy copies of the training data and apply SMOTE.

    Args:
        X_train: Original training feature matrix.
        y_train: Original training labels.
        feature_stds: Standard deviation of each feature in the full dataset.

    Returns:
        Tuple of augmented feature matrix and label vector.
    """
    binary_idx = [SAFE_FEATURES.index(col) for col in BINARY_COLS]
    num_lines_idx = SAFE_FEATURES.index("num_lines")

    noisy_X: list[np.ndarray] = []
    noisy_y: list[np.ndarray] = []

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
    X: np.ndarray, y: np.ndarray
) -> Pipeline:
    """Fit a scaling + SVM pipeline on the provided data.

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


def _cross_validate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    """Perform stratified 5‑fold cross‑validation and print metrics."""
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


def _evaluate_test_set(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Evaluate the model on the real test set and print accuracy."""
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test Accuracy (real data): {accuracy:.4f}")


def _save_model(pipeline: Pipeline) -> None:
    """Persist the trained SVM and scaler to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    svm_path = MODELS_DIR / "smell_svm.pkl"
    scaler_path = MODELS_DIR / "smell_scaler.pkl"
    joblib.dump(pipeline.named_steps["svm"], str(svm_path))
    joblib.dump(pipeline.named_steps["scaler"], str(scaler_path))
    print(f"\n✅ Model saved to: {svm_path}")
    print(f"✅ Scaler saved to: {scaler_path}")


def main() -> None:
    """Orchestrate data loading, training, evaluation, and model persistence."""
    print("=" * 60)
    print("  Training Long-Method Code Smell Detector")
    print("=" * 60)

    df_all = _load_dataset()
    le = LabelEncoder()
    y = le.fit_transform(df_all["long_method"])
    print(f"Class distribution: No={sum(y==0)}, Yes={sum(y==1)}")

    X_df = _prepare_features(df_all)
    print(f"Feature matrix: {X_df.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    feature_stds = X_df.std()
    X_aug, y_aug = _augment_training_data(
        X_train, y_train, feature_stds
    )

    pipeline = _train_pipeline(X_aug, y_aug)
    _cross_validate(pipeline, X_aug, y_aug)
    _evaluate_test_set(pipeline, X_test, y_test)
    _save_model(pipeline)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
