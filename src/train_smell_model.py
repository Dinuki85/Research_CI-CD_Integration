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
from typing import List, Tuple

import numpy as np
import pandas as pd

SEED = 42
NUM_PERTURBATIONS = 3
SHORT_LABEL = "No"
LONG_LABEL = "Yes"
SHORT_COMMENT = "# extra comment\n    pass"
LONG_COMMENT = (
    "# additional logging\n"
    "    logger.info(f\"Processed batch\")\n"
    "    metrics.record(count=len(results))"
)


def _create_templates() -> Tuple[List[str], List[str]]:
    """Return short‑function and long‑function template lists."""
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


def _extract_features(codes: List[str]) -> List[dict]:
    """Apply ``extract_code_features`` to each code snippet."""
    return [extract_code_features(code) for code in codes]


def _augment_codes(
    base_templates: List[str],
    label: str,
    comment: str,
) -> Tuple[List[str], List[dict], List[str]]:
    """Create perturbed variants of ``base_templates``."""
    codes: List[str] = []
    features: List[dict] = []
    labels: List[str] = []

    for _ in range(NUM_PERTURBATIONS):
        for tmpl in base_templates:
            variant = f"{tmpl}\n    {comment}"
            codes.append(variant)
            features.append(extract_code_features(variant))
            labels.append(label)

    return codes, features, labels


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic training dataset when ``codeworks/dataset.csv`` is missing.

    Returns:
        pd.DataFrame: DataFrame with extracted features and a ``long_method`` label.

    The function builds short‑method and long‑method code examples, augments them
    with small perturbations, extracts numeric features via ``extract_code_features``,
    and combines everything into a single DataFrame.
    """
    print(
        "  codeworks/dataset.csv not found — generating synthetic training data"
    )

    short_templates, long_templates = _create_templates()

    short_feats = _extract_features(short_templates)
    long_feats = _extract_features(long_templates)

    short_codes = short_templates[:]
    long_codes = long_templates[:]

    short_labels = [SHORT_LABEL] * len(short_codes)
    long_labels = [LONG_LABEL] * len(long_codes)

    # Add perturbed short variants
    short_aug_codes, short_aug_feats, short_aug_labels = _augment_codes(
        short_templates, SHORT_LABEL, SHORT_COMMENT
    )
    short_codes.extend(short_aug_codes)
    short_feats.extend(short_aug_feats)
    short_labels.extend(short_aug_labels)

    # Add perturbed long variants
    long_aug_codes, long_aug_feats, long_aug_labels = _augment_codes(
        long_templates, LONG_LABEL, LONG_COMMENT
    )
    long_codes.extend(long_aug_codes)
    long_feats.extend(long_aug_feats)
    long_labels.extend(long_aug_labels)

    all_features = short_feats + long_feats
    all_labels = short_labels + long_labels

    synthetic_df = pd.DataFrame(all_features)
    synthetic_df["long_method"] = all_labels

    print(
        f"  Generated {len(synthetic_df)} synthetic training examples "
        f"(No={all_labels.count(SHORT_LABEL)}, "
        f"Yes={all_labels.count(LONG_LABEL)})"
    )
    return synthetic_df


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
from typing import Tuple


# Constants
HEADER_LINE = "=" * 60
SIGMA = 0.25
NUM_LINES_NOISE_STD = 12.0
N_NOISE_COPIES = 4


def _print_header() -> None:
    """Print a formatted header for the training run."""
    print(HEADER_LINE)
    print("  Training Long-Method Code Smell Detector")
    print(HEADER_LINE)


def _load_dataset() -> pd.DataFrame:
    """
    Load the dataset from ``CODEWORKS_CSV`` if it exists,
    otherwise generate a synthetic dataset.

    Returns:
        pd.DataFrame: Dataframe containing feature columns and the target.
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


def _encode_target(df: pd.DataFrame) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode the ``long_method`` column into numeric labels.

    Args:
        df: Dataframe that contains the ``long_method`` column.

    Returns:
        Tuple containing the encoded target array and the fitted encoder.
    """
    le = LabelEncoder()
    y = le.fit_transform(df["long_method"])
    print(
        f"Class distribution: No={int(np.sum(y == 0))}, Yes={int(np.sum(y == 1))}"
    )
    return y, le


def _augment_training_data(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create noisy copies of the training data and apply SMOTE for balancing.

    Args:
        X_train: Feature matrix of the real training set.
        y_train: Target vector of the real training set.

    Returns:
        Tuple of augmented feature matrix and balanced target vector.
    """
    feature_stds = pd.DataFrame(X_train, columns=SAFE_FEATURES).std()
    binary_indices = [SAFE_FEATURES.index(col) for col in BINARY_COLS]
    num_lines_index = SAFE_FEATURES.index("num_lines")

    noisy_features: list[np.ndarray] = []
    noisy_targets: list[np.ndarray] = []

    for _ in range(N_NOISE_COPIES):
        noise = rng.normal(
            loc=0.0,
            scale=(SIGMA * feature_stds).values,
            size=X_train.shape,
        )
        noise[:, num_lines_index] = rng.normal(
            0, NUM_LINES_NOISE_STD, size=len(X_train)
        )
        X_noisy = X_train + noise
        X_noisy[:, binary_indices] = np.clip(
            np.rint(X_noisy[:, binary_indices]), 0, 1
        )
        X_noisy = np.clip(X_noisy, 0, None)
        noisy_features.append(X_noisy)
        noisy_targets.append(y_train.copy())

    X_augmented = np.vstack([X_train] + noisy_features)
    y_augmented = np.concatenate([y_train] + noisy_targets)

    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X_augmented, y_augmented)

    print(
        f"Augmented train: {X_balanced.shape}, "
        f"classes: {np.bincount(y_balanced)}"
    )
    return X_balanced, y_balanced


def _train_classifier(
    X: np.ndarray, y: np.ndarray
) -> Pipeline:
    """
    Train an SVM classifier wrapped in a preprocessing pipeline.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        Trained ``Pipeline`` object.
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
    """
    Perform stratified 5‑fold cross‑validation and print mean/std scores.

    Args:
        pipeline: Trained pipeline to evaluate.
        X: Feature matrix.
        y: Target vector.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy"
    )
    f1_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="f1"
    )
    print(f"\nCV Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"CV F1 Score:  {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")


def _evaluate(pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate the pipeline on a held‑out test set and print accuracy.

    Args:
        pipeline: Trained pipeline.
        X_test: Test feature matrix.
        y_test: Test target vector.
    """
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test Accuracy (real data): {accuracy:.4f}")


def _save_artifacts(pipeline: Pipeline) -> None:
    """
    Persist the trained SVM and scaler to ``MODELS_DIR``.

    Args:
        pipeline: Trained pipeline containing ``svm`` and ``scaler`` steps.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        pipeline.named_steps["svm"], str(MODELS_DIR / "smell_svm.pkl")
    )
    joblib.dump(
        pipeline.named_steps["scaler"], str(MODELS_DIR / "smell_scaler.pkl")
    )
    print(f"\n✅ Model saved to: {MODELS_DIR / 'smell_svm.pkl'}")
    print(f"✅ Scaler saved to: {MODELS_DIR / 'smell_scaler.pkl'}")


def main() -> None:
    """Orchestrate the entire training pipeline for the Long‑Method smell detector."""
    _print_header()

    df = _load_dataset()
    y, _ = _encode_target(df)

    X = df[SAFE_FEATURES].copy()
    print(f"Feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    X_balanced, y_balanced = _augment_training_data(X_train, y_train)

    classifier = _train_classifier(X_balanced, y_balanced)

    _cross_validate(classifier, X_balanced, y_balanced)

    _evaluate(classifier, X_test, y_test)

    _save_artifacts(classifier)


if __name__ == "__main__":
    main()
