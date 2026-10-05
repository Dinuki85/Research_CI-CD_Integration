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

SEED = 42  # default seed, replace with actual value if defined elsewhere
NUM_PERTURBATIONS = 3
SHORT_LABEL = "No"
LONG_LABEL = "Yes"


def _get_short_templates() -> List[str]:
    """Return a list of short/clean function code templates."""
    return [
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


def _get_long_templates() -> List[str]:
    """Return a list of long/bad‑smell function code templates."""
    return [
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


def _extract_features(
    code_snippets: List[str],
) -> List[dict]:
    """Extract feature dictionaries for a list of code snippets."""
    return [extract_code_features(snippet) for snippet in code_snippets]


def _add_variations(
    base_codes: List[str],
    base_labels: List[str],
    templates: List[str],
    label: str,
    perturbations: int,
    extra_comment: str,
) -> None:
    """
    Extend ``base_codes`` and ``base_labels`` with perturbed versions of ``templates``.

    Args:
        base_codes: List to which new code strings will be appended.
        base_labels: Parallel list of labels to extend.
        templates: Original code templates.
        label: Label to assign to all generated variants.
        perturbations: Number of times each template is perturbed.
        extra_comment: Code snippet to append for each perturbation.
    """
    for _ in range(perturbations):
        for template in templates:
            variant = f"{template}\n    # extra comment\n    pass"
            if extra_comment:
                variant = f"{template}\n    # additional logging\n    {extra_comment}"
            base_codes.append(variant)
            base_labels.append(label)


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic training dataset when ``codeworks/dataset.csv`` is missing.

    The dataset contains short (clean) and long (smelly) Python function snippets
    with corresponding ``long_method`` labels.  Code snippets are augmented with
    small perturbations to increase the training size.

    Returns:
        pandas.DataFrame: DataFrame where each row holds extracted code features
        and a ``long_method`` column with values ``"Yes"`` or ``"No"``.
    """
    print(
        "  codeworks/dataset.csv not found — generating synthetic training data"
    )
    rng = np.random.default_rng(SEED)  # noqa: F841, kept for compatibility

    short_templates = _get_short_templates()
    long_templates = _get_long_templates()

    # Base code collections
    short_codes: List[str] = short_templates.copy()
    long_codes: List[str] = long_templates.copy()

    short_features = _extract_features(short_codes)
    long_features = _extract_features(long_codes)

    short_labels = [SHORT_LABEL] * len(short_codes)
    long_labels = [LONG_LABEL] * len(long_codes)

    # Create perturbed variants
    _add_variations(
        base_codes=short_codes,
        base_labels=short_labels,
        templates=short_templates,
        label=SHORT_LABEL,
        perturbations=NUM_PERTURBATIONS,
        extra_comment="",
    )
    _add_variations(
        base_codes=long_codes,
        base_labels=long_labels,
        templates=long_templates,
        label=LONG_LABEL,
        perturbations=NUM_PERTURBATIONS,
        extra_comment='logger.info(f"Processed batch")\n    metrics.record(count=len(results))',
    )

    # Extract features for the new variants
    short_features.extend(_extract_features(short_codes[len(short_features) :]))
    long_features.extend(_extract_features(long_codes[len(long_features) :]))

    all_features = short_features + long_features
    all_labels = short_labels + long_labels

    synthetic_df = pd.DataFrame(all_features)
    synthetic_df["long_method"] = all_labels

    print(
        f"  Generated {len(synthetic_df)} synthetic training examples "
        f"(No={all_labels.count(SHORT_LABEL)}, "
        f"Yes={all_labels.count(LONG_LABEL)})"
    )
    return synthetic_df


import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from pathlib import Path
from typing import Tuple

# Magic numbers
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


def _augment_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_stds: pd.Series,
    binary_idx: list[int],
    num_lines_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create noisy copies of the training data and apply SMOTE.

    Args:
        X_train: Original training feature matrix.
        y_train: Corresponding labels.
        feature_stds: Standard deviation of each feature column.
        binary_idx: Indices of binary features to be rounded.
        num_lines_idx: Index of the ``num_lines`` feature.

    Returns:
        Tuple of augmented feature matrix and label vector.
    """
    X_noise_list: list[np.ndarray] = []
    y_noise_list: list[np.ndarray] = []

    for _ in range(N_NOISE_COPIES):
        noise = rng.normal(
            loc=0.0,
            scale=(SIGMA * feature_stds).values,
            size=X_train.shape,
        )
        noise[:, num_lines_idx] = rng.normal(
            0,
            NUM_LINES_NOISE_STD,
            size=len(X_train),
        )
        X_noisy = X_train + noise
        X_noisy[:, binary_idx] = np.clip(
            np.round(X_noisy[:, binary_idx]), 0, 1
        )
        X_noisy = np.clip(X_noisy, 0, None)
        X_noise_list.append(X_noisy)
        y_noise_list.append(y_train.copy())

    X_augmented = np.vstack([X_train] + X_noise_list)
    y_augmented = np.concatenate([y_train] + y_noise_list)

    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(
        X_augmented, y_augmented
    )
    return X_resampled, y_resampled


def _train_pipeline(
    X: np.ndarray, y: np.ndarray
) -> Pipeline:
    """Train an SVM pipeline on the provided data.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        Trained scikit‑learn Pipeline.
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
    """Perform stratified 5‑fold cross‑validation and print scores."""
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )
    cv_acc = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy"
    )
    cv_f1 = cross_val_score(
        pipeline, X, y, cv=cv, scoring="f1"
    )
    print(f"\nCV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"CV F1 Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")


def _save_artifacts(pipeline: Pipeline) -> None:
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
    """Train the Long‑Method code smell detector and persist the model.

    Steps:
        1. Load or generate the dataset.
        2. Encode labels and select safe features.
        3. Split into stratified train/test sets.
        4. Augment training data with noise and SMOTE.
        5. Train an SVM pipeline.
        6. Report cross‑validation metrics.
        7. Evaluate on the real test set.
        8. Save the trained model and scaler.
    """
    print("=" * 60)
    print("  Training Long-Method Code Smell Detector")
    print("=" * 60)

    df_all = _load_dataset()

    le = LabelEncoder()
    y = le.fit_transform(df_all["long_method"])
    print(f"Class distribution: No={sum(y==0)}, Yes={sum(y==1)}")

    X = df_all[SAFE_FEATURES].copy()
    print(f"Feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    feature_stds = X.std()
    binary_idx = [SAFE_FEATURES.index(col) for col in BINARY_COLS]
    num_lines_idx = SAFE_FEATURES.index("num_lines")

    X_aug, y_aug = _augment_training_data(
        X_train, y_train, feature_stds, binary_idx, num_lines_idx
    )
    print(
        f"Augmented train: {X_aug.shape}, "
        f"classes: {np.bincount(y_aug)}"
    )

    pipeline = _train_pipeline(X_aug, y_aug)

    _cross_validate(pipeline, X_aug, y_aug)

    test_acc = pipeline.score(X_test, y_test)
    print(f"Test Accuracy (real data): {test_acc:.4f}")

    _save_artifacts(pipeline)


if __name__ == "__main__":
    main()
