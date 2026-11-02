import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

import os
import sys
from pathlib import Path
from typing import List, Set, Tuple, Optional

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_REL_PATH = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS = {".py", ".java", ".cs", ".js", ".php"}
PYTHON_LANGUAGE = "python"
GITHUB_OUTPUT_ENV = "GITHUB_OUTPUT"
PYTHON_FILES_LIST = "python_files.txt"


def load_model_bundle(model_path: Path) -> Tuple:
    """Load the machine‑learning model bundle.

    Args:
        model_path: Path to the joblib file containing the model and vectorizers.

    Returns:
        A tuple containing (model, tfidf_char, tfidf_word, class_names).

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    bundle = joblib.load(model_path)
    return (
        bundle["model"],
        bundle["tfidf_char"],
        bundle["tfidf_word"],
        bundle["class_names"],
    )


def predict_language(
    file_path: Path,
    tfidf_char,
    tfidf_word,
    model,
    class_names: List[str],
) -> str:
    """Predict the programming language of a file.

    Args:
        file_path: Path to the source file.
        tfidf_char: Character‑level TF‑IDF transformer.
        tfidf_word: Word‑level TF‑IDF transformer.
        model: Trained classifier.
        class_names: Mapping from numeric label to language name.

    Returns:
        Predicted language name.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = len(content.splitlines())
    combined = f"{file_path.name.lower()}  {content}"

    x_char = tfidf_char.transform([combined])
    x_word = tfidf_word.transform([combined])
    x_lines = csr_matrix([[float(lines)]])

    if hasattr(model, "n_features_in_") and model.n_features_in_ == x_char.shape[1] + x_word.shape[1]:
        features = hstack([x_char, x_word])
    else:
        features = hstack([x_char, x_word, x_lines])

    pred_label = model.predict(features)[0]
    return class_names[pred_label]


def scan_repository(
    tfidf_char,
    tfidf_word,
    model,
    class_names: List[str],
) -> Tuple[bool, bool, List[str], Set[str]]:
    """Traverse the repository and detect languages.

    Returns:
        A tuple:
            - python_detected: True if any Python file is found.
            - other_language_detected: True if any non‑Python language is found.
            - python_files: List of paths (as strings) of detected Python files.
            - other_languages: Set of detected non‑Python language names.
    """
    python_detected = False
    other_language_detected = False
    python_files: List[str] = []
    other_languages: Set[str] = set()

    for file_path in Path(".").rglob("*"):
        if not file_path.is_file():
            continue
        if any(part.startswith(".") for part in file_path.parts):
            continue
        if "models" in file_path.parts:
            continue
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        try:
            prediction = predict_language(
                file_path, tfidf_char, tfidf_word, model, class_names
            )
            if prediction.lower() == PYTHON_LANGUAGE:
                python_detected = True
                python_files.append(str(file_path))
            else:
                other_language_detected = True
                other_languages.add(str(prediction))
                print(f"Detected other language ({prediction}) in file: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return python_detected, other_language_detected, python_files, other_languages


def write_python_file_list(python_files: List[str], output_path: str = PYTHON_FILES_LIST) -> None:
    """Write the list of Python file paths to a text file."""
    with open(output_path, "w", encoding="utf-8") as pf:
        for path in python_files:
            pf.write(f"{path}\n")


def update_github_output(github_output_path: Optional[str], **kwargs: str) -> None:
    """Append key=value pairs to the GitHub Actions output file."""
    if not github_output_path:
        return
    with open(github_output_path, "a", encoding="utf-8") as f:
        for key, value in kwargs.items():
            f.write(f"{key}={value}\n")


def main() -> None:
    """Entry point for language detection.

    Scans the repository, determines if Python files are present,
    and writes appropriate flags for downstream GitHub Actions steps.
    """
    try:
        model, tfidf_char, tfidf_word, class_names = load_model_bundle(MODEL_REL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("Scanning files for language detection...")
    python_detected, other_language_detected, python_files, other_languages = scan_repository(
        tfidf_char, tfidf_word, model, class_names
    )

    github_output = os.getenv(GITHUB_OUTPUT_ENV)
    other_langs_str = ", ".join(other_languages)

    if other_language_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {other_langs_str}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {other_langs_str}")
        write_python_file_list(python_files)
        update_github_output(
            github_output,
            proceed="true",
            python_only_mode="true",
            skipped_languages=other_langs_str,
        )
    elif other_language_detected:
        print(f"Non-Python language detected ({other_langs_str}). No Python files found. Stopping system.")
        update_github_output(
            github_output,
            proceed="false",
            detected_languages=other_langs_str,
        )
        sys.exit(1)
    elif python_detected:
        print("Only Python files detected. Proceeding...")
        update_github_output(github_output, proceed="true")
    else:
        print("No readable source files found.")
        update_github_output(github_output, proceed="false")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
