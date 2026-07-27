import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

import os
import sys
from pathlib import Path
from typing import Any, List, Set, Tuple

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_PATH: Path = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS: Set[str] = {".py", ".java", ".cs", ".js", ".php"}
GITHUB_OUTPUT_ENV: str = "GITHUB_OUTPUT"


def load_classifier(model_path: Path) -> Tuple[Any, Any, Any, List[str]]:
    """Load the language classifier bundle.

    Args:
        model_path: Path to the joblib file containing the model and vectorizers.

    Returns:
        A tuple containing the model, character TF‑IDF vectorizer,
        word TF‑IDF vectorizer, and list of class names.

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


def is_valid_source(file_path: Path) -> bool:
    """Determine whether a file should be processed.

    Skips hidden directories, the `models` directory, and files with
    extensions not listed in ``VALID_EXTENSIONS``.

    Args:
        file_path: Path to the file to evaluate.

    Returns:
        True if the file is a candidate for language detection.
    """
    if not file_path.is_file():
        return False
    if any(part.startswith(".") for part in file_path.parts):
        return False
    if "models" in file_path.parts:
        return False
    return file_path.suffix.lower() in VALID_EXTENSIONS


def predict_language(
    file_path: Path,
    tfidf_char: Any,
    tfidf_word: Any,
    model: Any,
    class_names: List[str],
) -> str:
    """Predict the programming language of a source file.

    Args:
        file_path: Path to the source file.
        tfidf_char: Character level TF‑IDF vectorizer.
        tfidf_word: Word level TF‑IDF vectorizer.
        model: Trained classifier.
        class_names: Mapping from numeric label to language name.

    Returns:
        Predicted language name.

    Raises:
        Exception: Propagates any error encountered while reading or processing the file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    line_count = len(content.splitlines())
    combined = f"{file_path.name.lower()}  {content}"

    x_char = tfidf_char.transform([combined])
    x_word = tfidf_word.transform([combined])
    x_lines = csr_matrix([[float(line_count)]])

    if hasattr(model, "n_features_in_") and model.n_features_in_ == x_char.shape[1] + x_word.shape[1]:
        features = hstack([x_char, x_word])
    else:
        features = hstack([x_char, x_word, x_lines])

    pred_label = model.predict(features)[0]
    return class_names[pred_label]


def scan_repository(
    tfidf_char: Any,
    tfidf_word: Any,
    model: Any,
    class_names: List[str],
) -> Tuple[bool, bool, Set[str], List[str]]:
    """Walk the repository and collect language detection results.

    Returns:
        A tuple containing:
            - python_detected: True if any Python file was found.
            - other_detected: True if any non‑Python language was found.
            - other_languages: Set of detected non‑Python language names.
            - python_files: List of paths (as strings) to Python files.
    """
    python_detected = False
    other_detected = False
    other_languages: Set[str] = set()
    python_files: List[str] = []

    for file_path in Path(".").rglob("*"):
        if not is_valid_source(file_path):
            continue

        try:
            prediction = predict_language(file_path, tfidf_char, tfidf_word, model, class_names)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        if prediction.lower() == "python":
            python_detected = True
            python_files.append(str(file_path))
        else:
            other_detected = True
            other_languages.add(str(prediction))
            print(f"Detected other language ({prediction}) in file: {file_path}")

    return python_detected, other_detected, other_languages, python_files


def write_github_output(lines: List[str]) -> None:
    """Append key‑value pairs to the GitHub Actions output file if defined."""
    output_path = os.environ.get(GITHUB_OUTPUT_ENV)
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main() -> None:
    """Entry point for language detection and pipeline decision making.

    The script loads a pre‑trained classifier, scans the repository for source
    files, and determines whether to proceed based on the presence of Python
    files and/or other languages. Results are communicated via the
    ``GITHUB_OUTPUT`` environment variable and auxiliary files.
    """
    try:
        model, tfidf_char, tfidf_word, class_names = load_classifier(MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("Scanning files for language detection...")
    python_detected, other_detected, other_languages, python_files = scan_repository(
        tfidf_char, tfidf_word, model, class_names
    )
    langs_str = ", ".join(other_languages)

    if other_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {langs_str}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {langs_str}")
        with open("python_files.txt", "w", encoding="utf-8") as pf:
            for path in python_files:
                pf.write(f"{path}\n")
        write_github_output(
            [
                "proceed=true",
                "python_only_mode=true",
                f"skipped_languages={langs_str}",
            ]
        )
    elif other_detected:
        print(f"Non-Python language detected ({langs_str}). No Python files found. Stopping system.")
        write_github_output(
            [
                "proceed=false",
                f"detected_languages={langs_str}",
            ]
        )
        sys.exit(1)
    elif python_detected:
        print("Only Python files detected. Proceeding...")
        write_github_output(["proceed=true"])
    else:
        print("No readable source files found.")
        write_github_output(["proceed=false"])


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
