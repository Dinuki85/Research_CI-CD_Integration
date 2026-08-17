import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

import os
import sys
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_REL_PATH = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS = {".py", ".java", ".cs", ".js", ".php"}
GITHUB_OUTPUT_ENV = "GITHUB_OUTPUT"
PYTHON_FILES_LIST = "python_files.txt"


def load_model(model_path: Path) -> Tuple[Any, Any, Any, List[str]]:
    """Load the language classifier model bundle.

    Args:
        model_path: Path to the joblib model file.

    Returns:
        A tuple containing the model, character TF‑IDF transformer,
        word TF‑IDF transformer, and list of class names.

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


def is_valid_source_file(file_path: Path) -> bool:
    """Determine whether a file should be processed.

    Skips hidden directories, the models directory, and files with
    unsupported extensions.

    Args:
        file_path: Path to the candidate file.

    Returns:
        True if the file is a readable source file, False otherwise.
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
    content: str,
    model: Any,
    tfidf_char: Any,
    tfidf_word: Any,
    class_names: List[str],
) -> str:
    """Predict the programming language of a file's content.

    Args:
        file_path: Path to the file (used for filename inclusion).
        content: Full text content of the file.
        model: Trained classifier.
        tfidf_char: Character‑level TF‑IDF transformer.
        tfidf_word: Word‑level TF‑IDF transformer.
        class_names: Mapping from label indices to language names.

    Returns:
        Predicted language name.
    """
    lines = len(content.splitlines())
    combined = f"{file_path.name.lower()}  {content}"
    x_char = tfidf_char.transform([combined])
    x_word = tfidf_word.transform([combined])
    x_lines = csr_matrix([[float(lines)]])

    if hasattr(model, "n_features_in_") and model.n_features_in_ == x_char.shape[1] + x_word.shape[1]:
        X = hstack([x_char, x_word])
    else:
        X = hstack([x_char, x_word, x_lines])

    pred_label = model.predict(X)[0]
    return class_names[pred_label]


def scan_repository(
    model: Any,
    tfidf_char: Any,
    tfidf_word: Any,
    class_names: List[str],
) -> Tuple[bool, bool, Set[str], List[str]]:
    """Walk the repository and detect languages.

    Returns:
        Tuple containing:
            - python_detected: Whether any Python file was found.
            - other_language_detected: Whether any non‑Python language was found.
            - detected_other_languages: Set of detected non‑Python language names.
            - python_files: List of paths to Python files.
    """
    python_detected = False
    other_language_detected = False
    detected_other_languages: Set[str] = set()
    python_files: List[str] = []

    for file_path in Path(".").rglob("*"):
        if not is_valid_source_file(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            prediction = predict_language(
                file_path, content, model, tfidf_char, tfidf_word, class_names
            )
            if prediction.lower() == "python":
                python_detected = True
                python_files.append(str(file_path))
            else:
                other_language_detected = True
                detected_other_languages.add(str(prediction))
                print(f"Detected other language ({prediction}) in file: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return python_detected, other_language_detected, detected_other_languages, python_files


def write_python_file_list(python_files: List[str]) -> None:
    """Write the list of Python file paths to a text file."""
    with open(PYTHON_FILES_LIST, "w", encoding="utf-8") as pf:
        for path in python_files:
            pf.write(f"{path}\n")


def update_github_output(
    github_output_path: str,
    **kwargs: str,
) -> None:
    """Append key‑value pairs to the GitHub Actions output file."""
    with open(github_output_path, "a", encoding="utf-8") as f:
        for key, value in kwargs.items():
            f.write(f"{key}={value}\n")


def main() -> None:
    """Entry point for language detection in a repository.

    Detects whether the repository contains Python files, other languages,
    or a mixture, and writes appropriate signals for downstream CI steps.
    """
    try:
        model, tfidf_char, tfidf_word, class_names = load_model(MODEL_REL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("Scanning files for language detection...")
    (
        python_detected,
        other_language_detected,
        detected_other_languages,
        python_files,
    ) = scan_repository(model, tfidf_char, tfidf_word, class_names)

    github_output = os.getenv(GITHUB_OUTPUT_ENV)
    langs = ", ".join(detected_other_languages)

    if other_language_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {langs}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {langs}")
        write_python_file_list(python_files)
        if github_output:
            update_github_output(
                github_output,
                proceed="true",
                python_only_mode="true",
                skipped_languages=langs,
            )
        return

    if other_language_detected:
        print(f"Non-Python language detected ({langs}). No Python files found. Stopping system.")
        if github_output:
            update_github_output(github_output, proceed="false", detected_languages=langs)
        sys.exit(1)

    if python_detected:
        print("Only Python files detected. Proceeding...")
        if github_output:
            update_github_output(github_output, proceed="true")
        return

    print("No readable source files found.")
    if github_output:
        update_github_output(github_output, proceed="false")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
