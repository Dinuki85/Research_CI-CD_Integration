import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

import os
import sys
from pathlib import Path
from typing import Any, List, Set, Tuple, Optional

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_REL_PATH = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS = {".py", ".java", ".cs", ".js", ".php"}
GITHUB_OUTPUT_ENV = "GITHUB_OUTPUT"
PYTHON_FILES_LIST = "python_files.txt"


def load_model_bundle(model_path: Path) -> Tuple[Any, Any, Any, List[str]]:
    """Load the language classifier model bundle.

    Args:
        model_path: Path to the joblib file containing the model and related objects.

    Returns:
        A tuple containing the model, character-level TF-IDF transformer,
        word-level TF-IDF transformer, and the list of class names.

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


def process_file(
    file_path: Path,
    model: Any,
    tfidf_char: Any,
    tfidf_word: Any,
    class_names: List[str],
) -> Tuple[bool, Optional[str], Optional[Exception]]:
    """Predict the language of a single file.

    Args:
        file_path: Path to the source file.
        model: Trained classifier model.
        tfidf_char: Character-level TF-IDF transformer.
        tfidf_word: Word-level TF-IDF transformer.
        class_names: Mapping from model output indices to language names.

    Returns:
        A tuple (is_python, other_language, error):
            - is_python: True if the file is detected as Python.
            - other_language: Detected language name if not Python, else None.
            - error: Exception instance if processing failed, else None.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
    except Exception as exc:
        return False, None, exc

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
    prediction = class_names[pred_label]

    if str(prediction).lower() == "python":
        return True, None, None
    return False, str(prediction), None


def scan_repository(
    model: Any,
    tfidf_char: Any,
    tfidf_word: Any,
    class_names: List[str],
) -> Tuple[bool, bool, List[str], Set[str]]:
    """Traverse the repository and detect languages.

    Returns:
        A tuple (python_detected, other_language_detected, python_files, other_languages).
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

        is_python, other_lang, error = process_file(
            file_path, model, tfidf_char, tfidf_word, class_names
        )
        if error:
            print(f"Error processing {file_path}: {error}")
            continue

        if is_python:
            python_detected = True
            python_files.append(str(file_path))
        else:
            other_language_detected = True
            other_languages.add(other_lang or "")
            print(f"Detected other language ({other_lang}) in file: {file_path}")

    return python_detected, other_language_detected, python_files, other_languages


def write_python_file_list(python_files: List[str], filename: str = PYTHON_FILES_LIST) -> None:
    """Write the list of detected Python file paths to a file."""
    with open(filename, "w", encoding="utf-8") as pf:
        for path in python_files:
            pf.write(f"{path}\n")


def append_to_github_output(output_path: str, **kwargs: str) -> None:
    """Append key=value lines to the GitHub Actions output file."""
    with open(output_path, "a", encoding="utf-8") as f:
        for key, value in kwargs.items():
            f.write(f"{key}={value}\n")


def decide_and_act(
    python_detected: bool,
    other_language_detected: bool,
    python_files: List[str],
    other_languages: Set[str],
) -> None:
    """Determine the next steps based on detection results and exit if needed."""
    github_output = os.getenv(GITHUB_OUTPUT_ENV)
    langs_str = ", ".join(other_languages)

    if other_language_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {langs_str}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {langs_str}")
        write_python_file_list(python_files)
        if github_output:
            append_to_github_output(
                github_output,
                proceed="true",
                python_only_mode="true",
                skipped_languages=langs_str,
            )
        return

    if other_language_detected and not python_detected:
        print(f"Non-Python language detected ({langs_str}). No Python files found. Stopping system.")
        if github_output:
            append_to_github_output(
                github_output,
                proceed="false",
                detected_languages=langs_str,
            )
        sys.exit(1)

    if python_detected and not other_language_detected:
        print("Only Python files detected. Proceeding...")
        if github_output:
            append_to_github_output(github_output, proceed="true")
        return

    print("No readable source files found.")
    if github_output:
        append_to_github_output(github_output, proceed="false")
    return


def main() -> None:
    """Entry point for language detection in a repository.

    Loads the classifier model, scans source files, and writes GitHub Action
    outputs based on detected languages.
    """
    try:
        model, tfidf_char, tfidf_word, class_names = load_model_bundle(MODEL_REL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("Loading language classifier bundle...")
    print("Scanning files for language detection...")

    python_detected, other_language_detected, python_files, other_languages = scan_repository(
        model, tfidf_char, tfidf_word, class_names
    )

    decide_and_act(
        python_detected,
        other_language_detected,
        python_files,
        other_languages,
    )


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
