import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

import os
import sys
from pathlib import Path
from typing import List, Set, Tuple, Dict

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_PATH: Path = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS: Set[str] = {".py", ".java", ".cs", ".js", ".php"}
PYTHON_LANGUAGE: str = "python"
PYTHON_FILES_OUTPUT: str = "python_files.txt"
GITHUB_OUTPUT_ENV: str = "GITHUB_OUTPUT"


def load_model(model_path: Path) -> Tuple[object, object, object, List[str]]:
    """Load the language classifier model bundle.

    Args:
        model_path: Path to the serialized joblib bundle.

    Returns:
        A tuple containing the model, character-level TF‑IDF transformer,
        word-level TF‑IDF transformer, and list of class names.

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

    Skips hidden directories, any path containing a "models" component,
    and files without a supported extension.

    Args:
        file_path: Path to the file.

    Returns:
        True if the file is a valid source file, False otherwise.
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
    model: object,
    tfidf_char: object,
    tfidf_word: object,
    class_names: List[str],
) -> str:
    """Predict the programming language of a file's content.

    Args:
        file_path: Path to the file (used for filename features).
        content: Full text content of the file.
        model: Trained classifier.
        tfidf_char: Character‑level TF‑IDF transformer.
        tfidf_word: Word‑level TF‑IDF transformer.
        class_names: Mapping from label indices to language names.

    Returns:
        Predicted language name.
    """
    line_count = len(content.splitlines())
    combined = f"{file_path.name.lower()}  {content}"

    x_char = tfidf_char.transform([combined])
    x_word = tfidf_word.transform([combined])
    x_lines = csr_matrix([[float(line_count)]])

    if hasattr(model, "n_features_in_") and model.n_features_in_ == x_char.shape[1] + x_word.shape[1]:
        features = hstack([x_char, x_word])
    else:
        features = hstack([x_char, x_word, x_lines])

    label_index = model.predict(features)[0]
    return class_names[label_index]


def scan_repository(
    root: Path,
    model: object,
    tfidf_char: object,
    tfidf_word: object,
    class_names: List[str],
) -> Tuple[bool, bool, Set[str], List[str]]:
    """Walk the repository and collect language detection results.

    Args:
        root: Directory to start scanning from.
        model: Trained classifier.
        tfidf_char: Character‑level TF‑IDF transformer.
        tfidf_word: Word‑level TF‑IDF transformer.
        class_names: Mapping from label indices to language names.

    Returns:
        A tuple:
            - python_detected: True if any Python file was found.
            - other_detected: True if any non‑Python language was found.
            - other_languages: Set of detected non‑Python language names.
            - python_files: List of paths (as strings) to Python files.
    """
    python_detected = False
    other_detected = False
    other_languages: Set[str] = set()
    python_files: List[str] = []

    for file_path in root.rglob("*"):
        if not is_valid_source(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        try:
            prediction = predict_language(
                file_path, content, model, tfidf_char, tfidf_word, class_names
            )
        except Exception as e:
            print(f"Error predicting language for {file_path}: {e}")
            continue

        if prediction.lower() == PYTHON_LANGUAGE:
            python_detected = True
            python_files.append(str(file_path))
        else:
            other_detected = True
            other_languages.add(str(prediction))
            print(f"Detected other language ({prediction}) in file: {file_path}")

    return python_detected, other_detected, other_languages, python_files


def write_python_file_list(python_files: List[str], output_path: str = PYTHON_FILES_OUTPUT) -> None:
    """Write the list of Python file paths to a text file.

    Args:
        python_files: List of file paths.
        output_path: Destination file name.
    """
    with open(output_path, "w", encoding="utf-8") as pf:
        for path in python_files:
            pf.write(f"{path}\n")


def update_github_output(output_file: str, entries: Dict[str, str]) -> None:
    """Append key‑value pairs to the GitHub Actions output file.

    Args:
        output_file: Path to the GITHUB_OUTPUT file.
        entries: Mapping of keys to string values.
    """
    with open(output_file, "a", encoding="utf-8") as f:
        for key, value in entries.items():
            f.write(f"{key}={value}\n")


def main() -> None:
    """Entry point for language detection and pipeline decision.

    Scans the repository, determines whether Python files are present,
    and writes appropriate signals for downstream GitHub Actions steps.
    """
    try:
        model, tfidf_char, tfidf_word, class_names = load_model(MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("Loading language classifier bundle...")
    print("Scanning files for language detection...")

    python_detected, other_detected, other_languages, python_files = scan_repository(
        Path("."), model, tfidf_char, tfidf_word, class_names
    )

    github_output = os.getenv(GITHUB_OUTPUT_ENV)
    other_langs_str = ", ".join(other_languages)

    if other_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {other_langs_str}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {other_langs_str}")
        write_python_file_list(python_files)
        if github_output:
            update_github_output(
                github_output,
                {
                    "proceed": "true",
                    "python_only_mode": "true",
                    "skipped_languages": other_langs_str,
                },
            )
    elif other_detected:
        print(f"Non-Python language detected ({other_langs_str}). No Python files found. Stopping system.")
        if github_output:
            update_github_output(
                github_output,
                {"proceed": "false", "detected_languages": other_langs_str},
            )
        sys.exit(1)
    elif python_detected:
        print("Only Python files detected. Proceeding...")
        if github_output:
            update_github_output(github_output, {"proceed": "true"})
    else:
        print("No readable source files found.")
        if github_output:
            update_github_output(github_output, {"proceed": "false"})


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
