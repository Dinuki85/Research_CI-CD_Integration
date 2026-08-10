import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

import os
import sys
from pathlib import Path
from typing import List, Set, Tuple

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_PATH = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS = {".py", ".java", ".cs", ".js", ".php"}
PYTHON_FILES_OUTPUT = "python_files.txt"


def load_model(model_path: Path) -> Tuple[object, object, object, List[str]]:
    """Load the language classifier model bundle.

    Args:
        model_path: Path to the joblib file containing the model bundle.

    Returns:
        A tuple of (model, tfidf_char, tfidf_word, class_names).

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


def is_hidden_path(path: Path) -> bool:
    """Check whether any part of the path is hidden (starts with a dot)."""
    return any(part.startswith(".") for part in path.parts)


def should_process_file(file_path: Path) -> bool:
    """Determine if a file should be scanned for language detection.

    Skips hidden files, files inside a 'models' directory, and files with
    extensions not listed in VALID_EXTENSIONS.

    Args:
        file_path: Path to the file.

    Returns:
        True if the file should be processed, False otherwise.
    """
    return (
        file_path.is_file()
        and not is_hidden_path(file_path)
        and "models" not in file_path.parts
        and file_path.suffix.lower() in VALID_EXTENSIONS
    )


def predict_language(
    file_path: Path,
    tfidf_char,
    tfidf_word,
    model,
    class_names: List[str],
) -> str:
    """Predict the programming language of a file using the loaded model.

    Args:
        file_path: Path to the source file.
        tfidf_char: Character-level TF-IDF transformer.
        tfidf_word: Word-level TF-IDF transformer.
        model: Trained classifier model.
        class_names: List mapping label indices to language names.

    Returns:
        Predicted language name as a string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

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
    return str(class_names[pred_label])


def write_python_file_list(python_files: List[str], output_path: str = PYTHON_FILES_OUTPUT) -> None:
    """Write the list of detected Python file paths to a text file."""
    with open(output_path, "w", encoding="utf-8") as pf:
        for path in python_files:
            pf.write(f"{path}\n")


def append_github_output(github_output_path: str, lines: List[str]) -> None:
    """Append key/value lines to the GitHub Actions output file."""
    with open(github_output_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main() -> None:
    """Entry point for language detection and repository classification.

    Scans the repository for source files, determines whether Python files
    are present alone or alongside other languages, and writes appropriate
    flags for downstream GitHub Actions steps.
    """
    try:
        model, tfidf_char, tfidf_word, class_names = load_model(MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("Scanning files for language detection...")

    python_detected = False
    other_language_detected = False
    detected_other_languages: Set[str] = set()
    python_files: List[str] = []

    for file_path in Path(".").rglob("*"):
        if not should_process_file(file_path):
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
            other_language_detected = True
            detected_other_languages.add(prediction)
            print(f"Detected other language ({prediction}) in file: {file_path}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    langs = ", ".join(detected_other_languages)

    if other_language_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {langs}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {langs}")
        write_python_file_list(python_files)
        if github_output:
            append_github_output(
                github_output,
                [
                    "proceed=true",
                    "python_only_mode=true",
                    f"skipped_languages={langs}",
                ],
            )
    elif other_language_detected:
        print(f"Non-Python language detected ({langs}). No Python files found. Stopping system.")
        if github_output:
            append_github_output(
                github_output,
                [
                    "proceed=false",
                    f"detected_languages={langs}",
                ],
            )
        sys.exit(1)
    elif python_detected:
        print("Only Python files detected. Proceeding...")
        if github_output:
            append_github_output(github_output, ["proceed=true"])
    else:
        print("No readable source files found.")
        if github_output:
            append_github_output(github_output, ["proceed=false"])


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
