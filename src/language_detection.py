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

MODEL_PATH: Path = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS: Set[str] = {".py", ".java", ".cs", ".js", ".php"}
GITHUB_OUTPUT_ENV: str = "GITHUB_OUTPUT"
PYTHON_FILES_OUTPUT: Path = Path("python_files.txt")


def load_classifier_bundle(
    model_path: Path,
) -> Tuple[object, object, object, List[str]]:
    """Load the language classifier bundle from a joblib file.

    Args:
        model_path: Path to the joblib file containing the model and vectorizers.

    Returns:
        A tuple containing the model, character-level TF‑IDF transformer,
        word-level TF‑IDF transformer, and the list of class names.

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
    content: str,
    model: object,
    tfidf_char: object,
    tfidf_word: object,
    class_names: List[str],
) -> str:
    """Predict the programming language of a file's content.

    Args:
        file_path: Path to the file (used for filename inclusion in the prediction).
        content: The full text content of the file.
        model: Trained classifier model.
        tfidf_char: Character‑level TF‑IDF transformer.
        tfidf_word: Word‑level TF‑IDF transformer.
        class_names: Mapping from numeric label to language name.

    Returns:
        The predicted language name.
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


def is_valid_source_file(file_path: Path) -> bool:
    """Determine whether a file should be processed for language detection.

    Args:
        file_path: Path to the file.

    Returns:
        True if the file is a regular file, not hidden, not under a "models"
        directory, and has a supported extension.
    """
    if not file_path.is_file():
        return False
    if any(part.startswith(".") for part in file_path.parts):
        return False
    if "models" in file_path.parts:
        return False
    return file_path.suffix.lower() in VALID_EXTENSIONS


def scan_repository(
    model: object,
    tfidf_char: object,
    tfidf_word: object,
    class_names: List[str],
) -> Tuple[bool, bool, List[str], Set[str]]:
    """Walk the repository and detect languages present.

    Args:
        model: Trained classifier model.
        tfidf_char: Character‑level TF‑IDF transformer.
        tfidf_word: Word‑level TF‑IDF transformer.
        class_names: Mapping from numeric label to language name.

    Returns:
        A tuple containing:
            - python_detected (bool)
            - other_language_detected (bool)
            - python_files (list of file paths as strings)
            - detected_other_languages (set of language names)
    """
    python_detected = False
    other_language_detected = False
    python_files: List[str] = []
    detected_other_languages: Set[str] = set()

    for file_path in Path(".").rglob("*"):
        if not is_valid_source_file(file_path):
            continue

        try:
            with file_path.open("r", encoding="utf-8") as f:
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

    return python_detected, other_language_detected, python_files, detected_other_languages


def write_github_output(lines: List[str]) -> None:
    """Append lines to the GitHub Actions output file if the environment variable is set.

    Args:
        lines: List of strings to write (each should include a newline if desired).
    """
    github_output_path = os.getenv(GITHUB_OUTPUT_ENV)
    if not github_output_path:
        return
    with open(github_output_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line)


def main() -> None:
    """Entry point for language detection and GitHub Actions output handling.

    The script loads a pre‑trained classifier, scans the repository for source
    files, determines whether Python files are present alone or alongside other
    languages, and writes appropriate flags for downstream workflow steps.
    """
    try:
        model, tfidf_char, tfidf_word, class_names = load_classifier_bundle(MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("Scanning files for language detection...")
    (
        python_detected,
        other_language_detected,
        python_files,
        detected_other_languages,
    ) = scan_repository(model, tfidf_char, tfidf_word, class_names)

    langs_str = ", ".join(detected_other_languages)

    if other_language_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {langs_str}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {langs_str}")
        PYTHON_FILES_OUTPUT.write_text("\n".join(python_files) + "\n", encoding="utf-8")
        write_github_output(
            [
                "proceed=true\n",
                "python_only_mode=true\n",
                f"skipped_languages={langs_str}\n",
            ]
        )
    elif other_language_detected:
        print(f"Non-Python language detected ({langs_str}). No Python files found. Stopping system.")
        write_github_output(
            [
                "proceed=false\n",
                f"detected_languages={langs_str}\n",
            ]
        )
        sys.exit(1)
    elif python_detected:
        print("Only Python files detected. Proceeding...")
        write_github_output(["proceed=true\n"])
    else:
        print("No readable source files found.")
        write_github_output(["proceed=false\n"])


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
