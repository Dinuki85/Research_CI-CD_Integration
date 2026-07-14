import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from pathlib import Path
import sys
import os
import joblib
from scipy.sparse import csr_matrix, hstack
from typing import List, Set, Tuple, Optional

MODEL_PATH: Path = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS: Set[str] = {".py", ".java", ".cs", ".js", ".php"}
PYTHON_LANG: str = "python"


def load_model_bundle(model_path: Path) -> Tuple[object, object, object, List[str]]:
    """Load the language classifier bundle from the given path.

    Args:
        model_path: Path to the joblib file containing the model bundle.

    Returns:
        A tuple containing the model, character-level TF-IDF transformer,
        word-level TF-IDF transformer, and class name list.

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
    model: object,
    tfidf_char: object,
    tfidf_word: object,
    class_names: List[str],
) -> Tuple[str, int]:
    """Predict the programming language of a file using the loaded model.

    Args:
        file_path: Path to the source file.
        model: Trained classifier model.
        tfidf_char: Character-level TF-IDF transformer.
        tfidf_word: Word-level TF-IDF transformer.
        class_names: List mapping model output indices to language names.

    Returns:
        A tuple of the predicted language name and the number of lines in the file.
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
    pred_index = model.predict(features)[0]
    prediction = class_names[pred_index]
    return str(prediction), line_count


def scan_repository(
    model: object,
    tfidf_char: object,
    tfidf_word: object,
    class_names: List[str],
) -> Tuple[bool, bool, List[str], Set[str]]:
    """Traverse the repository and detect languages present in source files.

    Args:
        model: Trained classifier model.
        tfidf_char: Character-level TF-IDF transformer.
        tfidf_word: Word-level TF-IDF transformer.
        class_names: List mapping model output indices to language names.

    Returns:
        A tuple containing flags for Python detection, other language detection,
        a list of Python file paths, and a set of detected non-Python languages.
    """
    python_detected = False
    other_detected = False
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
            prediction, _ = predict_language(
                file_path, model, tfidf_char, tfidf_word, class_names
            )
            if prediction.lower() == PYTHON_LANG:
                python_detected = True
                python_files.append(str(file_path))
            else:
                other_detected = True
                other_languages.add(prediction)
                print(f"Detected other language ({prediction}) in file: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return python_detected, other_detected, python_files, other_languages


def write_python_file_list(python_files: List[str]) -> None:
    """Write the list of Python file paths to 'python_files.txt'."""
    with open("python_files.txt", "w", encoding="utf-8") as pf:
        for path in python_files:
            pf.write(f"{path}\n")


def update_github_output(
    github_output: Optional[str],
    **kwargs: str,
) -> None:
    """Append key=value pairs to the GitHub Actions output file.

    Args:
        github_output: Path to the GITHUB_OUTPUT file.
        **kwargs: Arbitrary key=value pairs to write.
    """
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as f:
        for key, value in kwargs.items():
            f.write(f"{key}={value}\n")


def main() -> None:
    """Entry point for language detection and pipeline decision making."""
    try:
        model, tfidf_char, tfidf_word, class_names = load_model_bundle(MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("Scanning files for language detection...")
    python_detected, other_detected, python_files, other_languages = scan_repository(
        model, tfidf_char, tfidf_word, class_names
    )
    github_output = os.environ.get("GITHUB_OUTPUT")
    langs_str = ", ".join(other_languages)

    if other_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {langs_str}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {langs_str}")
        write_python_file_list(python_files)
        update_github_output(
            github_output,
            proceed="true",
            python_only_mode="true",
            skipped_languages=langs_str,
        )
    elif other_detected:
        print(f"Non-Python language detected ({langs_str}). No Python files found. Stopping system.")
        update_github_output(
            github_output,
            proceed="false",
            detected_languages=langs_str,
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
