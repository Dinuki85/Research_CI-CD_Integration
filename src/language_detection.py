import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from pathlib import Path
import sys
import os
import joblib
from typing import Tuple, List, Set, Optional, Dict, Any
from scipy.sparse import csr_matrix, hstack

MODEL_PATH = Path("models/best_language_classifier.joblib")
VALID_EXTENSIONS = {".py", ".java", ".cs", ".js", ".php"}


def load_model_bundle(model_path: Path) -> Tuple[Any, Any, Any, List[str]]:
    """Load the language classifier model bundle.

    Args:
        model_path: Path to the joblib file containing the model bundle.

    Returns:
        A tuple containing the model, character TF‑IDF transformer,
        word TF‑IDF transformer, and list of class names.

    Raises:
        SystemExit: If the model file does not exist.
    """
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        sys.exit(1)

    print("Loading language classifier bundle...")
    bundle = joblib.load(model_path)
    return (
        bundle["model"],
        bundle["tfidf_char"],
        bundle["tfidf_word"],
        bundle["class_names"],
    )


def is_valid_file(file_path: Path) -> bool:
    """Determine whether a file should be processed.

    Args:
        file_path: Path object representing a file.

    Returns:
        True if the file is not hidden, not inside a 'models' directory,
        and has a supported extension.
    """
    return (
        file_path.is_file()
        and not any(part.startswith(".") for part in file_path.parts)
        and "models" not in file_path.parts
        and file_path.suffix.lower() in VALID_EXTENSIONS
    )


def predict_file(
    file_path: Path,
    model: Any,
    tfidf_char: Any,
    tfidf_word: Any,
    class_names: List[str],
) -> Optional[str]:
    """Predict the programming language of a file.

    Args:
        file_path: Path to the source file.
        model: Trained classifier.
        tfidf_char: Character‑level TF‑IDF transformer.
        tfidf_word: Word‑level TF‑IDF transformer.
        class_names: Mapping from label indices to language names.

    Returns:
        Predicted language name, or None if an error occurs.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        lines = len(content.splitlines())
        combined = file_path.name.lower() + "  " + content

        x_char = tfidf_char.transform([combined])
        x_word = tfidf_word.transform([combined])
        x_lines = csr_matrix([[float(lines)]])

        if hasattr(model, "n_features_in_") and model.n_features_in_ == x_char.shape[1] + x_word.shape[1]:
            X = hstack([x_char, x_word])
        else:
            X = hstack([x_char, x_word, x_lines])

        pred_label = model.predict(X)[0]
        return class_names[pred_label]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def write_python_file_list(python_files: List[str]) -> None:
    """Write the list of detected Python files to 'python_files.txt'.

    Args:
        python_files: List of file paths as strings.
    """
    with open("python_files.txt", "w", encoding="utf-8") as pf:
        for path in python_files:
            pf.write(path + "\n")


def update_github_output(github_output: Optional[str], data: Dict[str, str]) -> None:
    """Append key‑value pairs to the GitHub Actions output file.

    Args:
        github_output: Path to the GITHUB_OUTPUT file, or None.
        data: Mapping of keys to string values to write.
    """
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as f:
        for key, value in data.items():
            f.write(f"{key}={value}\n")


def main() -> None:
    """Detect repository language composition and decide processing mode.

    Scans the repository for source files, classifies each file's language,
    and writes decision flags for downstream GitHub Actions steps.
    """
    model, tfidf_char, tfidf_word, class_names = load_model_bundle(MODEL_PATH)

    python_files: List[str] = []
    other_languages: Set[str] = set()
    python_detected = False
    other_language_detected = False

    print("Scanning files for language detection...")
    for file_path in Path(".").rglob("*"):
        if not is_valid_file(file_path):
            continue

        prediction = predict_file(file_path, model, tfidf_char, tfidf_word, class_names)
        if prediction is None:
            continue

        if str(prediction).lower() == "python":
            python_detected = True
            python_files.append(str(file_path))
        else:
            other_language_detected = True
            other_languages.add(str(prediction))
            print(f"Detected other language ({prediction}) in file: {file_path}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    langs = ", ".join(other_languages)

    if other_language_detected and python_detected:
        print(f"Mixed repo detected. Non-Python languages found: {langs}")
        print(f"Proceeding with {len(python_files)} Python file(s) only. Skipping: {langs}")
        write_python_file_list(python_files)
        update_github_output(github_output, {
            "proceed": "true",
            "python_only_mode": "true",
            "skipped_languages": langs,
        })
        return

    if other_language_detected:
        print(f"Non-Python language detected ({langs}). No Python files found. Stopping system.")
        update_github_output(github_output, {
            "proceed": "false",
            "detected_languages": langs,
        })
        sys.exit(1)

    if python_detected:
        print("Only Python files detected. Proceeding...")
        update_github_output(github_output, {"proceed": "true"})
        return

    print("No readable source files found.")
    update_github_output(github_output, {"proceed": "false"})


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
