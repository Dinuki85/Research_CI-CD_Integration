import os
import sys
import joblib
from pathlib import Path

def main():
    model_path = Path("models/best_language_classifier.joblib")
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        sys.exit(1)

    # 1. Load the ML model
    print("Loading language classifier model...")
    model = joblib.load(model_path)
    
    python_detected = False
    other_language_detected = False
    
    # 2. Scan files in the repository (skipping hidden folders and the models folder)
    print("Scanning files for language detection...")
    for file_path in Path(".").rglob("*"):
        if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts) and 'models' not in file_path.parts:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # IMPORTANT: If your language_classification.ipynb used any text preprocessing
                # (like removing special characters), apply it to 'content' here before predicting.
                prediction = model.predict([content])[0]
                
                if str(prediction).lower() == "python":
                    python_detected = True
                else:
                    other_language_detected = True
                    print(f"Detected other language ({prediction}) in file: {file_path}")
                    
            except Exception:
                # Skip binary files or unreadable files
                pass

    github_output = os.environ.get("GITHUB_OUTPUT")
    
    # 3. Determine if we should proceed
    if other_language_detected:
        print("Non-Python language detected. Stopping system.")
        if github_output:
            with open(github_output, "a", encoding="utf-8") as f:
                f.write("proceed=false\n")
    elif python_detected:
        print("Only Python files detected. Proceeding...")
        if github_output:
            with open(github_output, "a", encoding="utf-8") as f:
                f.write("proceed=true\n")
    else:
        print("No readable source files found.")
        if github_output:
            with open(github_output, "a", encoding="utf-8") as f:
                f.write("proceed=false\n")

if __name__ == "__main__":
    main()
