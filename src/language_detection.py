import os
import sys
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

def main():
    model_path = Path("models/best_language_classifier.joblib")
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        sys.exit(1)

    # 1. Load the ML model bundle
    print("Loading language classifier bundle...")
    bundle = joblib.load(model_path)
    model = bundle['model']
    tfidf_char = bundle['tfidf_char']
    tfidf_word = bundle['tfidf_word']
    class_names = bundle['class_names']
    
    python_detected = False
    other_language_detected = False
    detected_other_languages = set()
    
    # 2. Scan files in the repository (skipping hidden folders, models, and non-code files)
    print("Scanning files for language detection...")
    
    valid_extensions = {".py", ".java", ".cs", ".js", ".php"}
    
    for file_path in Path(".").rglob("*"):
        if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts) and 'models' not in file_path.parts:
            if file_path.suffix.lower() not in valid_extensions:
                continue
                
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # IMPORTANT: Preprocess and predict exactly like the notebook demo
                lines = len(content.splitlines())
                # Note: The notebook expected function names. For full files, we just use filename + content.
                combined = file_path.name.lower() + '  ' + content
                
                x_char = tfidf_char.transform([combined])
                x_word = tfidf_word.transform([combined])
                x_lines = csr_matrix([[float(lines)]])
                
                # Some models (like Naive Bayes) drop the 'lines' feature and only expect 25,000 text features
                if hasattr(model, "n_features_in_") and model.n_features_in_ == x_char.shape[1] + x_word.shape[1]:
                    X = hstack([x_char, x_word])
                else:
                    X = hstack([x_char, x_word, x_lines])
                
                pred_label = model.predict(X)[0]
                prediction = class_names[pred_label]
                
                if str(prediction).lower() == "python":
                    python_detected = True
                else:
                    other_language_detected = True
                    detected_other_languages.add(str(prediction))
                    print(f"Detected other language ({prediction}) in file: {file_path}")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                pass

    github_output = os.environ.get("GITHUB_OUTPUT")
    
    # 3. Determine if we should proceed
    if other_language_detected:
        print("Non-Python language detected. Stopping system.")
        if github_output:
            with open(github_output, "a", encoding="utf-8") as f:
                f.write("proceed=false\n")
                langs = ", ".join(detected_other_languages)
                f.write(f"detected_languages={langs}\n")
        sys.exit(1)
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
