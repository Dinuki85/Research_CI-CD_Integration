# CodeRefactor AI

CodeRefactor AI is an automated, intelligent Python code refactoring pipeline. It analyzes repositories for code smells (specifically long methods and poor structures) and automatically rewrites the flawed code using the most optimal Large Language Model (LLM) for that specific function.

Rather than relying on a single hardcoded AI, CodeRefactor AI uses a multi-stage machine learning approach (SVM, Random Forest, Gradient Boosting, etc.) to predict which LLM will provide the best refactoring quality, time, and cost efficiency.

## Features

- **AST-Based Repository Parsing:** Safely and accurately extracts Python functions across your repository, avoiding the pitfalls of naive regex parsing.
- **ML-Powered Code Smell Detection:** Uses a trained Support Vector Machine (SVM) and 12 structural features to accurately identify functions that need refactoring (e.g., long methods, deep nesting).
- **Intelligent LLM Routing:** Analyzes 50 complex features to route the refactoring task to the optimal LLM (evaluating Claude Opus, Claude Sonnet 4.6, Gemini 3.1 Pro, Gemini Flash, and GPT-OSS 120B) based on a composite score of Quality, Cost, Time, and Tokens.
- **Automated Refactoring:** Integrates with the OpenRouter API to instruct the chosen LLM to rewrite the code, maintain identical behavior, fix structural issues, and add docstrings/type hints.
- **CI/CD Integration:** Includes built-in support for Git pre-commit hooks and GitHub Actions to enforce code quality automatically during development.

## Prerequisites

- Python 3.9+
- OpenRouter API Key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/CodeRefactor-AI.git
cd CodeRefactor-AI
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set your OpenRouter API Key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
# On Windows PowerShell: $env:OPENROUTER_API_KEY="your-api-key-here"
```

## Usage

You can run the CLI to analyze and refactor a local repository or specific files.

**Analyze and refactor a local repository:**
```bash
python cli.py --repo /path/to/your/repo
```

**Analyze specific files in-place:**
```bash
python cli.py --files src/my_file.py --in-place
```

**Dry Run (Analyze smells and recommend LLMs without making API calls or changes):**
```bash
python cli.py --repo /path/to/your/repo --dry-run
```

**Other useful options:**
- `--verbose` or `-V`: Show detailed per-function smell analysis.
- `--output` or `-o`: Specify an output directory for the refactored code (if not using `--in-place`).
- `--json`: Save the analysis report as a `refactor-report.json` file.

## CI/CD and Developer Workflow

### Pre-commit Hook
CodeRefactor AI can automatically intercept bad code before it is committed.

To set up the pre-commit hook:
```bash
python src/setup_hooks.py
# Or manually copy:
# cp hooks/pre-commit .git/hooks/pre-commit
# chmod +x .git/hooks/pre-commit
```

Once installed, every time you run `git commit`, the hook will scan the staged Python files. If bad smells are detected, it will automatically refactor them in-place and re-stage the fixed files. 

You can control the hook with environment variables:
- `CODEREFACTOR_SKIP=1 git commit`: Skips the hook.
- `CODEREFACTOR_DRY_RUN=1 git commit`: Analyzes and reports smells without changing files.

### GitHub Actions
The project includes a GitHub Actions workflow (`.github/workflows/refactor-pipeline.yml`). When properly configured, it can automatically review Pull Requests and generate reports on code quality.

## Project Structure

- `cli.py`: The main command-line interface.
- `src/repo_analyzer.py`: AST parser for extracting functions from source code.
- `src/code_smell_detector.py`: SVM-based ML model to detect code smells.
- `src/llm_recommender.py`: Regression models that evaluate and select the best LLM for refactoring.
- `src/refactoring_engine.py`: Handles API communication with OpenRouter to rewrite the code.
- `models/`: Serialized `.pkl` machine learning models (scalers, encoders, SVMs, and regression trees).
- `hooks/`: Contains the Git pre-commit hook script.

## Contributing

Contributions are welcome! If you're interested in improving the ML models, adding support for new LLMs, or expanding the supported code smells, please open an issue or submit a pull request.
