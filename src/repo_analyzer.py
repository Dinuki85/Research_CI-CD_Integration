"""
Repository Analyzer
===================
Scans a local or remote Git repository for Python functions, extracts each
function's source code, and prepares them for code smell analysis.

Supports:
  - Local repository scanning (path)
  - Remote repository cloning (Git URL)
  - AST-based function extraction with source code preservation
  - Function metadata (name, file, line numbers, decorators)
"""

import ast
import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional


def clone_repository(repo_url: str, target_dir: Optional[Path] = None) -> Path:
    """
    Clone a git repository to a local directory.

    Args:
        repo_url: Git repository URL (HTTPS or SSH).
        target_dir: Target directory. If None, creates a temp directory.

    Returns:
        Path to the cloned repository.
    """
    if target_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="coderefactor-"))

    print(f"Cloning {repo_url} into {target_dir}...")

    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone repository: {result.stderr}")

    return target_dir


def find_python_files(repo_path: Path) -> List[Path]:
    """
    Recursively find all Python files in a repository, excluding
    common non-source directories.

    Returns:
        List of paths to .py files.
    """
    excluded_dirs = {
        ".git", "__pycache__", "venv", ".venv", "env", ".env",
        "node_modules", "dist", "build", ".tox", ".eggs",
        "egg-info", ".mypy_cache", ".pytest_cache", ".coverage",
    }

    python_files = []
    for root, dirs, files in os.walk(str(repo_path)):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                python_files.append(Path(root) / f)

    return sorted(python_files)


class FunctionExtractor(ast.NodeVisitor):
    """
    AST visitor that extracts top-level and nested function definitions
    with their source code ranges.
    """

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.source_lines = source_code.split("\n")
        self.functions: List[Dict] = []

import ast
from typing import Any, Dict, List, Optional


SELF_IDENTIFIERS = ("self", "cls")


def _extract_source_code(self, start_idx: int, end_idx: int) -> str:
    """Return the source code slice between *start_idx* and *end_idx* (exclusive)."""
    return "\n".join(self.source_lines[start_idx:end_idx])


def _extract_decorators(self, node: ast.FunctionDef) -> List[str]:
    """Collect decorator representations for *node*."""
    decorators: List[str] = []
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            decorators.append(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            decorators.append(ast.dump(decorator))
    return decorators


def _extract_parameters(self, node: ast.FunctionDef) -> List[str]:
    """Return a list of parameter names for *node*."""
    return [arg.arg for arg in node.args.args]


def _determine_flags(params: List[str], parent_class: Optional[str]) -> Dict[str, bool]:
    """Determine method/nested flags based on *params* and *parent_class*."""
    is_method = bool(params and params[0] in SELF_IDENTIFIERS)
    is_nested = parent_class is not None and not is_method
    return {"is_method": is_method, "is_nested": is_nested}


def _extract_function_info(self, node: ast.FunctionDef, parent_class: Optional[str] = None) -> Dict[str, Any]:
    """Extract metadata and source for a function definition.

    Args:
        node: AST node representing the function definition.
        parent_class: Name of the enclosing class, if any.

    Returns:
        Dictionary containing function metadata such as name, source code, line numbers,
        parameters, decorators, and flags indicating whether it is a method or nested function.

    Raises:
        AttributeError: If the function node lacks required attributes.
    """
    start_idx = node.lineno - 1  # convert to 0‑based index
    end_idx = getattr(node, "end_lineno", start_idx + 1)

    source_code = _extract_source_code(self, start_idx, end_idx)
    decorators = _extract_decorators(self, node)
    parameters = _extract_parameters(self, node)

    flags = _determine_flags(parameters, parent_class)

    full_name = f"{parent_class}.{node.name}" if parent_class else node.name

    return {
        "name": node.name,
        "full_name": full_name,
        "file_path": None,  # to be populated by the caller
        "source_code": source_code,
        "start_line": start_idx + 1,  # back to 1‑based indexing
        "end_line": end_idx,
        "num_lines": end_idx - start_idx,
        "params": parameters,
        "num_params": len(parameters),
        "decorators": decorators,
        "is_method": flags["is_method"],
        "is_nested": flags["is_nested"],
        "parent_class": parent_class,
        "docstring": ast.get_docstring(node),
    }

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Skip __init__ and other dunder methods (they're standard)
        if node.name.startswith("__") and node.name.endswith("__") and node.name != "__init__":
            return

        info = self._extract_function_info(node)
        self.functions.append(info)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        info = self._extract_function_info(node)
        self.functions.append(info)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions and extract their methods."""
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info = self._extract_function_info(item, parent_class=node.name)
                self.functions.append(info)
        # Don't visit nested classes recursively (for now)


def extract_functions_from_file(file_path: Path) -> List[Dict]:
    """
    Extract all function definitions from a single Python file.

    Returns:
        List of function info dicts with keys:
            name, full_name, file_path, source_code, start_line, end_line,
            num_lines, params, num_params, decorators, is_method, docstring
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source_code = f.read()
    except Exception as e:
        print(f"  [WARN] Cannot read {file_path}: {e}", file=sys.stderr)
        return []

    # Skip empty or very short files
    if len(source_code.strip()) < 20:
        return []

    # Try to parse AST
    try:
        tree = ast.parse(source_code, filename=str(file_path))
    except SyntaxError as e:
        print(f"  [WARN] Syntax error in {file_path}: {e}", file=sys.stderr)
        return []

    extractor = FunctionExtractor(source_code)
    extractor.visit(tree)

    # Set file path for each function
    for func in extractor.functions:
        func["file_path"] = str(file_path)

    return extractor.functions


def analyze_repository(repo_path: Path) -> List[Dict]:
    """
    Analyze an entire repository and extract all function definitions.

    Args:
        repo_path: Path to the local repository.

    Returns:
        List of all function info dicts across all Python files.
    """
    repo_path = Path(repo_path).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    python_files = find_python_files(repo_path)
    print(f"Found {len(python_files)} Python files in {repo_path}")

    all_functions = []
    for py_file in python_files:
        functions = extract_functions_from_file(py_file)
        all_functions.extend(functions)

    print(f"Extracted {len(all_functions)} functions total")

    return all_functions


def scan_functions(repo_path: Path) -> List[Dict]:
    """
    High-level wrapper: scan repo and return functions with their
    relative paths (for nicer display).
    """
    repo_path = Path(repo_path).resolve()
    functions = analyze_repository(repo_path)

    # Add relative file paths
    for func in functions:
        abs_path = Path(func["file_path"])
        try:
            func["rel_path"] = str(abs_path.relative_to(repo_path))
        except ValueError:
            func["rel_path"] = func["file_path"]

    return functions


from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PYTHON_SUFFIX = ".py"
WARN_PREFIX = "  [WARN]"


def _is_valid_python_file(file_path: Path) -> bool:
    """Check if the given path exists and is a Python file."""
    if not file_path.exists():
        print(f"{WARN_PREFIX} File does not exist: {file_path}", file=sys.stderr)
        return False
    if file_path.suffix != PYTHON_SUFFIX:
        print(f"{WARN_PREFIX} Not a Python file: {file_path}", file=sys.stderr)
        return False
    return True


def _set_relative_path(func_info: Dict[str, Any], root: Optional[Path]) -> None:
    """Add a relative path entry to a function info dictionary."""
    absolute_path = Path(func_info["file_path"])
    if root:
        try:
            func_info["rel_path"] = str(absolute_path.relative_to(root))
            return
        except ValueError:
            pass
    func_info["rel_path"] = func_info["file_path"]


def scan_files(file_paths: List[Path], root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scan a list of Python files and extract function information.

    Args:
        file_paths: List of paths to Python files to scan.
        root: Optional root directory used to compute relative file paths.

    Returns:
        A list of dictionaries, each containing information about a discovered
        function, enriched with ``rel_path`` entries.

    """
    resolved_root = Path(root).resolve() if root else None
    all_functions: List[Dict[str, Any]] = []

    for path in file_paths:
        py_file = Path(path).resolve()
        if not _is_valid_python_file(py_file):
            continue

        functions = extract_functions_from_file(py_file)

        for func in functions:
            _set_relative_path(func, resolved_root)

        all_functions.extend(functions)

    return all_functions
