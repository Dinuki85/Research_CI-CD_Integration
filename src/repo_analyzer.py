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


from pathlib import Path
import os
from typing import List, Iterable

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    "dist",
    "build",
    ".tox",
    ".eggs",
    "egg-info",
    ".mypy_cache",
    ".pytest_cache",
    ".coverage",
}
PYTHON_EXTENSION = ".py"


def _is_excluded(dir_name: str) -> bool:
    """Return ``True`` if a directory should be skipped during traversal."""
    return dir_name in EXCLUDED_DIRS or dir_name.startswith(".")


def _iter_python_files(repo_path: Path) -> Iterable[Path]:
    """Yield Python files found under ``repo_path`` respecting exclusion rules."""
    for root, dirs, files in os.walk(str(repo_path)):
        dirs[:] = [d for d in dirs if not _is_excluded(d)]
        for file_name in files:
            if file_name.endswith(PYTHON_EXTENSION):
                yield Path(root) / file_name


def find_python_files(repo_path: Path) -> List[Path]:
    """Recursively find all Python files in a repository, excluding common non‑source directories.

    Args:
        repo_path: Path to the root of the repository.

    Returns:
        Sorted list of :class:`pathlib.Path` objects pointing to ``.py`` files.

    Raises:
        ValueError: If ``repo_path`` does not exist or is not a directory.
    """
    if not repo_path.is_dir():
        raise ValueError(f"repo_path must be an existing directory: {repo_path}")
    return sorted(_iter_python_files(repo_path))


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

ONE_INDEX_ADJUST = 1
DEFAULT_END_LINE_OFFSET = 1


def _extract_source_code(source_lines: List[str], start: int, end: Optional[int]) -> str:
    """Return the source code slice for the given line range."""
    end_line = end if end is not None else start + DEFAULT_END_LINE_OFFSET
    return "\n".join(source_lines[start:end_line])


def _collect_decorators(decorator_list: List[ast.expr]) -> List[str]:
    """Extract decorator identifiers from an AST decorator list."""
    decorators: List[str] = []
    for decorator in decorator_list:
        if isinstance(decorator, ast.Name):
            decorators.append(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            decorators.append(ast.dump(decorator))
    return decorators


def _collect_parameters(args: ast.arguments) -> List[str]:
    """Return a list of parameter names for a function definition."""
    return [arg.arg for arg in args.args]


def _determine_method_flags(params: List[str], parent_class: Optional[str]) -> tuple[bool, bool]:
    """Determine if a function is a method and if it is nested."""
    is_method = bool(params and params[0] in ("self", "cls"))
    is_nested = parent_class is not None and not is_method
    return is_method, is_nested


def _extract_function_info(
    self,
    node: ast.FunctionDef,
    parent_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract metadata and source code for a function definition.

    Args:
        node: AST node representing the function definition.
        parent_class: Name of the enclosing class, if any.

    Returns:
        A dictionary containing function metadata such as name, full name,
        source code, line numbers, parameters, decorators, and flags indicating
        whether the function is a method or nested.

    Raises:
        ValueError: If the node does not contain required line information.
    """
    start_line = node.lineno - ONE_INDEX_ADJUST  # convert to 0‑indexed
    if not hasattr(node, "end_lineno"):
        raise ValueError("AST node missing 'end_lineno' attribute.")
    end_line = node.end_lineno

    source_code = _extract_source_code(self.source_lines, start_line, end_line)

    decorators = _collect_decorators(node.decorator_list)
    params = _collect_parameters(node.args)

    is_method, is_nested = _determine_method_flags(params, parent_class)

    full_name = f"{parent_class}.{node.name}" if parent_class else node.name

    return {
        "name": node.name,
        "full_name": full_name,
        "file_path": None,  # Set by caller
        "source_code": source_code,
        "start_line": start_line + ONE_INDEX_ADJUST,
        "end_line": end_line,
        "num_lines": end_line - start_line,
        "params": params,
        "num_params": len(params),
        "decorators": decorators,
        "is_method": is_method,
        "is_nested": is_nested,
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


from pathlib import Path
from typing import Any, Dict, List, Union


def _add_relative_paths(functions: List[Dict[str, Any]], base_path: Path) -> None:
    """Add a ``rel_path`` key to each function dict.

    The relative path is calculated with respect to ``base_path``. If the
    absolute path cannot be made relative (e.g., it is outside the repository),
    the original absolute path is retained.

    Args:
        functions: List of dictionaries representing functions. Each dict must
            contain a ``file_path`` key with an absolute path string.
        base_path: The repository root used to compute relative paths.
    """
    for function in functions:
        absolute_path = Path(function["file_path"])
        try:
            function["rel_path"] = str(absolute_path.relative_to(base_path))
        except ValueError:
            function["rel_path"] = function["file_path"]


def scan_functions(repo_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Scan a repository and return information about its functions.

    The function delegates the heavy‑lifting to ``analyze_repository`` and then
    enriches each result with a relative file path for easier display.

    Args:
        repo_path: Path to the repository root. Can be a string or ``Path`` object.

    Returns:
        A list of dictionaries, each representing a function found in the
        repository. Each dictionary includes the original data from
        ``analyze_repository`` plus a ``rel_path`` key.

    Raises:
        Any exception raised by ``analyze_repository`` is propagated unchanged.
    """
    repository_root = Path(repo_path).resolve()
    functions = analyze_repository(repository_root)
    _add_relative_paths(functions, repository_root)
    return functions


from pathlib import Path
import sys
from typing import List, Dict, Optional, Any

PYTHON_SUFFIX = ".py"
WARN_FILE_NOT_EXIST = "[WARN] File does not exist: {}"
WARN_NOT_PYTHON_FILE = "[WARN] Not a Python file: {}"


def _is_valid_python_file(file_path: Path) -> bool:
    """Check whether the given path exists and is a Python file.

    Args:
        file_path: Path to validate.

    Returns:
        True if the file exists and has a .py suffix, False otherwise.
    """
    if not file_path.exists():
        print(f"  {WARN_FILE_NOT_EXIST.format(file_path)}", file=sys.stderr)
        return False
    if file_path.suffix != PYTHON_SUFFIX:
        print(f"  {WARN_NOT_PYTHON_FILE.format(file_path)}", file=sys.stderr)
        return False
    return True


def _add_relative_path(func_info: Dict[str, Any], root: Optional[Path]) -> None:
    """Add a relative path to a function info dictionary.

    Args:
        func_info: Dictionary containing at least a "file_path" key.
        root: Root directory to compute the relative path from. If None,
              the absolute path is used.
    """
    absolute_path = Path(func_info["file_path"])
    if root:
        try:
            func_info["rel_path"] = str(absolute_path.relative_to(root))
        except ValueError:
            func_info["rel_path"] = func_info["file_path"]
    else:
        func_info["rel_path"] = func_info["file_path"]


def scan_files(file_paths: List[Path], root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scan a list of Python files and extract function information.

    Args:
        file_paths: List of paths to Python files to scan.
        root: Optional root path for computing relative paths.

    Returns:
        A list of dictionaries, each containing information about a function
        found in the provided files.

    Raises:
        None.
    """
    resolved_root = Path(root).resolve() if root else None
    all_functions: List[Dict[str, Any]] = []

    for file_path in file_paths:
        resolved_path = Path(file_path).resolve()
        if not _is_valid_python_file(resolved_path):
            continue

        functions = extract_functions_from_file(resolved_path)

        for func in functions:
            _add_relative_path(func, resolved_root)

        all_functions.extend(functions)

    return all_functions
