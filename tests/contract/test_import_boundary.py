"""Inference must never touch ground truth, the manifest or the eval package.

CLAUDE.md §6.1: code/rewind/ may not import rewind_eval, open annotation files or
read kit/manifest.csv. This scans the source with `ast`, so it also covers code
paths no test executes.
"""

import ast

import pytest

from tests.conftest import REPO_ROOT

REWIND_DIR = REPO_ROOT / "code" / "rewind"

FORBIDDEN_MODULES = ("rewind_eval",)
# Substrings that point at ground truth or the manifest when they appear in a string.
FORBIDDEN_STRINGS = (
    "manifest.csv",
    "EPIC_100_train",
    "EPIC_100_validation",
    "annotations",
    "rewind_eval",
)


def find_violations(source: str, filename: str = "<src>") -> list[str]:
    """Return one message per forbidden import or string literal in `source`."""
    tree = ast.parse(source, filename=filename)
    docstrings = _docstring_ids(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        for name in names:
            if name.split(".")[0] in FORBIDDEN_MODULES:
                found.append(f"{filename}:{node.lineno} imports {name}")
        is_code_string = (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        )
        if is_code_string:
            for bad in FORBIDDEN_STRINGS:
                if bad in node.value:
                    found.append(f"{filename}:{node.lineno} mentions {bad!r}")
    return found


def _docstring_ids(tree: ast.Module) -> set[int]:
    """ids of docstring nodes: documentation may name what the code must not do."""
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, owners) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


@pytest.mark.parametrize(
    "source",
    [
        "import rewind_eval",
        "from rewind_eval.coverage import coverage",
        "import rewind_eval.map as m",
        "importlib.import_module('rewind_eval')",
        "open('kit/manifest.csv')",
        "p = 'data/annotations/EPIC_100_validation.csv'",
        # A string after the docstring is code, not documentation.
        "def f():\n    'doc'\n    return 'kit/manifest.csv'",
    ],
)
def test_find_violations_flags_forbidden_source(source):
    # Positive control: the scanner itself must catch each forbidden pattern.
    assert find_violations(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np",
        "from rewind.constants import MIN_SEG",
        "x = 'evaluation'",
        '"""Never reads kit/manifest.csv or imports rewind_eval."""',
        "def f():\n    'Never opens EPIC_100_train.csv.'\n    return 1",
        "class C:\n    'Not for annotations.'",
    ],
)
def test_find_violations_allows_clean_source(source):
    assert find_violations(source) == []


def test_rewind_package_has_no_forbidden_imports_or_paths():
    files = sorted(REWIND_DIR.rglob("*.py"))
    assert files, "code/rewind/ has no Python files; wrong path?"

    violations = [
        v
        for path in files
        for v in find_violations(path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT)))
    ]

    assert violations == []
