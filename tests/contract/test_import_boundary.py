"""Inference must never touch ground truth, the manifest or the eval package.

CLAUDE.md §6.1: code/rewind/ may not import rewind_eval, open annotation files or
read kit/manifest.csv. The source scan uses `ast`, so it also covers code paths no
test executes; the runtime test catches imports the scan can't see. A file-open audit
hook on the run.sh contract test is the next layer (PROGRESS.md P1).
"""

import ast
import subprocess
import sys

import pytest
import yaml

from tests.conftest import DEFAULT_CONFIG, REPO_ROOT

REWIND_DIR = REPO_ROOT / "code" / "rewind"

FORBIDDEN_MODULES = ("rewind_eval",)
# Substrings (matched case-insensitively) that point at ground truth, the manifest or
# the data directory when they appear in a string literal or a config value.
FORBIDDEN_STRINGS = (
    "manifest.csv",
    "epic_100_train",
    "epic_100_validation",
    "annotation",
    "rewind_eval",
    "data/",
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
            found += [f"{filename}:{node.lineno} {m}" for m in _forbidden_in(node.value)]
    return found


def _forbidden_in(text: str) -> list[str]:
    lowered = text.lower()
    return [f"mentions {bad!r}" for bad in FORBIDDEN_STRINGS if bad in lowered]


def _config_strings(value: object) -> list[str]:
    """Every string value in a parsed YAML document (comments are already gone)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _config_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _config_strings(v)]
    return []


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
        "p = 'epic_100_validation.csv'",  # lowercase
        "p = 'fpvsum/annotation/importance.mat'",  # singular
        "root = 'data/videos'",
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


def test_config_values_have_no_forbidden_paths():
    # A path in the config is as much a bypass as one in code: open(cfg.paths.manifest).
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        values = _config_strings(yaml.safe_load(f))

    violations = [f"{DEFAULT_CONFIG.name}: {v!r} {m}" for v in values for m in _forbidden_in(v)]

    assert violations == []


def test_config_scanner_flags_forbidden_value():
    # Positive control for the config scan above.
    values = _config_strings({"paths": {"gt": ["x", "kit/manifest.csv"]}})

    assert [m for v in values for m in _forbidden_in(v)] == ["mentions 'manifest.csv'"]


def test_importing_every_rewind_module_never_loads_rewind_eval():
    # Catches imports the source scan can't see (e.g. through a third-party module).
    modules = sorted(
        ".".join(p.relative_to(REPO_ROOT / "code").with_suffix("").parts)
        for p in REWIND_DIR.rglob("*.py")
    )
    modules = [m.removesuffix(".__init__") for m in modules]
    code = (
        "import importlib, sys\n"
        f"for m in {modules!r}:\n"
        "    importlib.import_module(m)\n"
        "print(sorted(n for n in sys.modules if n.split('.')[0] == 'rewind_eval'))\n"
    )

    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "code")},
        capture_output=True,
        text=True,
        check=True,
    )

    assert out.stdout.strip() == "[]"
