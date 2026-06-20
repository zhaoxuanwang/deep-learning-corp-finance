"""Enforce strict v2/v3 isolation: v3 must never import v2 or legacy code.

Uses the AST so the forbidden module strings appearing in this test file itself
(in ``FORBIDDEN``) are not mistaken for imports.
"""
from __future__ import annotations

import ast
import pathlib

V3_ROOT = pathlib.Path(__file__).resolve().parents[1]  # .../src/v3
FORBIDDEN = ("src.v2", "src._legacy")


def _imported_modules(path: pathlib.Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                yield node.module


def test_v3_does_not_import_v2_or_legacy():
    offenders = []
    for path in sorted(V3_ROOT.rglob("*.py")):
        for mod in _imported_modules(path):
            if any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN):
                offenders.append(f"{path.relative_to(V3_ROOT.parent.parent)}: imports {mod}")
    assert not offenders, "v3 must be self-contained; found v2/_legacy imports:\n" + "\n".join(offenders)
