"""Every text file StrainLink reads or writes names its encoding.

Without it, Windows uses its old default code page (cp1252), which cannot
hold Greek letters or symbols such as γ, ε, µ or °: the first Windows build
crashed writing its report for exactly that reason. This test reads the
source code, so it catches the mistake on any system.
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "strainlink"
TEXT_CALLS = {"read_text", "write_text"}


def _binary_open(call: ast.Call) -> bool:
    mode = call.args[1] if len(call.args) > 1 else next((k.value for k in call.keywords if k.arg == "mode"), None)
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value


def test_every_text_read_and_write_names_its_encoding():
    missing = []
    for path in sorted(SOURCE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            is_text_call = name in TEXT_CALLS or (name == "open" and isinstance(func, ast.Name) and not _binary_open(node))
            if is_text_call and not any(k.arg == "encoding" for k in node.keywords):
                missing.append(f"{path.relative_to(SOURCE)}:{node.lineno} {name}()")
    assert not missing, "text read/write without encoding=:\n" + "\n".join(missing)
