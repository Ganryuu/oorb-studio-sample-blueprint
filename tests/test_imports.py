"""Structural guarantees about the package's import graph.

The core promise is that ``import vla_engine`` works on a robot, a CI runner,
or a laptop with no GPU and no torch. That is easy to state and easy to break
with one convenience import, so it is enforced here rather than documented and
hoped for.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "src" / "vla_engine"

# Modules whose whole purpose is to touch a heavy dependency. Everything else
# must defer those imports into a function body.
HEAVY = {"torch", "transformers", "lerobot", "fastapi", "uvicorn", "bitsandbytes", "flash_attn"}

# No exemptions: even the concrete adapters, which are only reached through the
# registry's lazy loader, defer their framework imports into function bodies.
EXEMPT_MODULES: set[str] = set()


def _module_level_imports(path: pathlib.Path) -> set[str]:
    """Top-level (non-deferred) imported root package names."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in tree.body:  # module level only, not nested in functions
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Try):
            # A guarded `try: import x except ImportError:` is still a
            # module-level import and would still fail an install without it,
            # so it counts too.
            for stmt in node.body:
                if isinstance(stmt, ast.Import):
                    found.update(alias.name.split(".")[0] for alias in stmt.names)
                elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module:
                    found.add(stmt.module.split(".")[0])
    return found


ALL_MODULES = sorted(PACKAGE.rglob("*.py"))


def test_package_files_were_found():
    assert len(ALL_MODULES) > 15


@pytest.mark.parametrize("path", ALL_MODULES, ids=lambda p: p.stem)
def test_no_module_level_heavy_imports(path: pathlib.Path):
    """Heavy dependencies must be imported inside functions, not at module scope."""
    if path.stem in EXEMPT_MODULES:
        pytest.skip(f"{path.stem} is exempt by design")
    offenders = _module_level_imports(path) & HEAVY
    assert not offenders, (
        f"{path.name} imports {sorted(offenders)} at module level; "
        "move it into the function that needs it so the core stays importable"
    )


def test_importing_the_package_does_not_import_torch():
    """The public API must be usable with no deep-learning stack installed.

    Run in a subprocess: other test modules import fastapi at collection time,
    so an in-process check would be measuring this session's import history
    rather than what ``import vla_engine`` actually pulls in.
    """
    script = "\n".join(
        [
            "import sys",
            "import vla_engine",
            f"heavy = {sorted(HEAVY)!r}",
            "roots = {name.split('.')[0] for name in sys.modules}",
            "print(','.join(sorted(roots.intersection(heavy))))",
        ]
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"importing vla_engine pulled in heavy dependencies: {leaked}"


def test_public_api_is_usable_without_torch():
    from vla_engine import VLAEngine, list_models

    assert VLAEngine("openvla").spec.action_dim == 7
    assert {card.key for card in list_models()} >= {"openvla", "pi0", "smolvla"}
