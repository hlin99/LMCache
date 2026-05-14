# SPDX-License-Identifier: Apache-2.0
"""Regression tests for optional imports in CLI commands requiring torch."""

# Standard
from pathlib import Path
import os
import subprocess
import sys


def test_cli_main_imports_without_torch(tmp_path: Path) -> None:
    """CLI module import must not require the full runtime package."""
    repo_root = Path(__file__).resolve().parents[3]
    import_guard_dir = tmp_path
    (import_guard_dir / "sitecustomize.py").write_text(
        "import builtins\n"
        "_real_import = builtins.__import__\n"
        "def _guarded_import(name, *args, **kwargs):\n"
        "    if name == 'torch':\n"
        "        raise ModuleNotFoundError(\"No module named 'torch'\")\n"
        "    return _real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = _guarded_import\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    pythonpath = [str(import_guard_dir), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    result = subprocess.run(
        [sys.executable, "-c", "import lmcache.cli.main; print('IMPORT_OK')"],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout
