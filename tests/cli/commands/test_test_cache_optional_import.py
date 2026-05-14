# SPDX-License-Identifier: Apache-2.0
"""Regression tests for optional imports in CLI commands requiring torch."""

# Standard
from pathlib import Path
import os
import subprocess
import sys

IMPORT_TEST_SCRIPT = "import lmcache.cli.main; print('IMPORT_OK')"


def test_cli_main_imports_without_torch(tmp_path: Path) -> None:
    """Verify the CLI still imports when a subprocess blocks ``torch``.

    The test writes a temporary ``sitecustomize.py`` that forces every
    ``import torch`` in the subprocess to fail with
    :class:`ModuleNotFoundError`. Importing :mod:`lmcache.cli.main`
    must still succeed so the thin ``lmcache-cli`` package can load and
    defer the install hint to ``_require_full_install()``.
    """
    repo_root = Path(__file__).resolve().parents[3]
    sitecustomize_dir = tmp_path
    (sitecustomize_dir / "sitecustomize.py").write_text(
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
    pythonpath = [str(sitecustomize_dir), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    result = subprocess.run(
        [sys.executable, "-c", IMPORT_TEST_SCRIPT],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout
