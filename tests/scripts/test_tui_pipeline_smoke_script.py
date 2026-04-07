from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "tui_pipeline_smoke.py"
    spec = importlib.util.spec_from_file_location("tui_pipeline_smoke", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTuiPipelineSmokeScript(unittest.TestCase):
    def test_candidate_python_commands_use_windows_venv_layout(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            python_path = repo_root / ".venv311" / "Scripts" / "python.exe"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")

            with patch.object(sys, "executable", "C:\\Python311\\python.exe"):
                commands = module._candidate_python_commands(repo_root, os_name="nt")

        self.assertEqual(commands[0], [str(python_path)])
        self.assertIn(["C:\\Python311\\python.exe"], commands)

    def test_resolve_python_command_uses_py_launcher_on_windows(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            with (
                patch.object(module, "_has_core_deps", side_effect=lambda cmd, _repo: cmd == ["py", "-3"]),
                patch.object(module.shutil, "which", side_effect=lambda name: "C:\\Windows\\py.exe" if name == "py" else None),
                patch.object(sys, "executable", ""),
            ):
                command = module._resolve_python_command(repo_root, os_name="nt")

        self.assertEqual(command, ["py", "-3"])
