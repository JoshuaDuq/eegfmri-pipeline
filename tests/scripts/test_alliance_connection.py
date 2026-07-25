from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from tests import REPO_ROOT

SOURCE_DIR = REPO_ROOT / "local_workflows" / "alliance_canada"


def _write_fake_ssh(bin_dir: Path, log_path: Path) -> None:
    bin_dir.mkdir()
    ssh_path = bin_dir / "ssh"
    ssh_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' \"$*\" >> {log_path!s}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    ssh_path.chmod(0o755)


def _workflow(tmp_path: Path) -> Path:
    workflow = tmp_path / "alliance_canada"
    workflow.mkdir()
    shutil.copy(SOURCE_DIR / "alliance_env.sh", workflow)
    shutil.copy(SOURCE_DIR / "load_cluster_profile.sh", workflow)
    shutil.copytree(SOURCE_DIR / "clusters", workflow / "clusters")
    shutil.copy(SOURCE_DIR / "start_alliance_connection.sh", workflow)
    shutil.copy(SOURCE_DIR / "stop_alliance_connection.sh", workflow)
    shutil.copytree(SOURCE_DIR / "lib", workflow / "lib")
    (workflow / "local_env.sh").write_text(
        'export ALLIANCE_CLUSTER="rorqual"\n',
        encoding="utf-8",
    )
    return workflow


def test_start_connection_uses_selected_profile(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    log_path = tmp_path / "ssh.log"
    bin_dir = tmp_path / "bin"
    _write_fake_ssh(bin_dir, log_path)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"

    result = subprocess.run(
        ["bash", str(workflow / "start_alliance_connection.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = log_path.read_text(encoding="utf-8")
    assert "/tmp/rorqual-joshduq-ssh-control" in calls
    assert "joshduq@rorqual.alliancecan.ca" in calls


def test_stop_connection_uses_selected_profile(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    log_path = tmp_path / "ssh.log"
    bin_dir = tmp_path / "bin"
    _write_fake_ssh(bin_dir, log_path)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"

    result = subprocess.run(
        ["bash", str(workflow / "stop_alliance_connection.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    call = log_path.read_text(encoding="utf-8")
    assert "-O exit" in call
    assert "joshduq@rorqual.alliancecan.ca" in call
