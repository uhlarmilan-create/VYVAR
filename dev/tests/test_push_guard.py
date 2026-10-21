# -*- coding: ascii -*-
"""CLOSE-OUT 2026-08-25: unauthorized origin/main push must be refused."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dev" / "scripts"))

from push_guard import decide_line, install_hook  # noqa: E402


def test_main_push_without_auth_file_is_refused(tmp_path: Path) -> None:
    code, msg = decide_line(
        "refs/heads/main",
        "7c086e8111c131650a666d06245b1c0eac2b640d",
        root=tmp_path,
        yyyymmdd="20260826",
        delete_on_allow=False,
    )
    assert code == 1
    assert "REFUSED" in msg
    assert "PUSH_AUTH_main_20260826.txt" in msg.replace("\\", "/")


def test_branch_push_is_unaffected(tmp_path: Path) -> None:
    code, msg = decide_line(
        "refs/heads/sel-ghost-01",
        "7c086e8111c131650a666d06245b1c0eac2b640d",
        root=tmp_path,
        yyyymmdd="20260826",
    )
    assert code == 0
    assert "non-main" in msg


def test_auth_file_matching_sha_allows_and_can_delete(tmp_path: Path) -> None:
    day = "20260826"
    sha = "7c086e8111c131650a666d06245b1c0eac2b640d"
    auth = tmp_path / "dev"
    auth.mkdir()
    path = auth / f"PUSH_AUTH_main_{day}.txt"
    path.write_text(sha + "\n", encoding="ascii")
    code, msg = decide_line(
        "refs/heads/main",
        sha,
        root=tmp_path,
        yyyymmdd=day,
        delete_on_allow=True,
    )
    assert code == 0
    assert "allow main" in msg
    assert not path.is_file()


def test_install_hook_writes_pre_push(tmp_path: Path) -> None:
    git = tmp_path / ".git" / "hooks"
    git.mkdir(parents=True)
    (tmp_path / "dev" / "scripts").mkdir(parents=True)
    (tmp_path / "dev" / "scripts" / "push_guard.py").write_text("# stub\n", encoding="ascii")
    hook = install_hook(tmp_path)
    text = hook.read_text(encoding="ascii")
    assert "push_guard.py" in text
    assert "--pre-push" in text
