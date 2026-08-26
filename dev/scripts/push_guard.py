# -*- coding: ascii -*-
"""Refuse unauthorized pushes to origin/main (CLOSE-OUT 2026-08-25 push incident).

Rule: `git push origin HEAD` and bare `git push` must not update main.
A push whose remote ref is refs/heads/main is allowed only when
`dev/PUSH_AUTH_main_<YYYYMMDD>.txt` exists (gitignored) and contains the
local SHA being pushed. The file is deleted after a successful allow.

Branch pushes (sel-ghost-01, etc.) are unaffected.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

MAIN_REFS = ("refs/heads/main", "refs/heads/master")


def repo_root_from_cwd() -> Path:
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / ".git").exists() or (p / "dev" / "scripts" / "push_guard.py").is_file():
            return p
    return here


def auth_path(root: Path, yyyymmdd: str) -> Path:
    return root / "dev" / f"PUSH_AUTH_main_{yyyymmdd}.txt"


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def read_auth_sha(path: Path) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="ascii", errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s.lower()
    return ""


def decide_line(
    remote_ref: str,
    local_sha: str,
    *,
    root: Path,
    yyyymmdd: str | None = None,
    delete_on_allow: bool = False,
) -> tuple[int, str]:
    """Return (exit_code, message). 0 = allow."""
    ref = (remote_ref or "").strip()
    sha = (local_sha or "").strip().lower()
    if ref not in MAIN_REFS:
        return 0, f"allow non-main ref {ref}"
    day = yyyymmdd or today_utc()
    path = auth_path(root, day)
    want = read_auth_sha(path)
    if not want:
        return 1, (
            "REFUSED: push to main requires gitignored "
            f"{path.relative_to(root).as_posix()} containing the target SHA. "
            "See PROCESS push incident 2026-08-25. Use "
            "`git push origin <local>:<remote>` and never `git push origin HEAD`."
        )
    if len(sha) < 7 or not (sha.startswith(want) or want.startswith(sha)):
        return 1, (
            f"REFUSED: PUSH_AUTH SHA {want[:12]} does not match local {sha[:12]}"
        )
    if delete_on_allow:
        try:
            path.unlink()
        except OSError:
            pass
    return 0, f"allow main {sha[:12]} (auth consumed)"


def guard_pre_push(stdin_text: str, *, root: Path, delete_on_allow: bool) -> int:
    code = 0
    msg = "no refs"
    for raw in stdin_text.splitlines():
        parts = raw.split()
        if len(parts) < 4:
            continue
        _local_ref, local_sha, remote_ref, _remote_sha = parts[:4]
        code, msg = decide_line(
            remote_ref,
            local_sha,
            root=root,
            delete_on_allow=delete_on_allow,
        )
        print(msg, file=sys.stderr)
        if code != 0:
            return code
    return 0


def install_hook(root: Path) -> Path:
    git_dir = root / ".git"
    if git_dir.is_file():
        text = git_dir.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.lower().startswith("gitdir:"):
                git_dir = Path(line.split(":", 1)[1].strip())
                if not git_dir.is_absolute():
                    git_dir = (root / git_dir).resolve()
                break
    hook = git_dir / "hooks" / "pre-push"
    hook.parent.mkdir(parents=True, exist_ok=True)
    script = root / "dev" / "scripts" / "push_guard.py"
    body = (
        "#!/bin/sh\n"
        "ROOT=$(git rev-parse --show-toplevel)\n"
        f'exec python "{script.as_posix()}" --pre-push --root "$ROOT"\n'
    )
    hook.write_text(body, encoding="ascii")
    try:
        os.chmod(hook, 0o755)
    except OSError:
        pass
    return hook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VYVAR main-branch push guard")
    parser.add_argument("--pre-push", action="store_true")
    parser.add_argument("--install-hook", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--dry-ref", default="")
    parser.add_argument("--dry-sha", default="")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else repo_root_from_cwd()
    if args.install_hook:
        hook = install_hook(root)
        print(str(hook))
        return 0
    if args.dry_ref:
        code, msg = decide_line(args.dry_ref, args.dry_sha, root=root, delete_on_allow=False)
        print(msg)
        return code
    if args.pre_push:
        stdin_text = sys.stdin.read()
        return guard_pre_push(stdin_text, root=root, delete_on_allow=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
