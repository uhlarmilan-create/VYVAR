# -*- coding: ascii -*-
"""Pinned embedded Python runtime downloads (R1). SHA256 verified on fetch."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePin:
    platform_key: str
    version: str
    url: str
    sha256: str
    archive_kind: str  # zip | tar.gz


RUNTIME_PINS: dict[str, RuntimePin] = {
    "win64": RuntimePin(
        platform_key="win64",
        version="3.12.10",
        url="https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip",
        sha256="4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3",
        archive_kind="zip",
    ),
    "linux-x64": RuntimePin(
        platform_key="linux-x64",
        version="3.12.8",
        url=(
            "https://github.com/astral-sh/python-build-standalone/releases/download/"
            "20241206/cpython-3.12.8+20241206-x86_64-unknown-linux-gnu-install_only.tar.gz"
        ),
        sha256="",  # verified on first download; builder writes cache sidecar
        archive_kind="tar.gz",
    ),
}

BUNDLE_REQUIREMENTS_EXCLUDE = frozenset({"pytest"})
