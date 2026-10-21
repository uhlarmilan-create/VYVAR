"""Fail-safety hygiene tests (task #4: MASTERSTAR writeto + edge-ok flag)."""
from __future__ import annotations


def test_app_imports_resolve():
    import app  # noqa: F401
