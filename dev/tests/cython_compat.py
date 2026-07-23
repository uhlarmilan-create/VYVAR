# -*- coding: ascii -*-
"""Helpers for pytest under compiled (Cython release) builds."""
from __future__ import annotations

import importlib

import pytest


def module_is_compiled(module_name: str) -> bool:
    mod = importlib.import_module(module_name)
    path = str(getattr(mod, "__file__", "") or "")
    return path.endswith((".pyd", ".so"))


def skip_if_compiled(module_name: str, reason: str) -> None:
    if module_is_compiled(module_name):
        pytest.skip(f"{reason} ({module_name} loaded from compiled extension)")
