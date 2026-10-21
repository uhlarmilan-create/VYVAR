"""Guard: no L.A.Cosmic / destructive per-frame CR cleaning on light science pixels."""

from __future__ import annotations

import inspect

from config import AppConfig
import pipeline as pl


def test_lacosmic_helpers_removed() -> None:
    assert not hasattr(pl, "_remove_cosmics_lacosmic")
    assert not hasattr(pl, "_lacosmic_gain_readnoise_from_header")


def test_qc_enrich_has_no_lacosmic_knobs() -> None:
    sig = inspect.signature(pl._qc_enrich_one_frame)
    for name in ("enable_lacosmic", "lacosmic_sigclip", "lacosmic_objlim"):
        assert name not in sig.parameters


def test_appconfig_has_no_lacosmic_keys() -> None:
    cfg = AppConfig()
    for name in ("enable_lacosmic", "lacosmic_sigclip", "lacosmic_objlim"):
        assert not hasattr(cfg, name)
