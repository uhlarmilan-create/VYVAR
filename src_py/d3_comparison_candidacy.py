# -*- coding: ascii -*-
"""D3: comparison candidacy predicates (before RMS ceiling): detected source, identity gate,
Gaia-DAO residual.

COMP-SNR-ONE-2026-10 (G07-26, Milan 2026-10-31): the comparison-star S/N cut is not here - it is the
one gate max(cfg.snr_min, COMP_SNR_MIN) in ``photometry_comp.apply_comp_quality_gates``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from masterstar_gaia_accounting import SOURCE_DETECTED_P1, SOURCE_DETECTED_P2

LOGGER = logging.getLogger(__name__)

D3_REQUIRED_COLUMNS = (
    "source_state",
    "vy_identity_gate",
    "gaia_dao_resid_px",
)
D3_DETECTED_STATES = frozenset({SOURCE_DETECTED_P1, SOURCE_DETECTED_P2})


class D3CandidacyError(ValueError):
    """Raised when a required D3 column is absent. ``args[0]`` is the column name."""


def d3_resid_ceiling_px(*, fwhm_dao_px: float, solve_rms_px: float | None) -> float:
    fw = max(0.5, float(fwhm_dao_px))
    try:
        rms = float(solve_rms_px) if solve_rms_px is not None else float("nan")
    except (TypeError, ValueError):
        rms = float("nan")
    extra = 2.0 * rms if np.isfinite(rms) else 0.0
    return float(max(3.0 * fw, extra))


def apply_d3_comparison_candidacy(
    df: pd.DataFrame,
    *,
    fwhm_dao_px: float,
    solve_rms_px: float | None,
    log_label: str = "pool",
) -> tuple[pd.Series, dict[str, Any]]:
    """Return a boolean mask of D3-eligible rows. Missing columns raise ``D3CandidacyError``."""
    if df is None or getattr(df, "empty", True):
        empty = pd.Series(dtype=bool)
        return empty, {"n_in": 0, "n_out": 0, "drops": {}}
    for col in D3_REQUIRED_COLUMNS:
        if col not in df.columns:
            raise D3CandidacyError(col)
    n_in = int(len(df))
    ceil = d3_resid_ceiling_px(fwhm_dao_px=fwhm_dao_px, solve_rms_px=solve_rms_px)
    st = df["source_state"].astype(str).str.strip()
    gate = df["vy_identity_gate"].astype(str).str.strip().str.lower()
    resid = pd.to_numeric(df["gaia_dao_resid_px"], errors="coerce")
    m_state = st.isin(D3_DETECTED_STATES)
    m_gate = gate.ne("fail")
    m_resid = resid.le(ceil)
    mask = m_state & m_gate & m_resid
    drops = {
        "source_state": int((~m_state).sum()),
        "vy_identity_gate": int((m_state & ~m_gate).sum()),
        "gaia_dao_resid_px": int((m_state & m_gate & ~m_resid).sum()),
    }
    n_out = int(mask.sum())
    LOGGER.info(
        "[D3] %s: n_in=%d n_out=%d drops state=%d gate=%d resid=%d (ceil=%.3f px)",
        str(log_label),
        n_in,
        n_out,
        drops["source_state"],
        drops["vy_identity_gate"],
        drops["gaia_dao_resid_px"],
        float(ceil),
    )
    return mask, {"n_in": n_in, "n_out": n_out, "drops": drops, "resid_ceiling_px": float(ceil)}
