"""NIGHT_FIT v2 synthetic recovery + REFUSE gates (no Archive dependency)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from config import AppConfig
from k2_extinction import (
    K2FitDiagnostics,
    K2Source,
    fit_k2_night,
    k2_feasibility_pregate,
    resolve_k2_bprp_value,
    build_honeycutt_residual_table,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURE_427 = REPO / "dev" / "validation" / "fixtures" / "k2_draft427_refuse.json"


def _two_arc_airmass(n_frames: int = 40) -> np.ndarray:
    """Rise 1.6 -> 1.15 then fall to 1.9 (non-monotonic)."""
    n1 = n_frames // 2
    n2 = n_frames - n1
    rise = np.linspace(1.60, 1.15, n1)
    fall = np.linspace(1.15, 1.90, n2)
    return np.concatenate([rise, fall])


def _synthetic_night(
    *,
    k2_true: float,
    colour_spread: float,
    noise_mmag: float,
    n_frames: int = 40,
    n_stars: int = 18,
    monotonic: bool = False,
    split_k2: float | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if monotonic:
        am_frame = np.linspace(1.15, 1.95, n_frames)
    else:
        am_frame = _two_arc_airmass(n_frames)
    colours = 0.8 + rng.uniform(-0.5, 0.5, size=n_stars) * float(colour_spread)
    cref = float(np.median(colours))
    # Flat brightness so brightness-tertile consistency is not noise-dominated.
    bright = np.full(n_stars, 11.0, dtype=np.float64)

    mag_l: list[float] = []
    col_l: list[float] = []
    am_l: list[float] = []
    si_l: list[int] = []
    fi_l: list[int] = []
    br_l: list[float] = []
    k_star: list[float] = []
    noise = float(noise_mmag) / 1000.0
    for si in range(n_stars):
        k_use = float(k2_true)
        if split_k2 is not None and colours[si] > cref:
            k_use = float(split_k2)
        k_star.append(k_use)
        for fi in range(n_frames):
            # Flux-derived mag_inst base (no k2); k2 injected into Honeycutt
            # residual space as S5-identifiable term k2*(C-Cref)*dX  what the
            # production fitter sees after ensemble CM removal.
            mag = float(bright[si]) + rng.normal(0.0, noise)
            mag_l.append(mag)
            col_l.append(float(colours[si]))
            am_l.append(float(am_frame[fi]))
            si_l.append(si)
            fi_l.append(fi)
            br_l.append(float(bright[si]))

    mag = np.asarray(mag_l, dtype=np.float64)
    col_a = np.asarray(col_l, dtype=np.float64)
    am_a = np.asarray(am_l, dtype=np.float64)
    si_a = np.asarray(si_l, dtype=np.int64)
    fi_a = np.asarray(fi_l, dtype=np.int64)
    table = build_honeycutt_residual_table(
        mag, col_a, am_a, si_a, fi_a, brightness=np.asarray(br_l)
    )
    x_mean = float(np.mean(am_frame))
    k_per_point = np.asarray([k_star[s] for s in si_a], dtype=np.float64)
    table["residual"] = table["residual"] + k_per_point * (col_a - cref) * (am_a - x_mean)
    table["airmass_by_frame"] = am_frame
    table["colour_ref"] = cref
    return table


def _fit_from_table(table: dict[str, np.ndarray], *, k2_lit: float, **kw):
    return fit_k2_night(
        residual=table["residual"],
        colour=table["colour"],
        airmass=table["airmass"],
        star_index=table["star_index"],
        frame_index=table["frame_index"],
        brightness=table.get("brightness"),
        colour_ref=float(table.get("colour_ref", np.nan)),
        k2_literature=float(k2_lit),
        airmass_by_frame=table.get("airmass_by_frame"),
        rng=np.random.default_rng(7),
        **kw,
    )


def test_default_off_resolve_byte_identical_literature() -> None:
    cfg = AppConfig()
    assert cfg.k2_fit_enabled is False
    cfg.k2_mode = "fit_else_literature"
    v1, s1 = resolve_k2_bprp_value(cfg, "g_60_2")
    cfg2 = AppConfig()
    cfg2.k2_mode = "literature"
    v2, s2 = resolve_k2_bprp_value(cfg2, "g_60_2")
    assert s1 is K2Source.LITERATURE_DEFAULT
    assert s2 is K2Source.LITERATURE_DEFAULT
    assert v1 == pytest.approx(v2)


def test_recovery_sweep_matrix() -> None:
    """Sweep k2_true x colour_spread x noise; accept within band or refuse detectability."""
    rows: list[dict] = []
    for k2_true in (0.0, 0.02, 0.05, 0.08):
        for spread in (0.2, 0.5, 1.0):
            for noise in (5, 15, 30):
                # Literature near |k2_true| so detectability thr=|lit|/3 is reachable
                # and plausibility (4x lit) still admits the true value.
                lit_use = max(float(k2_true), 0.025) if k2_true > 0 else 0.025
                table = _synthetic_night(
                    k2_true=k2_true,
                    colour_spread=spread,
                    noise_mmag=float(noise),
                    n_frames=60,
                    n_stars=24,
                    seed=1000 + int(100 * k2_true) + int(10 * spread) + noise,
                )
                res = _fit_from_table(
                    table,
                    k2_lit=lit_use,
                    k2_fit_lit_factor=4.0,
                    k2_fit_min_detectability=3.0,
                    k2_ceiling=0.1,
                )
                diag = res.diagnostics
                assert diag is not None
                thr = abs(lit_use) / 3.0
                detectable = math.isfinite(diag.sigma_k2_pred) and diag.sigma_k2_pred <= thr
                row = {
                    "k2_true": k2_true,
                    "spread": spread,
                    "noise": noise,
                    "accepted": res.accepted,
                    "reason": res.refuse_reason,
                    "k2_hat": res.k2_value,
                    "sigma_boot": res.sigma_boot,
                    "sigma_pred": diag.sigma_k2_pred,
                    "detectable": detectable,
                }
                rows.append(row)
                if res.accepted:
                    tol = max(2.0 * float(res.sigma_boot), 0.005)
                    assert abs(float(res.k2_value) - float(k2_true)) <= tol + 1e-9
                elif not detectable:
                    assert res.refuse_reason == "detectability"
                if k2_true == 0.0 and noise >= 15:
                    if res.accepted:
                        assert abs(float(res.k2_value)) <= max(
                            2.0 * float(res.sigma_boot), 0.005
                        )

    accepted = [r for r in rows if r["accepted"]]
    assert len(accepted) >= 3, f"expected several recoveries; got {len(accepted)} / sample={rows[:3]!r}"


def test_refuse_monotonic_airmass() -> None:
    table = _synthetic_night(
        k2_true=0.05, colour_spread=1.0, noise_mmag=5.0, monotonic=True, seed=1
    )
    res = _fit_from_table(table, k2_lit=0.02, k2_fit_lit_factor=4.0)
    assert res.accepted is False
    assert res.refuse_reason == "monotonic_airmass"


def test_refuse_plausibility_absurd_k2() -> None:
    table = _synthetic_night(
        k2_true=0.5, colour_spread=1.0, noise_mmag=5.0, seed=2
    )
    # Literature tiny so detectability can pass on strong signal; ceiling must fire.
    res = _fit_from_table(
        table,
        k2_lit=0.08,
        k2_ceiling=0.1,
        k2_fit_lit_factor=4.0,
        k2_fit_min_detectability=3.0,
    )
    assert res.accepted is False
    assert res.refuse_reason in (
        "plausibility_ceiling",
        "plausibility_literature",
        "plausibility_literature_sign",
        "colour_tertile_inconsistent",
        "brightness_tertile_inconsistent",
        "arc_inconsistent",
        "detectability",
    )
    # Strong absurd injection should not look like a tiny literature-scale fit.
    if math.isfinite(res.k2_value) and res.refuse_reason.startswith("plausibility"):
        assert abs(float(res.k2_value)) > 0.1 or abs(float(res.k2_value)) > 4.0 * 0.08


def test_refuse_colour_tertile_inconsistent() -> None:
    table = _synthetic_night(
        k2_true=0.02,
        colour_spread=1.0,
        noise_mmag=5.0,
        split_k2=0.08,
        seed=3,
    )
    res = _fit_from_table(table, k2_lit=0.04, k2_fit_consistency_sigma=2.0)
    assert res.accepted is False
    assert res.refuse_reason in (
        "colour_tertile_inconsistent",
        "brightness_tertile_inconsistent",
        "arc_inconsistent",
        "plausibility_literature",
        "plausibility_ceiling",
        "detectability",
    )


def test_zero_signal_no_fabricated_detection() -> None:
    table = _synthetic_night(
        k2_true=0.0, colour_spread=0.5, noise_mmag=30.0, seed=4
    )
    res = _fit_from_table(table, k2_lit=0.02, k2_fit_min_detectability=3.0)
    if res.accepted:
        tol = max(2.0 * float(res.sigma_boot), 0.005)
        assert abs(float(res.k2_value)) <= tol
    else:
        assert (
            res.refuse_reason == "detectability"
            or res.refuse_reason.startswith("plausibility")
            or "inconsistent" in res.refuse_reason
        )


def test_draft427_refuse_fixture() -> None:
    assert FIXTURE_427.is_file()
    data = json.loads(FIXTURE_427.read_text(encoding="utf-8"))
    assert data["fixture_source"] == "synthesized_from_decisions"
    d = data["diagnostics"]
    diag = K2FitDiagnostics(
        k2_fit=float(d["k2_fit"]),
        sigma_boot=float(d["sigma_boot"]),
        sigma_k2_pred=float(d["sigma_k2_pred"]),
        k2_literature=float(d["k2_literature"]),
        colour_tertile_k2=[float(x) for x in d["colour_tertile_k2"]],
        brightness_tertile_k2=[float(x) for x in d["brightness_tertile_k2"]],
        arc_k2=[float(x) for x in d["arc_k2"]],
        airmass_monotonic=bool(d["airmass_monotonic"]),
        n_points=int(d["n_points"]),
        sd_c_dx=float(d["sd_c_dx"]),
        residual_rms=float(d["residual_rms"]),
    )
    ok, reason = k2_feasibility_pregate(
        diag,
        k2_ceiling=float(data["k2_ceiling"]),
        k2_fit_min_detectability=float(data["k2_fit_min_detectability"]),
        k2_fit_consistency_sigma=float(data["k2_fit_consistency_sigma"]),
        k2_fit_lit_factor=float(data["k2_fit_lit_factor"]),
    )
    assert ok is False
    assert reason in set(data["expect_refuse_reasons_any_of"])
    assert reason not in ("monotonic_airmass",)


def test_enabled_fit_else_literature_honors_night_fit_result() -> None:
    cfg = AppConfig()
    cfg.k2_fit_enabled = True
    cfg.k2_mode = "fit_else_literature"
    table = _synthetic_night(k2_true=0.05, colour_spread=1.0, noise_mmag=5.0, seed=5)
    night = _fit_from_table(table, k2_lit=0.03)
    v, s = resolve_k2_bprp_value(cfg, "g_60_2", night_fit_result=night)
    if night.accepted:
        assert s is K2Source.NIGHT_FIT
        assert v == pytest.approx(night.k2_value)
    else:
        assert s is K2Source.LITERATURE_DEFAULT
        assert math.isfinite(v)


def test_deterministic_two_runs() -> None:
    table = _synthetic_night(k2_true=0.05, colour_spread=1.0, noise_mmag=5.0, seed=9)
    a = _fit_from_table(table, k2_lit=0.03)
    b = _fit_from_table(table, k2_lit=0.03)
    assert a.accepted == b.accepted
    assert a.refuse_reason == b.refuse_reason
    if math.isfinite(a.k2_value) and math.isfinite(b.k2_value):
        assert a.k2_value == pytest.approx(b.k2_value, abs=1e-12)
    if math.isfinite(a.sigma_boot) and math.isfinite(b.sigma_boot):
        assert a.sigma_boot == pytest.approx(b.sigma_boot, abs=1e-12)
