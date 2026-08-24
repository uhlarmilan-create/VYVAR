"""DAO-GAIA-XFER-01: sandbox gate pinned to hand params + identity stamps."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]

HAND_MS_STRICT = 0.9247889485801996
HAND_L001_STRICT = 0.9021488871834229
HAND_L076_STRICT = 0.9401381427475057
HAND_L148_STRICT = 0.9435917114351496


def _dummy_derived(*, pass2: float, seed: float, match: float = 3.0):
    from dao_gaia_calibration import DerivedTolerances, PopulationStats

    pop = PopulationStats(name="detection_identity", n=10, p50_px=1.0, p95_px=1.8)
    return DerivedTolerances(
        residual_p95_px=1.8,
        match_radius_px=match,
        pass2_center_tol_px=pass2,
        lock_pair_tol_px=match,
        lock_leftover_radius_px=match,
        forced_seed_centroid_max_px=seed,
        plate_scale_arcsec_per_px=0.56,
        fwhm_px=1.25,
        pass1_sigma=4.5,
        pass2_sigma=4.0,
        detection_identity=pop,
        faint_star_centroid=pop,
    )


def _csv_score_rows() -> list[dict]:
    return [
        {
            "frame": "MASTERSTAR",
            "g1_strict_le13": 0.9880478087649402,
            "g1_strict_le145": HAND_MS_STRICT,
            "g1_eye_le13": 0.9880478087649402,
            "g1_eye_le145": 0.930928626247122,
            "g1_eye_seed_le13": 1.0,
            "g1_eye_seed_le145": 0.9577897160399079,
            "g2": 0.0009090909090909091,
            "g3_g18": 0.014311270125223614,
            "g4_ok": True,
        },
        {
            "frame": "Light_001",
            "g1_strict_le13": 0.9870517928286853,
            "g1_strict_le145": HAND_L001_STRICT,
            "g1_eye_le13": 0.9870517928286853,
            "g1_eye_le145": 0.9102072141212586,
            "g1_eye_seed_le13": 0.999003984063745,
            "g1_eye_seed_le145": 0.9439754412893323,
            "g2": float("nan"),
            "g3_g18": 0.038069942452412575,
            "g4_ok": True,
        },
        {
            "frame": "Light_076",
            "g1_strict_le13": 0.9870517928286853,
            "g1_strict_le145": HAND_L076_STRICT,
            "g1_eye_le13": 0.9870517928286853,
            "g1_eye_le145": 0.9458940905602455,
            "g1_eye_seed_le13": 1.0,
            "g1_eye_seed_le145": 0.9723714504988488,
            "g2": float("nan"),
            "g3_g18": 0.034967845659163985,
            "g4_ok": True,
        },
        {
            "frame": "Light_148",
            "g1_strict_le13": 0.9870517928286853,
            "g1_strict_le145": HAND_L148_STRICT,
            "g1_eye_le13": 0.9870517928286853,
            "g1_eye_le145": 0.9481964696853415,
            "g1_eye_seed_le13": 1.0,
            "g1_eye_seed_le145": 0.9735226400613968,
            "g2": float("nan"),
            "g3_g18": 0.038338658146964855,
            "g4_ok": True,
        },
    ]


class _FakeParams:
    def __init__(self, **kw):
        self.pass1_sigma = kw.get("pass1_sigma", 4.5)
        self.pass2_sigma = kw.get("pass2_sigma", 4.0)
        self.match_radius_px = kw.get("match_radius_px", 3.0)
        self.pass2_center_tol_px = kw.get("pass2_center_tol_px", 2.0)
        self.seed_centroid_max_px = kw.get("seed_centroid_max_px", 2.0)
        self.seed_snr_min = kw.get("seed_snr_min", 4.0)

    @classmethod
    def hand_validated(cls):
        return cls()


def _install_fake_iter4(monkeypatch, seen: list) -> None:
    class FakeIter4:
        ValidationParams = _FakeParams

        @staticmethod
        def score_validation_params(params, gaia_db=None, rng=None):
            seen.append(params)
            return _csv_score_rows()

    monkeypatch.setattr(
        "dao_gaia_stage_validation._import_iter4",
        lambda repo_root: FakeIter4,
    )


def test_hand_csv_recorded_digits_regression_floor() -> None:
    """W4.2: 2026-08-19 CSV digits still on disk at current tip."""
    from dao_gaia_stage_validation import hand_csv_path

    path = hand_csv_path(ROOT)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "0.9247889485801996" in text
    assert "0.9021488871834229" in text
    assert "0.9401381427475057" in text
    assert "0.9435917114351496" in text
    assert "0.9577897160399079" in text


def test_derived_1px_tols_do_not_change_sandbox_score(monkeypatch, tmp_path: Path) -> None:
    """W4.1: 520 failure mode. Derived 1.0 px must not enter the sandbox rescore."""
    from dao_gaia_stage_validation import run_validation_gate

    seen: list = []
    _install_fake_iter4(monkeypatch, seen)
    gaia = tmp_path / "gaia.db"
    gaia.write_bytes(b"x")
    monkeypatch.setattr(
        "config.AppConfig",
        lambda: SimpleNamespace(gaia_db_path=str(gaia)),
    )

    r_tight = run_validation_gate(
        _dummy_derived(pass2=1.0, seed=1.0, match=3.5),
        pass1_sigma=4.5,
        pass2_sigma=4.0,
        seed_snr_min=4.0,
        repo_root=ROOT,
    )
    r_handish = run_validation_gate(
        _dummy_derived(pass2=2.0, seed=2.0, match=3.0),
        pass1_sigma=4.5,
        pass2_sigma=4.0,
        seed_snr_min=4.0,
        repo_root=ROOT,
    )
    assert r_tight.status == "PASS"
    assert r_handish.status == "PASS"
    assert r_tight.derived_scores["MASTERSTAR"]["g1_strict_le145"] == HAND_MS_STRICT
    assert r_handish.derived_scores["MASTERSTAR"]["g1_strict_le145"] == HAND_MS_STRICT
    assert r_tight.derived_scores["Light_001"]["g1_strict_le145"] == HAND_L001_STRICT
    assert r_handish.derived_scores["Light_001"]["g1_strict_le145"] == HAND_L001_STRICT
    assert len(seen) == 2
    for params in seen:
        assert params.pass2_center_tol_px == 2.0
        assert params.seed_centroid_max_px == 2.0
        assert params.match_radius_px == 3.0


def test_hand_recompute_matches_csv_when_sandbox_returns_lock_digits(monkeypatch, tmp_path: Path) -> None:
    """W4.2: gate PASSes when sandbox scores equal the 2026-08-19 CSV digits."""
    from dao_gaia_stage_validation import run_validation_gate

    seen: list = []
    _install_fake_iter4(monkeypatch, seen)
    gaia = tmp_path / "gaia.db"
    gaia.write_bytes(b"x")
    monkeypatch.setattr(
        "config.AppConfig",
        lambda: SimpleNamespace(gaia_db_path=str(gaia)),
    )
    result = run_validation_gate(
        _dummy_derived(pass2=1.0, seed=1.0),
        pass1_sigma=4.5,
        pass2_sigma=4.0,
        seed_snr_min=4.0,
        repo_root=ROOT,
    )
    assert result.status == "PASS"
    assert result.fail_reason is None
    assert result.max_regression_pp <= 0.005
    assert result.derived_scores["Light_001"]["g1_strict_le145"] == HAND_L001_STRICT
    assert result.derived_scores["Light_076"]["g1_strict_le145"] == HAND_L076_STRICT
    assert result.derived_scores["Light_148"]["g1_strict_le145"] == HAND_L148_STRICT


def test_certificate_identity_stamps_present(tmp_path: Path) -> None:
    """W4.3: every XFER-01 identity field is written."""
    from dao_gaia_calibration import (
        DaoGaiaCalibrationCertificate,
        EmptySkyAudit,
        write_calibration_certificate,
    )

    ms = ROOT / "Archive" / "Drafts" / "draft_000516" / "platesolve" / "NoFilter_60_2" / "MASTERSTAR.fits"
    if not ms.is_file():
        pytest.skip("draft 516 sandbox MASTERSTAR not present")
    cert = DaoGaiaCalibrationCertificate(
        setup="g_60_4",
        built_utc="2026-08-24T00:00:00+00:00",
        status="PASS",
        fail_reason=None,
        derived=_dummy_derived(pass2=1.0, seed=1.0, match=3.5),
        empty_sky=EmptySkyAudit(
            n_positions=2200,
            pass2_accept=0,
            pass2_rate=0.0,
            seed_accept=0,
            seed_rate=0.0,
            inv_det="PASS",
            inv_seed="PASS",
        ),
    )
    path = write_calibration_certificate(cert, tmp_path, fail_closed=True, repo_root=ROOT)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "gaia_fingerprint",
        "vsx_fingerprint",
        "sandbox",
        "hand_csv",
        "lock_rig",
        "production_tolerances",
        "derived_pass2_center_tol_px",
        "derived_forced_seed_centroid_max_px",
        "tol_drift_warn",
    ):
        assert key in payload, f"missing stamp {key}"
        assert payload[key] not in (None, "", {})
    sandbox = payload["sandbox"]
    assert sandbox["draft_id"] == 516
    for k in ("masterstar_sha256", "light_001_sha256", "light_076_sha256", "light_148_sha256"):
        assert isinstance(sandbox[k], str) and len(sandbox[k]) == 64
    assert payload["hand_csv"]["sha256"]
    assert Path(payload["hand_csv"]["path"]).name == "final_scores.csv"
    lock = payload["lock_rig"]
    assert lock["draft_id"] == 516
    assert float(lock["plate_scale_arcsec_per_px"]) > 0
    assert float(lock["fwhm_px"]) > 0
    assert payload["derived_pass2_center_tol_px"] == 1.0
    assert payload["derived_forced_seed_centroid_max_px"] == 1.0
    prod = payload["production_tolerances"]
    assert prod["scope"] == "production_photometry_current_set"
    assert prod["derived_pass2_center_tol_px"] == 1.0
    assert payload["tol_drift_warn"]["status"] == "WARN"
    assert payload["tol_drift_warn"]["blocks"] is False
    assert payload["sandbox_params"] == "hand_validated"


def test_missing_hand_csv_fails_loud(monkeypatch, tmp_path: Path) -> None:
    """W4.4: missing hand CSV is DAO-GAIA-IDENTITY, not a skip."""
    from dao_gaia_stage_validation import run_validation_gate
    from invariants_runtime import InvariantViolation

    missing = tmp_path / "no_such.csv"
    monkeypatch.setattr(
        "dao_gaia_stage_validation.hand_csv_path",
        lambda repo_root=None: missing,
    )
    gaia = tmp_path / "gaia.db"
    gaia.write_bytes(b"x")
    monkeypatch.setattr(
        "config.AppConfig",
        lambda: SimpleNamespace(gaia_db_path=str(gaia)),
    )
    seen: list = []
    _install_fake_iter4(monkeypatch, seen)
    with pytest.raises(InvariantViolation) as ei:
        run_validation_gate(
            _dummy_derived(pass2=2.0, seed=2.0),
            pass1_sigma=4.5,
            pass2_sigma=4.0,
            seed_snr_min=4.0,
            repo_root=ROOT,
        )
    assert ei.value.inv_id == "DAO-GAIA-IDENTITY"
    assert "missing" in str(ei.value).lower() or "baseline" in str(ei.value).lower()


def test_missing_gaia_fingerprint_fails_loud(monkeypatch, tmp_path: Path) -> None:
    """W4.4: missing catalog fingerprint fails loud on certificate write."""
    from dao_gaia_calibration import (
        DaoGaiaCalibrationCertificate,
        EmptySkyAudit,
        write_calibration_certificate,
    )
    from invariants_runtime import InvariantViolation

    monkeypatch.setattr(
        "catalog_provenance.fingerprint_gaia_db",
        lambda path: None,
    )
    cert = DaoGaiaCalibrationCertificate(
        setup="test",
        built_utc="2026-08-24T00:00:00+00:00",
        status="PASS",
        fail_reason=None,
        derived=_dummy_derived(pass2=2.0, seed=2.0),
        empty_sky=EmptySkyAudit(
            n_positions=10,
            pass2_accept=0,
            pass2_rate=0.0,
            seed_accept=0,
            seed_rate=0.0,
            inv_det="PASS",
            inv_seed="PASS",
        ),
    )
    with pytest.raises(InvariantViolation) as ei:
        write_calibration_certificate(cert, tmp_path, fail_closed=True, repo_root=ROOT)
    assert ei.value.inv_id == "DAO-GAIA-IDENTITY"
    assert "gaia" in str(ei.value).lower()
