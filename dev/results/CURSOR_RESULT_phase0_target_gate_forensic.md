> **PROVENANCE WARNING (added 2026-07-30).** The numbers in this document derive from **draft_000450**, produced after the in-place preprocess architecture landed (`013cb0c`, 2026-07-22) and before the sky-surface idempotency guard (`84174ae`, 2026-07-30). During that window a repeated preprocess pass could subtract the sky surface twice, at a measured cost of order 500 ADU. **draft_000450** is no longer available, so its status is UNKNOWN, not clean. Treat these numbers as indicative, not validated.

CURSOR RESULT - PHASE0-TARGET-GATE-FORENSIC - 2026-07-26

Read-only forensic for Milan DRY runs (draft_000450, BO CVn, wide rig, Friday evening).
Tree reference: private tip `10608bb` (2026-07-24). Line citations below are verified
against `10608bb` unless noted.

---

## 1. STEP 0 - environment and provenance

| # | Item | Value |
|---|------|-------|
| 1a | `git rev-parse HEAD` | `cb78b25172e561fb1d8e274d5d2bce2913c2a7c3` (**not** `10608bb`; +1 commit ahead) |
| 1b | `git status --porcelain` | `?? dev/scripts/dy_peg_night_run_bvr.py`<br>`?? dev/scripts/qatar8_night_run_v.py`<br>`?? vyvar.sqlite3-shm`<br>`?? vyvar.sqlite3-wal` |
| 1c | Diff vs `10608bb` (relevant modules) | `src_py/config.py` (+1/-1 observer default), `src_py/photometry_core.py` (+9 zero-target early return), `src_py/pipeline.py` (+97 VSX fail-loud / BORDER defer). **None of these touch the Phase 0 ROT gate or `select_active_targets` matching logic.** |
| 2 | `git log --oneline -5` | `cb78b25` Field-run findings #11-#13<br>`10608bb` docs: name 4e5971f in DB threading result append<br>`4e5971f` docs: BUNDLE-DB-THREADING close append<br>`2be87fc` fix(field): expose lastrowid/rowcount on locked sqlite cursor<br>`ff7dca6` fix(field): thread-safe cached sqlite connection |
| 3 | Interpreted vs compiled (this machine) | **Interpreted.** No `*.pyd` / `*.so` under `src_py/` at forensic time. Cannot compare artefact mtimes to `draft_000450` run timestamp (draft absent). |
| 4 | Resolved `config.json` (this machine) | Path: `C:\ASTRO\python\VYVAR\config.json`<br>`vsx_out_of_scope_types`: `[]` (JSON array, empty)<br>`variability_mag_limit`: `14.5`<br>`faintest_mag_limit`: **not a registered/persisted key** in `config.json` or `AppConfig.to_json()` at `10608bb`<br>`vsx_local_db_path`: `C:\ASTRO\python\VYVAR\VSX\vyvar_vsx_local_v2.db`<br>`gaia_db_path`: `C:\ASTRO\python\VYVAR\GAIA_DR3\vyvar_gaia_dr3.db` |
| 5 | Draft config snapshot (`draft_000450`) | **BLOCKED - draft tree not found on this DEV PC.** Searched `C:\ASTRO\python\VYVAR\Archive\Drafts`, broader `C:\ASTRO`, and `draft_000450` by name: **no directory.** Cannot compare on-disk config vs draft snapshot. |

**Dirty-tree note:** Working tree is not clean (untracked scripts + sqlite WAL). This does **not** affect Phase 0 logic on the committed tip, but it means this forensic machine is **not** the Friday run environment unless Milan confirms otherwise.

**Friday-run environment (HYPOTHESIS):** Milan's DEV PC with `draft_000450` under his `archive_root`. All draft-quantity tables in sections A.1 / B.1 require that tree or an infolog export from Milan.

---

## 2. Findings A - `vsx_out_of_scope_types` (ROT)

### A-F1. Draft evidence unavailable on forensic machine

**Statement:** No `variable_targets.csv`, `active_targets.csv`, light curves, or `pipeline_meta.json` from `draft_000450` could be read.

**Evidence:** Glob/search returned zero paths for `draft_000450` under `C:\ASTRO`.

**Confidence:** CONFIRMED (absence on this machine).

**Blast radius:** This report's A.1 counts and B.1 histograms are **not populated**; code-path analysis below substitutes.

---

### A-F2. Production `variable_targets.csv` uses `catalog: "VSX"` (filter-eligible)

**Statement:** Auto-VSX rows written by `write_photometry_plan_files` carry `catalog="VSX"`, so `is_vsx_auto_selected_target()` should return `True` for them.

**Evidence:**

```6791:6791:src_py/pipeline.py
                        "catalog": "VSX",
```

```65:80:src_py/vsx_type_scope.py
def is_vsx_auto_selected_target(row: dict | object) -> bool:
    ...
    if cat == "VSX":
        return True
    ...
    if cat:
        return False
    return bool(str(get(get("vsx_name", "") or "").strip())
```

**Confidence:** CONFIRMED (code at `10608bb`).

**Blast radius:** Failure mode #3 from the task ("non-VSX `catalog` disables filter") is **unlikely for standard pipeline-produced VT rows**, unless rows were hand-edited, merged from exoplanet promotion with a different `catalog`, or regenerated outside `write_photometry_plan_files`.

---

### A-F3. ROT filter is mask-first, not removal-from-`active_targets`

**Statement:** When the gate fires, ROT auto-VSX targets **remain** in `active_targets.csv` with `skip_photometry=True` and `skip_reason='vsx_type_out_of_scope'`. They are skipped in Phase 1 comp selection and Phase 2A photometry, but still appear in the target list and may get a stub row in `photometry_summary.csv` (`n_frames=0`, empty `lc_csv`).

**Evidence:**

```12598:12600:src_py/photometry_core.py
    - ``vsx_out_of_scope_types`` (config): VSX auto-selected targets whose type tokens match
      are kept in active_targets with ``skip_photometry=True`` and
      ``skip_reason='vsx_type_out_of_scope'`` (mask-first). Manual targets are never filtered.
```

```12844:12852:src_py/photometry_core.py
        _voos = list(getattr(_cfg, "vsx_out_of_scope_types", []) or [])
        if (
            (not skip_ph)
            and _voos
            and is_vsx_auto_selected_target(vrow)
            and vsx_type_is_out_of_scope(str(vrow.get("vsx_type", "") or ""), _voos)
        ):
            skip_ph = True
            skip_reason = "vsx_type_out_of_scope"
```

```14792:14796:src_py/photometry_core.py
            if _skip_reason == "vsx_type_out_of_scope":
                _n_oos_skipped += 1
                continue
```

```8993:9021:src_py/photometry_core.py
    if skip_photo:
        ...
        logging.info(f"[FAZA 2A] Preskakujem fotometriu ({_skip_reason}): {target_name}")
        _skip_sum: dict[str, Any] = {
            ...
            "n_frames": 0,
            ...
            "lc_csv": "",
            "lc_png": "",
        }
        summary_rows.append(_skip_sum)
```

**Confidence:** CONFIRMED.

**Blast radius:** If Milan counted "processed" as "present in `active_targets` / `photometry_summary`", a **working** ROT filter can still look like "no effect". Symptom A needs distinguishing: **real LC files** vs **stub summary rows**.

---

### A-F4. Config load: empty/wrong type silently yields `[]` (filter inactive)

**Statement:** If `vsx_out_of_scope_types` is missing, null, or an unsupported JSON type, `AppConfig` sets `[]` with **no warning**. An empty list makes the gate conjunct `_voos` false at `photometry_core.py:12845`.

**Evidence:**

```1316:1324:src_py/config.py
        _voos = data.get("vsx_out_of_scope_types", self.vsx_out_of_scope_types)
        if _voos is None:
            self.vsx_out_of_scope_types = []
        elif isinstance(_voos, str):
            self.vsx_out_of_scope_types = [p.strip() for p in _voos.split(",") if p.strip()]
        elif isinstance(_voos, (list, tuple)):
            self.vsx_out_of_scope_types = [str(p).strip() for p in _voos if str(p).strip()]
        else:
            self.vsx_out_of_scope_types = []
```

```576:576:src_py/config.py
    vsx_out_of_scope_types: list[str] = field(default_factory=list)
```

**On this machine:** `config.json` literal is `[]`, not `["ROT"]`.

**Confidence:** CONFIRMED (code + local config). **Whether Friday run had `["ROT"]` is HYPOTHESIS** pending Milan's draft snapshot / config copy.

**Blast radius:** Any run where effective `_voos` is `[]` behaves as if the feature were absent.

---

### A-F5. `select_active_targets` call sites - `cfg` passed on all production paths

**Statement:** The only definition/call of `select_active_targets` in `src_py/` is inside `run_phase0_and_phase1`, which passes `cfg=_cfg_p01`. Callers `run_full_photometry_pipeline` -> used from `app.py`, `night_run.py`, `ui_aperture_photometry.py` all pass `cfg`.

**Evidence:**

```14591:14603:src_py/photometry_core.py
    active = select_active_targets(
        ...
        cfg=_cfg_p01,
    )
```

```12668:12668:src_py/photometry_core.py
    _cfg = cfg if cfg is not None else AppConfig()
```

Direct callers pass `cfg`; the `AppConfig()` fallback triggers only if a future caller passes `cfg=None`.

**Confidence:** CONFIRMED.

**Blast radius:** Failure mode #1 (`cfg=None` default) is **not** the normal RUN VYVAR / Aperture Photometry / night_run path unless a custom script calls `select_active_targets` directly (none in `src_py/`).

---

### A-F6. Observability gap - no gate for "non-empty config, zero markers"

**Statement:** `INV-CFG-01` in `invariants_runtime.py:574-605` checks only: if `vsx_out_of_scope_types=[]`, then `skip_reason=vsx_type_out_of_scope` must **not** appear. There is **no** inverse check (non-empty config must produce markers when ROT VSX rows exist).

**Evidence:**

```574:605:src_py/invariants_runtime.py
    # Empty vsx_out_of_scope_types => no out-of-scope skip markers.
    ...
    if not _voos_list and photometry_dir is not None:
        ...
            if "vsx_type_out_of_scope" in vals:
                issues.append(
                    f"skip_reason=vsx_type_out_of_scope in {rel.name} while "
                    "vsx_out_of_scope_types=[]"
                )
```

**Confidence:** CONFIRMED.

**Proposal (not implemented):** At Phase 0 close, log  
`out_of_scope: cfg=[ROT,...] auto_vsx_rows=N matched=M masked=M`;  
WARN when `cfg` non-empty and `M==0` but ROT-family rows exist in `variable_targets.csv`.

---

### A.1 table (draft_000450) - NOT POPULATED

| Quantity | Result |
|----------|--------|
| All A.1 rows | **BLOCKED** - `draft_000450` not on forensic machine |

**Experiment to settle A:** On Milan's PC, run read-only script on  
`Archive/Drafts/draft_000450/platesolve/<setup>/` counting ROT tokens via `tokenize_vsx_type`,  
`skip_reason` breakdown, and `lightcurves/*.csv` existence for ROT `catalog_id`s.

---

## 3. Findings B - non-DAO / faint VSX became active targets

### B-F1. Draft evidence unavailable (same as A-F1)

B.1 quantity table: **BLOCKED** on this machine. Pre-regression ~160-target BO CVn draft: **not found** locally for side-by-side.

---

### B-F2. `variability_mag_limit` does **not** bound VSX target export

**Statement:** `variability_mag_limit` (default 14.5, persisted in config) is used by the **variability candidate detector**, not by VSX `variable_targets.csv` production.

**Evidence:**

```992:992:src_py/config.py
    variability_mag_limit: float = 14.5
```

```5685:5687:src_py/photometry_core.py
        mag_limit = float(cfg.variability_mag_limit)
    ...
        mag_limit = 14.5
```

(inside variability export path, not `write_photometry_plan_files`)

VSX export uses frame bbox query with **no** `variability_mag_limit` argument (`pipeline.py:6493-6500` `_query_vsx_local_frame_bbox`).

**Confidence:** CONFIRMED.

**Blast radius:** Lowering/raising `variability_mag_limit` in Settings does **not** shrink/grow auto-VSX target set for photometry.

---

### B-F3. MASTERSTAR depth uses hard 18.0 mag floor, not user `variability_mag_limit`

**Statement:** MASTERSTAR / detect-time catalog matching applies `faintest_mag_limit` with an effective floor of **18.0 mag** (`MASTERSTAR_FAINTEST_MAG_FLOOR`), regardless of `variability_mag_limit`. This depth is for **astrometry anchor density**, not photometric detection limit.

**Evidence:**

```30:30:src_py/dao_reconcile.py
MASTERSTAR_FAINTEST_MAG_FLOOR = 18.0
```

```12022:12036:src_py/pipeline.py
    if faintest_mag_limit is None:
        _ms_faintest_mag_eff: float | None = 18.0
    else:
        ...
            _ms_faintest_mag_eff = max(float(faintest_mag_limit), 18.0)
    ...
        faintest_mag_limit=_ms_faintest_mag_eff,
```

```4422:4422:src_py/pipeline.py
                faintest_mag_limit=18.0 if _is_masterstar else None,
```

`faintest_mag_limit` is **not** exposed as a persisted config key in `config.py` at `10608bb`.

**Confidence:** CONFIRMED.

**Blast radius:** Wide-field MASTERSTAR builds a deep DAO+Gaia masterstar list (G<=18). Phase 0 can match many faint VSX entries to **some** masterstar row within radius, even when the VSX star is not visually detected on the frame.

---

### B-F4. Phase 0 "DAO+Gaia match" is **spatial nearest-neighbor**, not triple identity

**Statement:** The governing decision text (`VYVAR_DECISIONS.md:580-583`) says VSX without masterstar (DAO+Gaia) cross-match are excluded. In code, "match" means: nearest row in `masterstars_full_match.csv` within adaptive `match_radius_arcsec`, with non-empty `catalog_id` from that masterstar row. There is **no** requirement that VSX `catalog_id` (from VT CSV Gaia fallback) equals the matched masterstar `catalog_id`, nor that the VSX position equals the DAO centroid.

**Evidence:**

```12773:12796:src_py/photometry_core.py
    for vidx, vrow in vt_in.iterrows():
        ...
        best_idx = int(np.argmin(dists))
        best_dist_arcsec = dists[best_idx] * 3600.0
        if best_dist_arcsec > match_radius_arcsec:
            continue
        ms_row = ms.iloc[best_idx]
```

```12721:12728:src_py/photometry_core.py
        _adaptive = _plate_nominal * 5.0
        _cfg_floor = float(_cfg.phase01_match_radius_arcsec)
        match_radius_arcsec = max(_adaptive, _cfg_floor)
```

Default floor: `phase01_match_radius_arcsec: 10.0` (`config.py:850`).

```12910:12912:src_py/photometry_core.py
    n_excluded_no_dao_match = int((~vt_in.index.isin(matched_vt_idx)).sum())
    ...
        excluded_rows.append(_excluded_target_row(vrow, "no_dao_gaia_match"))
```

**Confidence:** CONFIRMED (code semantics). **Whether this explains >360 actives on draft_450 is HYPOTHESIS** without draft CSVs.

**Blast radius:** All rigs/fields using auto-VSX + deep MASTERSTAR; wide plate scale increases adaptive radius in arcsec.

---

### B-F5. `zone_flag == "catalog_only"` is not assigned by current MASTERSTAR zone logic

**Statement:** `_assign_masterstar_zones` (`pipeline.py:6250-6297`) sets only `linear`, `noisy1/2/3`, `saturated`. The string `catalog_only` appears in **log/meta** as "Gaia cone rows without DAO detection" (`pipeline.py:8991`), not as a masterstar `zone` value. `_active_target_zone_flag` (`photometry_core.py:12254-12268`) passes unknown zones through unchanged, so **legacy** `catalog_only` in an old masterstars CSV would **not** be masked (only `saturated` sets `skip_ph` at `12836`).

**Evidence:**

```12254:12268:src_py/photometry_core.py
def _active_target_zone_flag(ms_row: pd.Series, zone_val_raw: str) -> str:
    ...
    if not z:
        return "neznama_zona"
    return z
```

```12836:12837:src_py/photometry_core.py
        skip_ph = zone_flag == "saturated"
        skip_reason = "zone_flag" if skip_ph else ""
```

```8991:8991:src_py/pipeline.py
            "| catalog_only (undetected): %d",
```

**Confidence:** CONFIRMED for **current** zone assignment; **HYPOTHESIS** that draft_450 `active_targets` contains `zone_flag=catalog_only` depends on draft data.

**Blast radius:** If old CSVs carry `catalog_only`, they would become active with photometry (contradicts `dev/scripts/_integration_test_todos.py:107` expectation).

---

### B-F6. Phase 2A variable flux requires direct `catalog_id` hit (DAO path)

**Statement:** Even if a target is active, Phase 2A measures variables only on direct `catalog_id` presence in per-frame proc CSV (`read_flux_from_csv`). Missing DAO hit -> `no_data` / NaN LC (per `VYVAR_DECISIONS.md:585-587`).

**Evidence:** Decision text at `docs/VYVAR_DECISIONS.md:585-587`; implementation `read_flux_from_csv` / `flag: no_data` at `photometry_core.py:2113+`.

**Confidence:** CONFIRMED (decision + code path).

**Implication:** Surplus **active** targets can exist with **empty/no_data LCs** if masterstar match attached wrong `catalog_id` or faint DAO is intermittent. Symptom B may be "too many actives" even when LCs are empty.

---

### B.3 one-sentence answers (code-level)

| Membership | Where enforced | Required in Friday run? |
|------------|----------------|-------------------------|
| **VSX** | `variable_targets.csv` from VSX frame bbox (`pipeline.py:6493+`) | Yes for auto pipeline |
| **Gaia** | Implicit via masterstar `catalog_id` + optional VT Gaia fallback (`pipeline.py:6603+`) | Partial - spatial ms match, not VT Gaia ID equality |
| **DAO** | Masterstars are DAO detections; Phase 0 nearest-neighbor to ms row | **Weakly** - neighbor within radius, not verified DAO detection of VSX position |

**Which membership fails the Milan contract?** **CONFIRMED:** the code does **not** enforce **GAIA == DAO == VSX identity conjunction**; it enforces **VSX position -> nearest DAO+Gaia masterstar within radius** (`photometry_core.py:12773-12815`). DAO membership for photometry is **direct catalog_id hit in proc CSV**, not the Phase 0 spatial link.

---

### B.1 table - NOT POPULATED

All rows **BLOCKED** pending `draft_000450` on Milan's machine.

---

## 4. Root cause candidates (ranked)

| Rank | Candidate | Confidence | One experiment to settle |
|------|-----------|------------|---------------------------|
| **1** | Effective `vsx_out_of_scope_types` was `[]` at run time (config not saved, wrong data dir, or snapshot mismatch) despite Milan intending `["ROT"]` | HYPOTHESIS | Read `pipeline_meta.json` / provenance `config_snapshot` in draft_450; compare to disk `config.json` under resolved `VYVAR_DATA_DIR` |
| **2** | ROT filter **did** mask (`skip_photometry=True`) but Milan expected **removal** from actives / counted stub summary rows as "processed" | HYPOTHESIS | Count ROT rows with `skip_reason=vsx_type_out_of_scope` vs ROT rows with non-empty `lightcurves/*.csv` |
| **3** | VSX export unbounded by science mag + MASTERSTAR G<=18 floor -> many faint VT rows; Phase 0 spatial match inflates actives (~160->360) | HYPOTHESIS | Compare `len(variable_targets.csv)` and `len(active_targets.csv)` vs older draft; histogram `mag`; count actives with zero DAO frames in proc CSVs |
| **4** | Adaptive `match_radius_arcsec` (5x plate scale, min 10") attaches VSX to wrong masterstar | HYPOTHESIS | Distribution of implied match separations VT xy vs ms xy for actives; count separations > 1 px |
| **5** | Non-`VSX` `catalog` values on VT rows (manual/exo/legacy) bypass ROT filter | HYPOTHESIS | `variable_targets.csv` value counts on `catalog` column |
| **6** | Stale `active_targets.csv` from pre-ROT run (Phase 0 not re-run) | LOW HYPOTHESIS | Compare mtime `active_targets.csv` vs `variable_targets.csv` / infolog Phase 0 timestamp; normal pipeline always rewrites in `run_phase0_and_phase1:14616` |

---

## 5. Proposed fixes (description only - NOT implemented)

### P1. ROT / out-of-scope observability

- Log at Phase 0 close: `out_of_scope: cfg=[...] vt_auto_vsx=N masked=M`.
- WARN when `cfg` non-empty and `masked==0` but VT contains matching types.
- Extend `INV-CFG-01` with reverse direction (optional FAIL/WARN policy).
- **Regression test:** config `["ROT"]` + VT with `catalog="VSX", vsx_type="ROT"` -> exactly one active row with `skip_photometry=True`, zero LC files; plus test with `catalog="MANUAL"` unchanged.

### P2. Align Phase 0 with GAIA==DAO==VSX contract

- After spatial ms match, require `normalize(vrow.catalog_id) == normalize(ms_row.catalog_id)` when VT carries non-empty `catalog_id`, else exclude with reason `vsx_gaia_id_mismatch`.
- Optionally require `best_dist_arcsec < 3"` for auto-VSX (stricter than comp selection).
- Mask `zone_flag in ("catalog_only", "neznama_zona")` with `skip_photometry=True` (or exclude entirely per DECISIONS).
- **Regression test:** `_integration_test_todos.py:107` style + synthetic VT/ms mismatch must not become active.

### P3. Decouple VSX science depth from MASTERSTAR astrometry depth

- Derive VSX faint cutoff from **measured DAO depth** (Fleming / `compute_gaia_dao_reconcile` / frame proc CSV faintest detection), not `MASTERSTAR_FAINTEST_MAG_FLOOR=18`.
- Expose or reuse a single config key (e.g. extend `variability_mag_limit` or new `vsx_auto_mag_limit`) applied in `_query_vsx_local_frame_bbox` or VT write filter.
- **Regression test:** deep MASTERSTAR + shallow proc CSV -> VT/actives count bounded by proc depth.

### P4. Documentation

- Clarify in PARAMS help: `vsx_out_of_scope_types` is **mask-first** (targets remain listed).
- Clarify `variability_mag_limit` does not filter auto-VSX photometry targets.
- Restate DECISIONS DAO+Gaia-only with explicit identity rule vs current spatial match.

---

## 6. Open questions for Milan

1. **Where is `draft_000450`?** (full path to `Archive/Drafts/draft_000450` or tarball) - required to close A.1/B.1 with CONFIRMED counts.
2. **Exact DRY entry point:** RUN VYVAR full chain, Aperture Photometry button only, night_run CLI, or custom script?
3. For Symptom A: did ROT targets get **non-empty light curve CSVs**, or merely appear in `active_targets` / `photometry_summary` with `n_frames=0`?
4. Can you paste the five config keys from **`pipeline_meta.json` / provenance snapshot inside draft_450** plus the Friday infolog lines containing `select_active_targets:` and `out_of_scope` (if any)?
5. Is there a pre-regression BO CVn draft (~160 actives) still on disk for delta table?
6. Was Friday run on **interpreted** `src_py` or a **release bundle** with compiled modules? If bundle, which preview SHA?
7. For Symptom B: of the >360 actives, how many have **measurable** LCs vs all-NaN / `no_data`?

---

## 7. Docs impact (follow-up fix task - not done here)

Minimum set flagged by task: `VYVAR_DECISIONS.md`, `VYVAR_PARAMS.md`, config registry help for `vsx_out_of_scope_types` + VSX mag limit key, `VYVAR_INVARIANTS.md` (reverse CFG gate + `catalog_only` count), `flow_doc_facts.py` if fact counts change.

---

## 8. Gates

No `--fast` / `--full` runs (read-only task). No code changes. Report file uncommitted per task spec.
