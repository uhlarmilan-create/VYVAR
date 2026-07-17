CURSOR RESULT — 2026-07-10 (F-BINGAIN-1 FIX)

What I did
Implemented Milan-approved empirical background-noise term in the photometric error model
(empty-aperture `sigma_bkg_ap` + source Poisson), config keys, proc-CSV provenance columns,
SNR-table measured per-pixel bkg variance hook, citations, unit tests, and docs. Did **not**
commit or push — draft chi2 acceptance gate could not run (no draft folders in local Archive).

## Output / findings

### Part 0 — pedestal / detrending trace

| Item | Finding |
|------|---------|
| C5A bias/dark inventory (Stage B) | **0** bias/dark frames in Archive for draft_426 equipment |
| Dark subtraction | `pipeline.py` `_calibrate_one_light_apply_masters_in_ram` L14643–14646: `data -= md_data` only when `md_data is not None`; with no master dark ? **no bias/dark subtracted** |
| Pedestal in level | Remains in ADU level by construction ? level-based `sky_pp/g·A` background term is wrong (non-Poisson pedestal + resampling correlation) |
| Photon-transfer P fit | Stage C sandbox methodology: `var = (level - P)/g`; numeric P table requires draft_426 FITS (not present in `Archive/Drafts/` this session) |
| draft_426 photometry input | Pre-calibrated path ? `detrended_aligned/lights` (alignment/detrend; not full cal library dark) |

### Part 1 — implementation summary

- **Empirical mode:** `err = sqrt(F/g + sigma_bkg_ap²) / F`; no separate RN/annulus term (double-count guard documented).
- **Howell mode:** byte-identical legacy via `_howell_variance_adu2` / `_photometric_error`.
- **Fallback:** crowding ? Howell + `err_bkg_source=howell_fallback` + `log_event` once per frame.
- **New proc-CSV columns:** `sigma_bkg_ap`, `err_bkg_source`.
- **Config defaults:** `err_background_mode=empirical`, `err_empty_apertures_n=64`, `err_empty_apertures_min=16`.

### Part 2 — validation

| Check | Result |
|-------|--------|
| pytest full suite | **733 passed**, 15 skipped |
| New unit tests | `tests/test_err_background_empirical.py` (10 tests): white noise, correlated common-mode > Howell, crowding fallback, clamps, provenance columns, SNR table bkg var |
| draft_424/425/426 chi2 matrix | **NOT RUN** — no `Archive/Drafts/draft_*` on disk |
| Byte-identity non-err columns | Not verified on real draft (requires re-run) |
| PDF overflow | Not re-run (no draft photometry re-run) |
| session_baseline_check | Not run (no baseline re-anchor without Milan chi2 PASS) |

**STOP condition triggered:** draft_426 V0611 chi2 ? 0.8–1.2 not verified ? **leave uncommitted**.

### Stage C reference numbers (pre-fix baseline)

| Metric | Values |
|--------|--------|
| Aperture r_ap (empirical/model) | g/i/r/z = 0.54/0.25/0.22/0.08 |
| V0611 g chi2 scaled (Stage C sandbox) | 0.805 |
| Post/pre closure ratio | 0.58–0.72 |

## Errors (if any)

None in pytest. Validation blocked: local `Archive/Drafts/` directory empty.

## Files changed

**Production / config**
- `photometry_core.py` — empty-aperture measurement, err mode switch, proc-CSV columns, SNR table bkg var
- `pipeline.py` — wire config into catalog BPM worker st dict
- `config.py` — 3 new keys + load/save

**Tests / scripts**
- `tests/test_err_background_empirical.py` (new)
- `scripts/bingain_fix_validate.py` (stub — needs reprocessed proc CSVs)

**Docs / citations**
- `CITATIONS.bib` — Merline1995, Fruchter2002, Casertano2000, Labbe2003
- `citations.py` — CORE entries
- `docs/config_schema.md`, `docs/VYVAR_PARAMS.md`
- `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`

**Not committed** (pending chi2 gate on reprocessed drafts).

## Next steps (Milan / operator)

1. Re-run photometry on draft_424, draft_425 (B/V/R), draft_426 with empirical mode (default).
2. Run check-star chi2 matrix; confirm draft_426 V0611-class ? 0.8–1.2; wide-rig unchanged within noise.
3. Re-anchor byte-identity baseline for `err` column per PROCESS.
4. Commit + push after acceptance PASS.
