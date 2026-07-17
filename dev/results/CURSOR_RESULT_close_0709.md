CURSOR RESULT -- 2026-07-09 SESSION-CLOSE-0709

What I did
Verified draft_426 FITS headers against equipment DB rows before any change; concluded OBS_DRAFT eq4 is correct (no `--apply`). Added `scripts/fix_draft_equipment.py` + tests. Recorded sigma Phase A wide-rig DONE, F-BINGAIN-1 reframe, sparse-comp gate proposal in ROADMAP. STATE/JOURNAL session close. pytest + session_baseline_check PASS. Pushed to origin/main.

## Part 1 -- draft_426 equipment attribution

### Header verification (4 setups sampled: g_60_4, i_70_4, r_60_4, z_90_4)

| Keyword | Value |
|---------|-------|
| INSTRUME | C5A-150M |
| NAXIS1 x NAXIS2 | 3552 x 2664 (bin4) |
| XBINNING / YBINNING | 4 / 4 |
| GAIN | 12.48 e-/ADU |
| IMAGETYP | OBJECT |
| NCOMBINE | absent |
| EXPTIME | 60 / 70 / 90 s (per setup) |

### Cross-check vs equipment rows

| Evidence | eq2 C3-26000/IMX571 | eq4 C5A-150M/IMX411 |
|----------|---------------------|---------------------|
| INSTRUME | C3-26000 | **C5A-150M (match)** |
| Pre-bin geometry | 6252x4176 -> 1563x1044 @ bin4 (no) | **14208x10656 -> 3552x2664 (match)** |
| GAIN=12.48 | 0.78x16=12.48 (match) | 1.0x16=16.0 (no) |

**Verdict:** Headers **unambiguously identify eq4**. GAIN=12.48 is anomalous (matches eq2 bin4 scale) but does **not** override INSTRUME + geometry. **STOP on re-attribution to eq2** -- Milan decision already satisfied by header identity.

### DB before/after

| Field | Before | After |
|-------|--------|-------|
| OBS_DRAFT.ID | 426 | 426 |
| ID_EQUIPMENTS | **4** | **4** (unchanged) |

Dry-run (`scripts/fix_draft_equipment.py --draft 426`): eq4 score=600, eq2 score=40 -> `No change needed (already correct).`

### Cached/per-draft artifacts (list only -- regeneration NOT in scope)

No archive artifact stores a **wrong** `equipment_id` for draft_426 (DB was already eq4). Related items that carry **header-derived** gain/RN (not equipment-row mismatch):

- `Archive/Drafts/draft_000426/platesolve/*/photometry/pipeline_meta.json` -- `dynamic_params.gain=12.48`, `read_noise=14.08` (from FITS headers)
- `tmp/sigma_budget/bin4_sigma_forensics.json` -- forensics on header gain vs sigma_exp
- `tmp/sigma_budget/sparse_comp_diag.json` -- draft_426 SS Cam diagnostics (rig D from telescope, not equipment_id)

## Part 2 -- ROADMAP closeout (recorded)

- **Sigma budget Phase A -> DONE (wide rig):** photon + Honeycutt ensemble SEM + 6.5 mmag floor; scint ~2 mmag negligible D=0.2 m; f_resid->0. Attribution: k2 ZERO; phase strongest (6.5->4.5 mmag); ~4.5 mmag rig constant. Open: **PROD-SIGMA-FLOOR**, **SIGMA-NEWTON**.
- **F-BINGAIN-1:** gain accounting excluded as chi2-deficit cause; RN scaling, sky/area, stack hypothesis recorded (NCOMBINE absent).
- **Sparse-comp / SS Cam:** field-wide comp_rms = offset structure (~95% cancels); temporal 8-12 mmag; check scatter healthy with CI; proposed gate redesign (not enacted); SS Cam YELLOW.

## Part 3 -- session close

- STATE + JOURNAL updated (DAO-RECONCILE -> sigma A->A4 -> sparse-comp -> equipment verify).
- **pytest:** 681 passed, 15 skipped (+2 new equipment tests).
- **session_baseline_check (default/fast):** OVERALL PASS.
- **origin/main:** `f9acb2d`

## Errors (if any)

None.

## Files changed

- `scripts/fix_draft_equipment.py` (new)
- `tests/test_fix_draft_equipment.py` (new)
- `docs/VYVAR_ROADMAP.md`
- `docs/VYVAR_STATE.md`
- `docs/VYVAR_JOURNAL.md`
- `CURSOR_RESULT_close_0709.md` (this file)

Commits: `25d6936` (session close), `f9acb2d` (pytest count fix)
