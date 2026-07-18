CURSOR RESULT - 2026-07-18

Task: RECON-DAO435-K2DATA - two report-only diagnostics before the K2 arc
Baseline: origin/main 191be0e (Anchor #3 ACTIVE draft_000435; pytest 928/19)
Scope: REPORT-ONLY. No code changes, no config changes. Run outputs stay in tmp/.
Single commit (not pushed); Milan authorizes push later.

What I did
- PART A: located the canonical DAO-RECONCILE runner, ran it on the draft_435
  sky-surface anchor config (draft_000435 and draft_000436), compared miss@G90 /
  completeness_50 / G_lim against the closed-arc baselines, and rendered the
  reopen-trigger verdict.
- PART B: scanned the drafts DB (vyvar.sqlite3) + archive metadata and built a
  candidate-night inventory of every filtered draft (airmass span, calibration
  status, flux-derived comp residual floor, NIGHT_FIT gate verdict); confirmed the
  K2 v2 fit path is config-stubs only and named the exact code insertion points;
  cross-checked the spec's "BVR night with dX >= 0.3" gate against the holdings.

================================================================================
PART A - DAO-RECONCILE health on the draft_435 config (reopen-trigger check)
================================================================================

## A.1 Canonical runner (named)

Runner: dev/scripts/dao_reconcile_diag.py
  (kernel: src_py/dao_reconcile.py -> compute_gaia_dao_reconcile,
   fleming_completeness, resolve_effective_match_depth)

This is the exact tool that produced tmp/dao_reconcile/cross_draft_summary.json
(header banner "Gaia<->DAO reconciliation (R-2b footprint)"; --all-drafts writes
the cross_draft_summary.json; --draft N runs a single draft). Invoked here as:

  python dev/scripts/dao_reconcile_diag.py --draft 435
  python dev/scripts/dao_reconcile_diag.py --draft 436

Both drafts are the home wide rig, setup NoFilter_60_2 (Carl-Zeiss 200 mm +
QHY294MM, Jirny), i.e. the same rig/field class as the closed-arc wide baseline
draft_424 - a genuine apples-to-apples comparison for the sky-surface change.

## A.2 Numbers (draft_000435 and draft_000436)

draft_000435 and draft_000436 are byte-identical in the reconcile output (same
field, same anchor config; 436 is the anchor-config e2e re-run). Both report:

| Metric                     | draft_000435 | draft_000436 |
|----------------------------|--------------|--------------|
| completeness_50            | 88.99%       | 88.99%       |
| G_lim_50                   | 14.52        | 14.52        |
| G_lim_50 censoring         | NOT censored (Fleming erf crossing found; g_lim_50_censored=False) | same |
| G_lim_90                   | 13.69 (not censored) | same |
| fit_method                 | fleming1995_erf | same     |
| missed_total (genuinely)   | 285          | 285          |
| missed_below_g90           | 13           | 13           |
| missed_fadezone            | 272          | 272          |
| match_depth                | 18.0 (MASTERSTAR _ms_faintest_mag_eff; faintest_mag_limit unset -> default) | same |
| n_ref_in_frame / off-frame | 12454 / 884  | same         |
| matched / below-limit / blended | 2802 / 9292 / 75 | same    |

Run artifacts (tmp/, report-only):
  tmp/dao_reconcile/draft_000435/NoFilter_60_2/report.json
  tmp/dao_reconcile/draft_000436/NoFilter_60_2/report.json

## A.3 Comparison vs closed-arc baselines

Reopen trigger (DAO-RECONCILE, CLOSED 2026-07-09): "missed@G90 material (hundreds)
on a new rig/config". The draft_435 anchor introduced the order-2 sky-surface
background subtraction AFTER that close - this check verifies the trigger does not
fire on a background change that could in principle move faint-end detection.

Wide-rig, same field (NoFilter_60_2):

| Draft      | config              | compl_50 | miss@G90 | fadezone |
|------------|---------------------|----------|----------|----------|
| draft_424  | pre-sky-surface     | 89.67%   | 15       | 338      |
| draft_435  | sky-surface (order2)| 88.99%   | 13       | 272      |
| draft_436  | sky-surface (order2)| 88.99%   | 13       | 272      |

Cross-rig closed-arc completeness_50 range (2026-07-09): 89.7% - 98.3%.
draft_435 completeness_50 = 88.99% sits 0.68 pp below the 424 wide reference and
just under the low end of the cross-rig band - a fraction-of-a-percent move, not a
regime change.

## A.4 VERDICT: HEALTHY

- miss@G90 = 13 on the sky-surface config vs 15 on the pre-sky-surface wide
  baseline (draft_424). This is COMPARABLE, in fact marginally lower - the order-2
  background subtraction did NOT increase genuine faint-end misses. 13 is nowhere
  near the "hundreds" material threshold that would reopen the arc.
- fadezone dropped 338 -> 272 and completeness_50 moved 89.67% -> 88.99% (both
  within run-to-run/field noise; G_lim_50 = 14.52 uncensored, Fleming crossing
  found, consistent with the wide rig's ~15.0 characterization).
- One-line note for the ledger: the sky-surface (order-2 background) anchor config
  was checked against DAO-RECONCILE on 2026-07-18; miss@G90 = 13 (draft_435/436) vs
  15 (draft_424 baseline). DAO-RECONCILE STAYS CLOSED; reopen trigger did not fire.

================================================================================
PART B - K2 NIGHT_FIT v2 readiness (data inventory + implementation gap)
================================================================================

## B.1 CANDIDATE-NIGHT INVENTORY (key deliverable)

Sources and exact metric definitions used:
- Draft / rig / band / n_frames per band: vyvar.sqlite3 tables OBS_DRAFT,
  OBS_FILES (IMAGETYP='light'), EQUIPMENTS, TELESCOPE, LOCATION.
- Airmass span dX = X_max - X_min: computed READ-ONLY from stored per-frame
  metadata OBS_FILES.RA / OBS_FILES.DE / OBS_FILES.INSPECTION_JD + the draft's
  LOCATION (lat/lon/alt), via astropy AltAz and the Kasten & Young (1989) airmass
  (matches the pipeline's airmass law). Where per-frame RA/DE are NULL (the Newton
  BVR chi&h-Per drafts) and the OBS_DRAFT field centre is null-island (0,0), dX from
  the DB is UNRELIABLE - the authoritative dX=0.014 for that night comes from the
  K2 design campaign (VYVAR_K2_DESIGN_SPEC.md Section 1) and is used instead.
- Calibration status: OBS_DRAFT.CALIBRATION_MODE / IS_CALIBRATED.
- Comp residual floor: sigma_resid_honeycutt_robust = the FLUX-DERIVED (dao_flux,
  never proc catalog `mag`) robust Honeycutt per-comp residual scatter on the
  QC-clean (gated) frame set - this is exactly the NIGHT_FIT input metric named in
  VYVAR_K2_DESIGN_SPEC.md Section 6 item 1. Values from the K2 v2 sandbox
  tmp/k2_fit427_v2/k2_fit_v2.json (drafts 427 and 426). Live proc outputs for the
  filtered drafts no longer exist on disk (Archive currently holds only 435 /
  435_snapshot / 436), so these sandbox numbers are the authoritative residual-floor
  record; no filtered draft can be re-measured from disk right now.

GATE (K2-DATA-BLOCKER): QUALIFIES needs ALL of {filtered; calibrated; dX >= ~0.3;
comp residual floor << 15 mmag (+ per spec Section 6: non-monotonic X(t) and k''
detectable, sigma_k2_pred <= |k2_lit|/3)}.

Sorted by dX descending:

| Night (drafts)                | rig (DB)                         | band(s) | date       | n/band | dX     | calib          | resid floor (flux Honeycutt robust) | GATE |
|-------------------------------|----------------------------------|---------|------------|--------|--------|----------------|-------------------------------------|------|
| Boyden g/r (406/408/412/413/414/416/427) | Celestron 14" + C3-26000, Boyden JAR | g, r    | 2026-06-05 | 162/160| 0.538 (g), 0.531 (r) | pre_calibrated | g=88.5 mmag, r=71.0 mmag (draft 427 gated) | NO   |
| Zdanice bin4 g/r/i/z (407/409/426) | C5A-150M + AZ800*, Zdanice        | g,r,i,z | 2026-06-08 | 25 ea  | 0.231 (g) 0.230 (r/i/z) | pre_calibrated | g=15.4 mmag, r=13.8 mmag (draft 426) | NO   |
| Newton BVR chi&h Per (417/418/419/420/421/425) | Newton 300/1200 + C3-26000, Dablice | B, V, R | 2025-08-10 | 12 ea  | 0.014 (spec) | pre_calibrated | not computed (proc gone; 12-min arc) | NO   |
| V-only (423)                  | Newton 300/1200 + C3-26000, Dablice | V       | 2025-03-20 | 125    | 0.065  | pre_calibrated | not computed (single band)          | NO   |
| V-only (422)                  | Newton 300/1200 + C3-26000, Dablice | V       | 2025-04-22 | 78     | 0.056  | pre_calibrated | not computed (single band)          | NO   |

*DB rig-label note: draft 426 is tagged ID_TELESCOPE=Celestron in the DB while its
same-night twins 407/409 are tagged AZ800; STATE/K2-COHORT treat 426's bin4 g/r/i/z
as the "Newton" cohort cells. The dX and residual-floor facts are rig-label
independent. All three are the same 2026-06-08 Zdanice bin4 g/r/i/z night.

Per-band verdict detail:
- Boyden g/r: EXCELLENT dX (~0.53, well above 0.3) but DISQUALIFIED on the residual
  floor: flux-derived Honeycutt robust residual 71-89 mmag on the QC-clean frames -
  5-6x the 15 mmag ceiling. This is the documented Boyden flat/flip systematics class
  (V454 CrA meridian-flip diagnostics; VYVAR_K2_DESIGN_SPEC.md Section 1.1/1.3). It
  also fails NIGHT_FIT consistency: ungated k''=+56 mmag vs gated -1 mmag (bad frames
  fake k''). Verdict: NO.
- Zdanice bin4 g/r/i/z: dX 0.23 is BELOW the 0.3 gate; X(t) is monotonic (time
  aliasing unbreakable per spec Section 1); residual floor ~13.8-15.4 mmag sits AT,
  not well below, 15 mmag; and sigma_k2_predicted ~44-51 mmag (k2_fit_v2.json) far
  exceeds |k2_lit| (~16 mmag for g) so k'' is undetectable (fails spec Section 6
  item 2). Verdict: NO / MARGINAL-at-best on dX.
- Newton BVR chi&h Per: dX=0.014 - two orders of magnitude below the gate; only a
  ~12-minute arc near transit; k''*C*dX degenerate with the colour term (spec
  Section 1). Verdict: NO.
- V-only Newton (422, 423): dX < 0.07 and single-band (no colour*airmass leverage;
  V literature k'' ~ 0 anyway). Verdict: NO.

Home wide rig (Carl-Zeiss 200 mm + QHY294MM, the vyvar_calibrated rig with a VYVAR
CalibrationLibrary): EVERY home-rig draft (410/411/415/424/428-436) is NoFilter -
there is NO filtered, VYVAR-calibrated draft anywhere. The CalibrationLibrary holds
only home-rig darks + a single NoFilter flat (no filtered flats for any rig).

### B.1 conclusion: NOTHING QUALIFIES.

No held draft passes the NIGHT_FIT gate. The trade-off is stark and consistent:
the only nights with dX >= 0.3 are the Boyden g/r nights, and they carry a 71-89
mmag residual floor (flat/flip systematics); every night with a tolerable residual
floor has dX <= 0.23 (Zdanice) or dX ~ 0.01-0.07 (Newton). This means the K2 arc
starts with an OBSERVING task, not a coding task: a filtered night (BVR or g/r) on a
photometrically calibrated rig with dX >= ~0.3 AND comp residual floor << 15 mmag
does not yet exist in the holdings.

IMPORTANT caveat on "Milan says he has data": this inventory reflects the drafts
recorded in the current vyvar.sqlite3 (mostly historical drafts already deleted from
Archive). If Milan has NEW filtered frames not yet imported, they are NOT visible
here - they must be ingested into a draft_NNNNNN first, then this exact inventory
(dao/airmass + a k2_fit_v2-style residual measurement) re-run to score them against
the gate. Milan should point at the candidate night's raw/calibrated frames so it can
be imported and measured.

## B.2 IMPLEMENTATION GAP (v2 is config-stubs only)

Confirmed: K2 v2 (NIGHT_FIT) is NOT implemented - only config stubs exist.
- src_py/config.py:551-554 defines defaults k2_fit_enabled=False,
  k2_fit_min_detectability=3.0, k2_fit_consistency_sigma=2.0, k2_fit_lit_factor=4.0;
  loader :1562-1582; to_dict :2410-2413. These keys have NO reader anywhere outside
  config.py (grep of src_py confirms zero production consumers).
- src_py/k2_extinction.py: K2Source.NIGHT_FIT enum value exists (:85) but is NEVER
  produced. resolve_k2_bprp_value (:146-180) returns only LITERATURE_DEFAULT or NONE;
  resolve_k2_mode (:137-143) will echo "fit"/"fit_else_literature"/"night_fit"/"auto"
  but resolve_k2_bprp_value does not branch on them - any non-"off" mode falls through
  to the literature lookup. apply_k2_per_frame (:205-239) only acts when
  source is LITERATURE_DEFAULT, so a NIGHT_FIT source would be an inert no-op. There
  is no fit routine and no pre-gate evaluation in the codebase.

NIGHT_FIT acceptance criteria (from VYVAR_K2_DESIGN_SPEC.md Section 6 - accept only
if ALL hold):
1. Inputs: flux-derived Honeycutt residuals (never proc catalog `mag`); fit frames =
   QC-clean subset computed READ-ONLY from always-on QC (align_residual_px in
   alignment_report.csv + B.2 quality metrics) - the photometry frame set is NOT
   changed by the fit.
2. Detectability (leverage): sigma_k2_pred <= |k2_literature| / k2_fit_min_detectability
   (default 3.0) for the band, using sd(C*dX) and N from the actual night.
3. Consistency: colour-tertile and brightness-tertile k'' agree within
   k2_fit_consistency_sigma (default 2.0) * sigma_boot; if X(t) is non-monotonic,
   per-arc k'' agree within the same band; if X(t) is monotonic, the fit is REFUSED
   (time aliasing unbreakable).
4. Plausibility: |k2_fit| <= k2_ceiling (0.1) AND sign/magnitude within
   k2_fit_lit_factor (default 4.0) of the literature default.
   Any failure -> fall back to LITERATURE_DEFAULT with the reason logged. Draft 427
   is the permanent REFUSE regression fixture (fails items 3 and 4).
Band handling: fit runs only for STANDARD_FILTER bands (band_classify); CLEAR / CV /
CR / L / UNKNOWN and OSC RGB tokens -> k2 none (no fit, no literature).

Code insertion points where the fit + its gate would live (same three sites v1
already threads the literature value through, so the fit result reuses the existing
apply path by producing a k2_value with K2Source.NIGHT_FIT):
- SOURCE RESOLUTION / band routing: photometry_core.py:8291-8358
  (resolve_k2_bprp_value -> state.k2_source at 6738). The pre-gate + fit would compute
  the per-night k2_value here (per obs_group), then set source=NIGHT_FIT on pass or
  fall back to LITERATURE_DEFAULT on any gate failure.
- GROUP CT FIT: photometry_core.py:3962-3972 (apply_k2_to_comp_mag_inst applied to
  the comp instrumental-mag arrays BEFORE fit_color_term_c1) - the fitted k'' must be
  used here too or c1 double-counts k''*X_bar.
- PER-TARGET PHASE 2A: photometry_core.py:9061-9083 (apply_k2_per_frame before
  apply_color_term at :9103).
- METHOD/REPORT LC builder: method_lc_output.py:209-264 (the second production path;
  identical correction via the shared helper).
- PROVENANCE COLUMNS: photometry_core.py:4711-4831 already writes k2_source / k2_value;
  the fit path only needs to emit source="night_fit" and the fitted value (plus,
  optionally, the pre-gate outcome for the PDF methods line).
The pre-gate machinery already exists in sandbox form (tmp/k2_fit427_v2/,
tmp/k2_sigma_fix/, tmp/k2_rerun427/) - productionizing it is the v2 coding task, gated
behind k2_fit_enabled and the acceptance criteria above.

## B.3 Cross-check: spec "BVR night with dX >= 0.3" vs the inventory

Tension flagged. The spec/cohort fit-quality gate is stated as "BVR night with
dX >= 0.3" (K2_BAND_AWARE_SPEC.md line 33; VYVAR_K2_DESIGN_SPEC.md K2-DATA-BLOCKER).
The inventory shows:
- The ONLY BVR night VYVAR holds (chi&h Per, drafts 417-421/425) has dX = 0.014 -
  ~20x below the 0.3 requirement. No BVR night in the holdings comes anywhere close.
- The only filtered nights that DO reach dX >= 0.3 are the Boyden g/r (Sloan, not
  Johnson BVR) nights, and they are disqualified by a 71-89 mmag residual floor.
So the spec's canonical example ("a BVR night at dX >= 0.3") is satisfiable in
principle but is NOT met by any dataset on hand: the high-dX filtered nights are
Sloan g/r with flat/flip systematics, and the one BVR night is a near-transit
12-minute arc. Practical resolution: the gate should read "a filtered night (BVR OR
Sloan g/r) with dX >= ~0.3 AND comp residual floor << 15 mmag on a photometrically
(VYVAR-)calibrated rig" - and that dataset requires a NEW observation. The
K2-DATA-BLOCKER is therefore still unsatisfied; the "home rig + flats qualifies"
expectation (ROADMAP K2 ledger) has not yet been realized as an actual filtered,
wide-dX, VYVAR-calibrated draft.

================================================================================
## Output / findings (summary)
================================================================================
- PART A: DAO-RECONCILE HEALTHY on the sky-surface anchor. draft_435/436 miss@G90=13
  vs draft_424 baseline 15; completeness_50 88.99% vs 89.67%; G_lim_50 14.52
  uncensored. Reopen trigger does NOT fire; arc stays CLOSED.
- PART B: NOTHING QUALIFIES for NIGHT_FIT. Best dX (Boyden g/r ~0.53) is killed by a
  71-89 mmag residual floor; best residual floor (Zdanice ~14 mmag) is killed by dX
  0.23 + monotonic X(t) + undetectable k''. v2 is config-stubs only; insertion points
  named. K2 arc starts with an observing task (a filtered, wide-dX, cleanly-calibrated
  night), not a coding task.

## Runner names
- PART A: dev/scripts/dao_reconcile_diag.py (kernel src_py/dao_reconcile.py)
- PART B inventory (this task, ad hoc read-only DB/airmass queries): scratch helpers
  under tmp/ (_recon_*.py); residual-floor authority tmp/k2_fit427_v2/k2_fit_v2.json
  from the K2 v2 sandbox.

## Errors (if any)
- None. Both DAO-RECONCILE runs exited 0 (one benign astropy FITSFixedWarning about
  the deprecated RADECSYS keyword). Newton BVR / V-only drafts have NULL per-frame
  RA/DE and null-island field centres in the DB, so their DB airmass is unreliable;
  the spec's authoritative dX=0.014 is used for the BVR night, and V-only dX comes
  from valid field centres.

## Files changed
- dev/results/CURSOR_RESULT_recon_dao435_k2data.md (this file; new).
- No src_py/, config.json, or docs/ changes. tmp/ scratch is gitignored.
