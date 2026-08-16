# CURSOR TASK - WIDE-ERR-02 (calibrated exported errors + weighted SEM)

Date issued: 2026-08-16
Baseline: draft 515 (run SHA da9cce4); local tip per current session state.
Ships together, per standing decision: the SEM fix (measured ratio
sem_weighted/sem_current = 0.677) is ONLY allowed to land inside this task.
Push: NOT authorized.

## Ground established before this task (architect, sandbox, draft 515)

Empirical per-star scatter vs global field ZP, 2589 clean stars x 134 frames
(reference CSV supplied by Milan: wide_err_515_empirical.csv; quantity:
1.4826*MAD of ZP-corrected inst-mag series, mmag):

- measured / (photon + sky) deficit: 4.0x at G 8-8.5, 2.0x at G 9.5-10,
  1.0x (0.97-1.04) at G 11.5-14.5, 1.17-1.27x at G 14.5-15.5.
- Photon and sky terms are CORRECT in the mid range; the bright-end deficit
  is a floor, not a wrong shape.
- The 14.8 mmag fitted floor is GLOBAL-frame (contains field systematics
  that local ensembles cancel); the LC-frame floor implied by check sidecars
  is ~6-8 mmag. Scintillation expectation for D=70 mm, t=60 s: 2.2-3.1 mmag.
- Fitted EFFECTIVE gain from the photon term: 0.24-0.32 e-/ADU (covaries
  with k_sky in [1.0, 1.13]).

Literature basis (add citations to docs): error rescaling to chi2_red = 1
plus a red-noise floor added in quadrature is standard published practice
(Pont, Zucker & Queloz 2006; Gillon et al. 2009; Winn et al. 2008 beta
factor); field systematics dominating bright stars in ensemble photometry is
the documented motivation for SysRem/TFA (Tamuz, Mazeh & Zucker 2005;
Kovacs, Bakos & Noyes 2005). SysRem itself is explicitly OUT OF SCOPE for
v1.0 (modifies light curves); this task fixes error bars only.

## Part W1 - Production error dump (measurement, no change)

W1a. For the same clean-star concept (linear, non-variable, non-saturated,
     edge-safe, G 8-15.5) evaluate the PRODUCTION error assembly
     (err_total^2 = photon^2 + sem^2 + scint^2 + sys^2) per star on draft
     515, and dump per-star medians of each component in mmag plus err_total,
     to dev/results/WIDE_ERR_02_prod_components.json, keyed by catalog_id
     with G.
W1b. Report per G bin (0.5 mag bins): median err_total and median of each
     component. One line: what do sys and scint contribute at G 8-9 today?
     Pre-registered expectation from the architect's measurement: if
     sys+scint gives ~3-4 mmag where the LC-frame truth is ~6-8, the ~2x
     bright deficit is reproduced; if production sys already gives 6-8, the
     deficit lives elsewhere and Part W3's design must be revisited before
     implementation - stop and report in that case.
W1c. Gain check: read the gain (e-/ADU) the production noise model actually
     uses for NoFilter_60_2 (equipment DB / header resolver - cite the
     source row). Compare with the architect's fitted effective 0.24-0.32.
     Agreement within ~30 percent: photon term confirmed end-to-end.
     Disagreement beyond 2x: STOP - the photon term itself is suspect and
     the calibration design below would mask it; report before implementing.

## Part W2 - The calibration meter (mag_calib frame, comps only)

Build the per-draft empirical truth in the SAME frame as the published
product (this is the decisive lesson of XVAL-BO-01 - meters must match the
product):

W2a. For every clean comp star of every LC target on 515, compute the
     mag_calib-frame series scatter exactly as the check sidecar does
     (star differenced against that target's ensemble, pytics weights),
     excluding the star from its own ensemble where applicable. Reuse the
     01B machinery. Quantity label: lc_frame_scatter_mad_mmag.
W2b. Per G bin: median lc_frame_scatter vs median production err_total for
     the same stars -> the LC-frame deficit table. This table, not the
     global-ZP one, is what the calibration must flatten.
     NEVER include science targets in this set (variables would inflate the
     calibration and suppress their own exported errors) - add an explicit
     guard with a fire proof (inject one known variable, show it is
     rejected).

## Part W3 - Implement the calibrated error model + weighted SEM (one wave)

W3a. Exported per-epoch error becomes:
       err_exported^2 = (s * err_model)^2 + sigma_r^2
     with s (white-scale) and sigma_r (floor, rel-flux domain) calibrated
     per draft x rig x G bin (or a smooth function of G if bins are noisy)
     on the Part W2 comp set, by requiring median chi2 = 1 per bin.
     Literature form: Pont+2006 / Gillon 2009. Both parameters persisted in
     the draft (manifest or sidecar) with their calibration n per bin -
     no silent constants (INV-NO-SILENT family; params registry entry with
     units).
W3b. SEM fix ships in the same wave: replace sem_current with the weighted
     SEM matching the weighted mag_calib combine (the written 0.677 fix).
     Separable commit, but same task and same acceptance run - per the
     standing decision it must never ship alone.
W3c. Config: export_err_mode = calibrated (default) | model (legacy,
     byte-identical fallback). UI parity per house rule. AAVSO/VarAstro
     export headers gain one comment line naming the mode and the
     calibration (s, sigma_r ranges) so a referee sees it.
W3d. Fire proofs (a verification that cannot fail is not a verification):
     - chi2 gate fires on the OLD model: show median chi2 != 1 at G 8-9
       before calibration, = 1.0 +/- 0.1 after, per bin, on 515.
     - the variable-star guard fires (W2b injection).
     - legacy mode byte-identity on a fixed input.
W3e. Pre-registered acceptance for the whole task: on draft 515, per G bin,
     median(lc_frame_scatter / err_exported) in [0.9, 1.1] INCLUDING the
     G 8-9 bin; and exported err at G 8-9 does not drop below the
     scintillation floor 2.2 mmag. If any bin cannot reach the window,
     report which and why - do not force it by widening the window.

## Part W4 - Docs

- DECISIONS: WIDE-ERR-02 entry - the two-parameter calibrated form, the
  frame lesson (calibrate in mag_calib frame, comps only), SysRem explicitly
  deferred (modifies data; out of v1 scope).
- Citations added: Tamuz+2005, Pont+2006, Kovacs+2005, Gillon 2009,
  Winn 2008.
- Register: WIDE-ERR -> CLOSED (with SHAs) if W3e passes; SEM item closed
  with it; WIDE-ERR-CROSSRIG stays OPEN (Newton/Boyden get their own
  calibration when their drafts exist - the mechanism is per-rig by design).

## Report

dev/results/CURSOR_RESULT_WIDE_ERR_02.md + JSON with the W1b/W2b/W3e tables.
Every number: quantity, units, frame (global-ZP vs mag_calib), SHA.
session_baseline_check.py --fast OVERALL PASS required. Defects in this
spec: name them; W1b and W1c contain explicit STOP conditions - honour them.

ASCII only. English.
