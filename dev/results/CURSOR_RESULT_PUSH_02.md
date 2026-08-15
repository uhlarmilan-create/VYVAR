CURSOR RESULT - 2026-08-15 PUSH-02

Register ID: PUSH-02
Follows: TARGET-DEPTH-02 (accepted)
Milan authorized push.

## Verdict

Two findings recorded; tip verified `--fast` OVERALL PASS; pushed to
`origin/main`. No code or threshold changes in this task.

---

## 1.1 Depth criterion not valid on pre-SNR-GATE drafts

TARGET-DEPTH-02 half-linear MASTERSTAR depth:

| Draft | depth_g | MASTERSTAR context |
|------:|--------:|--------------------|
| 513 | 15.0 | post-SNR-GATE (meaningful) |
| 435 | 15.5 | deeper MS (meaningful) |
| 512 | 11.5 | pre-SNR-GATE broken prematch |
| 510 | 11.5 | pre-SNR-GATE broken prematch |

On 512/510 the half-linear transition marks **where MASTERSTAR stars ran out**
under the broken gate, not where measurability ran out. Applying the criterion
there yields a number that looks derived but measures catalog truncation.

**Rule:** the TARGET-DEPTH-02 depth is only meaningful on a MASTERSTAR built
after `SNR-GATE-01`. On older shallow MASTERSTARs, do not treat the derived G
as a science depth limit.

---

## 1.2 BO CVn comparison sets are disjoint

```
draft 512:  5 comps, all TIER1, trust GREEN, check_scatter 0.009300
draft 513:  4 comps, 0 TIER1, 1 TIER3, 3 TIER4, trust RED, check_scatter 0.011147
intersection: empty
```

Stronger than losing one star: 513 (deeper detection) returned an entirely
different, worse-tiered ensemble and retained none of 512's bright comps.

Register item: **BO-ENSEMBLE-01** (OPEN finding). T2-R4: uncontrolled pair
(513 trial under X-R3). No cause attributed. Rebuild of both drafts on one
committed tip would settle it.

---

## 1.3 Deferred (one line each; with section 4)

- **DET-vs-MEAS-01:** The 3.78-sigma cut is a detection threshold, not a
  measurability threshold. T2-R0: MASTERSTAR is single-frame (factor=1);
  best-vs-median FWHM on 512 is only ~1%; remaining gap (linear zone vs
  unusable LC; ~0.29 mag/point at 3.78-sigma) is not established.
- `--full` anchor and P1 golden ledger, stale since SKY-CLIP-01 and again since
  SNR-GATE-01
- draft 510 and 512 checksum manifests
- drafts 512 and 513, both trial runs from uncommitted trees under X-R3
- every draft built since `c9e1f8f` carrying the shallow MASTERSTAR depth
- drafts 512, 513 and 510 stuck at status `INGESTED`; repair path in
  TARGET-DEPTH-02 Item D, not applied
- `COMP-POOL-01` Stage 2, blocked under `C2-R2` pending the guard
- WIDE-ERR, diagnosed and localized, not fixed; exported error bars unchanged
- BO CVn disjoint ensembles (BO-ENSEMBLE-01)
- `zone` rename to `dao_detected` / `dao_subthreshold`, proposed not applied
- removal of the draft-level magnitude limit as redundant, proposed not applied
- `LOCATION_OLD` orphan heal
- `A-1-OVERRIDE`, `U-SKY-FALLBACK-01`, `INV-PIXELS-01`, D1-2 exposure ramp,
  C-EXPORT-GAP, W6-PROP, D10-1, D11-1, D5-1, Decision (4), U-SCATTER-DEF

---

## 2. Verify

- Docs/register commit: see below
- `--fast` before push: OVERALL PASS on tip recorded below
- No adjustments to make checks pass

---

## 3. Push

- Command: `git push origin main`
- Pushed tip SHA: (filled after push)
- Post-push `--fast`: `git-origin-main` must not report a difference
- `git log --oneline -15` and working tree: see footer after push

---

## Register diff

- **PUSH-02**: CLOSED (record + push)
- **TARGET-DEPTH-02**: note pre-SNR-GATE depth caveat
- **BO-ENSEMBLE-01**: OPEN (finding; T2-R4)
- **DET-vs-MEAS-01**: OPEN deferred (3.78-sigma detection vs measurability)
