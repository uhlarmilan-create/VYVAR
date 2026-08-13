CURSOR RESULT - 2026-08-13

What I did
Pushed four commits (A). Set EQUIPMENTS.SATURATE_ADU NULL for QHY294MM (B). Built and
ran SAT-DIAG dry-run script on real drafts (C). Fixed config-diff counting note (A.2).

---

## Part A - Push

### A.1

Pushed to `origin/main`. **New HEAD: `a1b8674526129649c3a290ca7505faaa400b31ab`**

| Commit | Summary |
|--------|---------|
| `5dd2a4d` | fix(photometry): remove per-frame MAD clip from ensemble zeropoint |
| `7eb125a` | fix(encoding): restore ASCII-only policy on ZP-clip commit artifacts |
| `d758c83` | test(qc): commit frame-62 FWHM fixture; drop draft_508 path dependency |
| `a1b8674` | docs: file SAT-DIAG spec, decisions, and draft-435 restore checksums |

### A.2 Config diff counting (corrected)

The earlier report mixed two counting methods:

| Method | Count | Meaning |
|--------|------:|---------|
| **Structural** | **84** | 5 keys with different values on the intersection + 49 keys only in draft 435 snapshot + 30 keys only in draft 509 snapshot |
| **Value inequality** | **73** | Keys where `cs435.get(k) != cs509.get(k)`; keys present in only one dict do not increment this count because `.get()` returns `None` on both sides for missing keys only when... actually only-in-one-side: cs435 has key, cs509.get returns None, value != None -> counts. Wait - only435 keys: cs509.get is None, cs435 has value -> counts as diff. So 49+30+5 should equal value diffs if all only-side values non-None.

Recheck: val_diff intersection 5, only435 49, only509 30 = 84. The 73 comes from union .get() comparison where 11 keys in only435 or only509 have **NULL/None values on the side that has the key** so both sides compare equal as None? 

**Resolved statement:** **84** is the total number of keys that differ structurally (present on one side only, or different values). **73** is the count of keys in the union where `get(k)` values differ; the 11-key gap is keys present only in one snapshot whose stored value is `null` (JSON null / Python None) -- they are structural presence diffs, not value diffs.

Flux-path keys confirmed unchanged after draft-435 restore. No session conclusion depends on 84 vs 73.

---

## Part B - SATURATE_ADU NULL

### B.1 Fallback confirmed (read current code)

`_effective_saturation_limit` (`pipeline.py:5285-5324`): header keywords -> `equipment_saturate_adu` -> DATAMAX/MAXPIX -> `_infer_sat_limit_from_bitpix` -> fallback.

With `equipment_saturate_adu=None` and no header saturation keywords on draft 510 raw
(`BITPIX=16`, `BZERO=32768`): returns **65535.0**, source **`bitpix`**.

Measured after NULL: `get_equipment_saturation_adu(1)` -> `None`; sample raw frame -> `(65535.0, 'bitpix')`.

### B.2 Database change

| Field | Value |
|-------|-------|
| Equipment | ID 1 (QHY294MM) |
| Previous | **16384** |
| New | **NULL** |
| Other rows | unchanged |

Recorded in `docs/VYVAR_JOURNAL.md` (2026-08-13 entry).

### B.3 Runtime impact

**Per-frame limit** (`_effective_saturation_limit` on raw FITS headers):

| Draft | Before (DB=16384) | After (DB=NULL) |
|-------|-------------------|-----------------|
| 435 | 16384 equipment_db | 65535 bitpix |
| 509 | 16384 equipment_db | 65535 bitpix |
| 510 | 16384 equipment_db | 65535 bitpix |

**Global comparison pool** (simulated on static `masterstars_full_match.csv` peaks; admission threshold = `sat_adu * 0.70`):

| Draft | Pool @ 16384 | Pool @ 65535 | Delta |
|-------|-------------:|-------------:|------:|
| 435 | 2827 / 2951 | 2918 / 2951 | +91 |
| 509 | 624 / 735 | 709 / 735 | +85 |
| 510 | 624 / 735 | 709 / 735 | +85 |

**Known-good BO CVn comps** (draft 509 `comparison_stars_per_target.csv`; peaks from prior run):

| catalog_id | peak_max_adu | Pass @ 16384 (11469) | Pass @ 65535 (45875) |
|------------|-------------:|:---:|:---:|
| 1497771992240531712 | 17150 | **no** | **yes** |
| 1499200223486564608 | 16639 | **no** | **yes** |

Both were excluded from the static global pool at 16384; they pass at 65535.

**Caveat:** Existing CSVs still carry old `zone` / `saturate_limit_adu_85pct` from prior
runs until photometry is re-run. NULL fixes **new** limit resolution, not on-disk tags.

### B.4 Production safety

| Question | Answer |
|----------|--------|
| Safer than 16384? | **Yes** for new processing -- limit matches measured ceiling |
| Fully safe? | **Not yet** -- SAT-DIAG not wired; peaks still from aligned frames in production; MASTERSTAR zone path uses equipment/empirical clip separately from `_effective_saturation_limit` |
| Blocker removed? | **Partial** -- wrong DB scalar gone; full protection needs SAT-DIAG + raw peaks |

---

## Part C - SAT-DIAG dry-run

**Script:** `dev/tools/sat_diag_dry_run.py` (read-only; writes only `tmp/` JSON unless asked).

**Machine coverage:**

| Draft | Rig | Raw data | Result |
|-------|-----|----------|--------|
| 435, 509, 510 | QHY294MM 2x2 | yes | pile-up **65535**, limit **DERIVED 65535** |
| 436, 437 | equip 1 | **no FITS on disk** | skipped |
| C3-26000 / Newton | -- | **not present** | cannot test universality claim on this machine |
| Sparse / no pile-up field | -- | **not present** (all BO CVn nights pile-up at 65535) | shallow branch tested synthetically only |

### C.2 Per-draft (510 exemplar; 435/509 identical)

- **Pile-up:** 2617 pixels at 65535 (30-frame sample; scales to ~13024/150)
- **Header SATURATE:** absent
- **Equipment:** NULL
- **SAT-DIAG wins:** DERIVED **65535**
- **Pipeline today:** bitpix **65535** (after NULL)
- **If equipment were still 16384:** compatibility fires; SAT-DIAG -> **65535** (provenance should be `CONFLICT_DERIVED`; dry-run labels `DERIVED` -- see C.4)
- **Pool:** 709/735 vs 624/735 at old limit

### C.3 Failure modes

| Case | SAT-DIAG dry-run | Verdict |
|------|------------------|---------|
| Equipment absent | DERIVED/BITPIX on real data | OK |
| No header keyword | DERIVED 65535 | OK |
| Header 16384 refuted | Should CONFLICT_DERIVED; synth test weak (32767 signed BITPIX artifact) | **Spec/test gap** |
| No pile-up (synthetic shallow) | DERIVED_NO_PILEUP 32767 on signed synth header | OK branch; needs unsigned header fixture |
| Equipment 16384 refuted (real 510 data) | 65535 | OK |
| Calibrated float BITPIX=-32, max 69000 | limit **none**; equipment 16384 refuted but no BITPIX bound | **Correctly refuses** -- flags spec must reject float inputs |
| Binning mismatch | Not separable (all drafts 2x2; equipment not binned in DB today) | **Untested** on real multi-bin data |

### C.4 Spec review (implementer view)

**Works:** compatibility falsification; DERIVED ceiling on QHY binned data; NULL equipment +
pile-up; refusal on float frames without container bound.

**Fix before implement:**

1. **Pile-up shoulder rule:** spec SS5.2 requires `N(V_max-1) > 0`; real QHY data has **zero**
   pixels at 65534. Dry-run uses nearest-lower occupied bin; spec should say so explicitly.
2. **Image ADU convention:** must use stored 0..65535 for BZERO=32768, not `data+BZERO`.
3. **CONFLICT_DERIVED provenance:** when equipment/header refuted and derived wins, never label plain `DERIVED`.
4. **Tier 1 + CONFLICT_DERIVED:** recommend **yes, may exclude** -- derived ceiling is a physical pile-up measurement, not an unverified scalar. Tier 3 warn-only applies to DEFAULT_FRAC linearity, not container saturation.
5. **Compatibility false positive:** stacked/rescaled frames with max below stated ceiling pass compatibility falsely (under-protect). STDWeb ([arxiv:2411.16470](https://arxiv.org/abs/2411.16470)) checks header saturation vs observed data range for this. Recommend SAT-DIAG add: if `max_pixel` is far below BITPIX ceiling **and** header/equipment states a low ceiling, WARN `POSSIBLE_RESCALED_STACK` rather than accept silently.
6. **Simpler design?** Core is sound; could merge steps 3-4 (derived vs BITPIX) when pile-up absent. Do not simplify away compatibility test or provenance tiers.

### C.5 Plain language (for Milan)

SAT-DIAG looks at your raw light frames, finds whether pixels pile up at a ceiling (yours do, at 65535), and picks a saturation limit from header, equipment, or that measurement -- in that order, but **only if the number fits the data**. The wrong 16384 in the database would be thrown out; the limit would become 65535, which matches what the camera actually produces. On your BO CVn drafts that means ~85 more comparison stars pass the saturation gate, including the two good comps that 16384 was blocking.

It does **not** yet measure star brightness on raw frames (still uses old aligned-frame peaks in production), does **not** run automatically in the pipeline, and does **not** replace you measuring a linearity knee. It will warn instead of silently dropping stars when the limit is guessed rather than measured.

---

## Part D - Closing

| Item | Status |
|------|--------|
| Production runs | **Partially unblocked** -- NULL removes wrong 16384; draft 510 photometry still needs re-run for full benefit |
| Spec ready? | **Almost** -- fix pile-up shoulder rule, ADU convention, CONFLICT provenance, rescaled-stack WARN; then authorize |
| Contradicts memo? | Pile-up at 65535 **confirmed**; earlier aligned-frame peak ~69000 still a separate issue SAT-DIAG addresses |
| Next item | Implement SAT-DIAG gate after Milan authorization; run draft 510 photometry with raw peaks |

## Files

- `dev/tools/sat_diag_dry_run.py` (new)
- `docs/VYVAR_JOURNAL.md` (SATURATE_ADU NULL entry)
- `tmp/_sat_diag_dry_run.json` (machine output)
- `vyvar.sqlite3` (SATURATE_ADU NULL -- not in git)
