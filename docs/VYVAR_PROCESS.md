# VYVAR - Process & conventions

How we work. The aim is that every change is **understood in its impact**, reflected in
**config and UI together**, and **documented** - so nothing drifts silently. Definition-of-Done
discipline lives here; open harness items **DEV-PROCESS-A/B** are spec'd in ROADMAP.

**Session init:** read STATE, ROADMAP, latest JOURNAL, PROCESS, and
`docs/VYVAR_CLAUDE_OPERATING_PRINCIPLES.md` (the last governs how Claude reasons and answers; it is
not optional context). At the **first session of each week**, also run
`python dev/tools/invariants_report.py` (optionally `--write`) -- the weekly
invariants cadence; `--fast` itself is unchanged.

## Repository layout & ritual paths (post REPO-REORG, 2026-07-17)

Production code lives in `src_py/`; development material in `dev/`
(`dev/tests`, `dev/tools`, `dev/validation`, `dev/scripts`, `dev/sandbox`,
`dev/orchestrator`); disposable scratch in `tmp/` (gitignored). See CLAUDE.md
for the full map.

- **Session baseline / anchor gate:** `python dev/scripts/session_baseline_check.py --fast`
  (includes `PRAGMA quick_check` on `vyvar.sqlite3` since MS-SOURCES-RETIRE-01 Phase 4).
  (default tier: git state, config paths, full `dev/tests` pytest, anchor manifest-db parity hint,
  deps outdated hint). Record the **git commit SHA** next to every reported PASS/FAIL; a result
  from commit A is not evidence on commit B.
  (git/config/ledger + pytest) and `--full` (headless `run_full_photometry_pipeline`
  byte-identical anchor gate).
- **Registry -> docs:** `python dev/tools/gen_params_md.py` regenerates
  `docs/VYVAR_PARAMS.md` from `dev/validation/params_registry.json`.
- **Ledger:** `dev/validation/VYVAR_VALIDATION_LEDGER.json`.
- **No git during `--full`:** do not run git operations (`add`, `commit`, `checkout`)
  while `session_baseline_check.py --full` is running; the 2026-08-21
  `index.lock` / `config.json` incident is the reason.

### Full-suite pytest comparability

Pass/skip counts from `python -m pytest dev/tests` are only comparable across commits when
taken on the **same on-disk data state**. A linked worktree or fresh clone (tracked files
only) silently drops gitignored paths (`Archive/`, `GAIA_DR3/`, exoplanet DB, VSX): data-gated
tests skip instead of running, and at least one data-gated assert fails outright.
**Worked example:** `test_exoplanet_promotion_restore.py:106-110` asserts the exoplanet local
DB path is a file (`p.is_file()`); it passes on the main tree and fails in a worktree without
the gitignored DB. For before/after commit comparisons, run the full suite on **main** with
gitignored data present (or stash on main); do not use a bare worktree as the baseline.

- **Result-file location rule:** ALL `CURSOR_RESULT_*.md` and `CURSOR_TASK_*.md`
  working documents go in `dev/results/` (never at the repo root). Sandbox
  harnesses stay in `tmp/`.
- **`docs/` rule:** `docs/` holds ONLY the user-facing living documentation family
  -- how the project is designed, set up, and operated (state, roadmap, journal,
  process, decisions, runbook, code map, schemas, calibration/validation, config
  guides). Development/design `*_SPEC.md` artifacts live in `dev/results/specs/`
  (never in `docs/`). Historical audit outputs, one-off investigation writeups,
  and target- or draft-specific analyses are work results and belong in
  `dev/results/`. `docs/` contains no subdirectories and no `CURSOR_*` files. It is
  Markdown-only, except for PDF guides (`*.pdf`) that a committed builder under
  `dev/tools/docs_pdf/` regenerates (e.g. the CZ parameter handbook, install
  guide, and FLOW pipeline doc). The `dev/tests/test_docs_layout.py` and
  `dev/tests/test_docs_sync_guard.py` guards enforce this.

---

## DOCS-SYNC ritual (mandatory)

Whenever code or behavior changes, **all affected documentation must be updated
in the same work** -- not as a later habit. Drift turns tests red.

1. **CURSOR task** must include a "Docs impact" assessment (which docs will
   change, or why none).
2. **RESULT file** must include a **"Docs impact:"** section -- either the list
   of updated documents, or the explicit line `Docs impact: none (reason)`. A
   RESULT without this section is incomplete.
2b. **RESULT file** must also include a **"Recurrence:"** one-liner (see
   Recurrence rule below):
   `Recurrence: new test <name> | existing <name> | n/a (first occurrence / not a bug-class)`.
   A RESULT without this field is incomplete (same enforcement style as Docs impact).
3. **Docs impact checklist** (walk each item):
   - **STATE** -- always (snapshot of what is true now).
   - **DECISIONS** -- if a decision was made or reversed.
   - **ROADMAP** -- item status changes.
   - **FLOW builder + PDF** -- if pipeline flow, algorithms, or parameter
     semantics/defaults changed: edit `dev/tools/docs_pdf/build_flow_doc.py`,
     update `dev/tools/docs_pdf/flow_doc_facts.py`, regenerate
     `docs/VYVAR_FLOW_CZ.pdf`.
   - **CONFIG_GUIDE / parameter handbook** -- if parameters were
     added/renamed/re-defaulted.
   - **INSTALL guide** -- if setup / first-run changed.
4. **Machine guard:** `dev/tests/test_docs_sync_guard.py` enforces the
   config-facts and function-name layers automatically (plus zero `*_SPEC*.md`
   under `docs/`). A red docs-sync test blocks the commit like any other
   invariant.
5. **Layout rule:** `*_SPEC.md` files live in `dev/results/specs/`
   (development artifacts), never in `docs/`. `docs/` holds the user-facing
   family only.

---

## Recurrence and forensic discipline

### Recurrence rule

Formalizes the existing two-correction practice:

- A bug class fixed for the **second** time gets a permanent regression test
  named `test_recur_<class>...` plus a DECISIONS pointer.
- Every RESULT file must carry the mandatory one-line field:
  `Recurrence: new test <name> | existing <name> | n/a (first occurrence / not a bug-class)`.

**Closing without a guard.** A defect closure that does not add a regression guard (test,
infolog milestone, or ledger gate that would fail if the defect returns) is incomplete. That is
how the same bug class comes back silently -- F-431 (2026-07-16) and SKIPPROC (2026-07-22) both
removed the order-2 preprocess sky-surface step; the first fix had no guard, so the second
occurrence was not caught until science comparison. Prefer a named guard (`INV-PREP-01`,
`test_recur_*`, photometry SHA gate) over narrative "fixed" notes alone.

### Forensic promotion rubric

Every forensic / one-off investigation script (`dev/scripts/forensic_*`, or a
`tmp/` probe that graduates into the repo) must end in **exactly one** state,
recorded in the closing RESULT:

| State | Meaning |
|-------|---------|
| **PROMOTE** | Its discriminating assertion moves into a permanent test; script deleted or archived after promotion. |
| **ARCHIVE** | Script stays tracked with a mandatory header block (purpose, finding ID, RESULT/DECISIONS pointer, date, state: `ARCHIVED` -- no live role). |
| **DELETE** | Knowledge fully captured elsewhere; pointer recorded. |

### Weekly invariants report cadence

At the first session of each week, run `python dev/tools/invariants_report.py`
(optionally `--write` to emit `dev/results/INVARIANTS_WEEKLY_<yyyymmdd>.md`).
This is an addition to the session ritual text; `session_baseline_check.py --fast`
itself is unchanged.

---

## Definition of Done (every change)

A change is not done until all six hold:

1. **Audit the existing code first.** Read and note (with `file:line`) what is already there
   in the area you are touching, *before* deciding the change. Do not build on an unaudited
   base. (Lesson 2026-06-02: the comp-selection audit came late and revealed the design only
   after work depended on it - audit first next time.)
2. **Impact analysis.** State what the change touches - numbers / reports / exports /
   downstream stages - and whether the photometry output is expected **byte-identical** or is
   an **intended** change. If intended, write the re-validation plan (empirical cross-validation
   vs AIJ/SIPS when byte-identity anchors are retired - see STATE 2026-06-16).
3. **Config.** Any new parameter is added to `config.py` with a sensible default and a clamp.
4. **UI parity.** The same parameter is surfaced in the Settings UI (label, help, range), and
   recorded in `docs/VYVAR_PARAMS.md`. *A parameter that exists only in config is not done.*
5. **Docs + citation.** Update `docs/VYVAR_STATE/ROADMAP/DECISIONS/JOURNAL` as appropriate; close
   any stale TODO. If the change adds a *method*, add a `CITATIONS.bib` entry and wire it into
   the conditional emitter (gated to when the method actually runs).
6. **Verification.** Reproduce the expected numbers; keep **0 PDF overflow** (R1); `pytest` green.
   Byte-identity SHA when anchors are active; otherwise ground-truth metrics (DoD-A/B, SIPS parity).

## Decision-grounding rule + sandbox discipline (2026-06-15/16)

**Grounding rule** (`docs/VYVAR_DECISION_GROUNDING_RULE.md`): design forks must cite physics,
literature, or field practice - not bare engineering preference.

**Sandbox arc discipline** (simple-differential + draft_409 trust cleanup):

1. **Sandbox** - prototype / measure under `tmp/phase*` (not committed).
2. **Measure** - DoD metrics vs AIJ / SIPS / constant calibrator.
3. **Milan review** - PDF + numbers before production commit.
4. **Commit** - source + tests + docs only; harnesses stay in `tmp/`.

**Production-path harness rule (2026-07-27, POST-453):** A sandbox or replay harness that
does not go through the production code path cannot characterise production behaviour.
Example: the photutils 3.0 "+1.1% DAO pass-1" claim came from a replay harness using
config-default FWHM (2.5 px) while production derives FWHM from the FITS header (~3.23 px on
BO CVn). Draft 452 under photutils 3.0 with production FWHM reproduced pass-1 **2552** exactly.

### Session context retention (`dev/results/context/`)

Committed measurement sessions under `dev/results/context/` follow:

| Rule | Value |
|------|-------|
| Retention | 30 days **or** until the next reference cut (whichever is sooner) |
| Allowed types | CSV, JSON, text only (no FITS blobs) |
| Size cap | 5 MB per session directory |
| Over cap | SHA-256 manifest of omitted files instead of binary blobs |
| Purpose | Push raw numbers to origin so Claude/architect can analyse without prose-only handoff |

## External data (Brno) - PSF / NEIGHBOR-SUB gate

Before publishable PSF or NEIGHBOR-SUB on incoming external data: characterize plate scale,
pixel sampling, ePSF-vs-star Moffat mismatch (decisive), and crowding (`compute_crowding_index`).
Fine scale + mismatch ~1.0 is the validated regime (draft 367). Coarse / under-sampled data
(mismatch > ~3%) falls back to SAFE_LOW_YIELD - bright-neighbour blends **REFUSE**, not silent
deblend. PSF is validated publication-grade on synthetic fine-scale truth only; enable on real
data only after this gate passes. Standing rule: `docs/VYVAR_DECISIONS.md` (Brno section).

## Synthetic harness discipline

When the inject-and-recover harness **fails to reproduce** a real-field effect after ~2-3 fix
attempts, stop guessing prototype tweaks. Decompose on the **real production functions** (as V3d
bias decomposition did) and/or consult the literature. Example (2026-06-09): mid-mag PSF bias was
solved by SNLS/ZTF sky-only weighting practice (Astier 2013; Lacroix 2025), not by more sky
estimator prototypes alone.

## Session lessons (trust/anchor, 2026-06-11)

1. **Byte-identity after every trust/QA change** - re-run `compute_photometry_sha` on the anchor
   draft; trust columns and sidecars are out of the SHA set but LC/comp selection are not.
2. **Footprint-first** before a policy that might move the anchor (e.g. raising Phase-1
   `n_comp_min` - 45 per-setup hits on draft_387; Option B trust-only split avoided re-cut).
3. **Completeness gate** - `audit_photometry_completeness` is the false-success guard for night
   runs (truncated photometry must not report `night_run_success`).
4. **Specs before implementation** - trust hardening, check-star selection, comp-floor policy
   filed under `docs/` before code landed.
5. **Confirm-reproducibility-before-locking** - two independent fresh runs must match on SHA
   before recording a new anchor (`draft_386 == draft_387`).

## Plate-solver change lock (2026-06-14)

**Locked:** scoped solver - full-cone SIP ON; ROWORDER Y-flip **OFF**; legacy mirror sweep ON;
**catalog-recovery VERIFIED gate** on MASTERSTAR (defaults: recovery >=0.65, floor 40, centre RMS
<=1.20 px; benign edge/centre ratio <=3.20). **`hint_sep`** warning-only when verified; wide tripwire
when not (`max(1.5 deg, FOV)`). Stale-hint Gaia cone recenter unchanged.

**Test-vs-production gap is recurring.** Validate solver/photometry changes on the **production
entry** (`generate_masterstar_and_catalog` / a fresh UI draft), not a sandbox harness on
`solve_wcs_with_local_gaia` alone. The 83.1% Brno artifact and re-cut harness drift are both instances.

**Regression gate before lock:** anchor footprint + science-meaningful comparator catches silent
corruption before lock (caught the 320 px ROWORDER break).

**Guard relaxation only with overlay-confirmed correctness** - never on match% alone.

**Acceptance method:** same-harness legacy-vs-scoped control
(`sandbox/anchor387_legacy_vs_scoped_gate.py`); **(B) vs (A) = 0 science failures** -> lock.
Re-cut vs frozen archive alone conflates harness drift (~1087 failures, B max |Deltamag| ~ 2.26).

**Follow-up (does not block lock):** `[TODO-RECUT-HARNESS-FIDELITY]` in ROADMAP.

## Byte-identity discipline

- **Raw byte SHA** (`compute_photometry_sha`): canonical fingerprint for a locked anchor cut.
  Two independent fresh runs must match before lock (`draft_386 == draft_387` class).
- **Science-meaningful acceptance** (`compare_photometry_science_meaningful` in
  `tests/photometry_sha.py`): use for re-baseline / additive gates vs a prior anchor when raw
  bytes differ for non-science reasons. Excludes provenance columns on
  `comparison_stars_per_target.csv` (`comp_path`, funnel cols) and LC QC columns (`err`,
  `err_inflation`, `flag`, `method`, `source_file`, lunar metadata). Tolerances: BJD/HJD
  `<= 1e-6` d; differential photometry columns (`mag_*`, `flux*`, `delta_mag`) `<= 1e-6`.
  Per-frame `err` is **out of scope** for anchor science identity (QC export; mag unchanged).
- **Read-only change** (QA, metadata, reports-only): numeric photometry must be
  **byte-identical** *or* pass the science-meaningful gate when comparing to a historical cut.
  Prove raw drift with SHA-256 over `lightcurve_*.csv`, `comp_quality_*.json`,
  `comparison_stars_per_target.csv` (core subset), plus `comp_qa_*.json` sidecars when comp QA
  methodology is in scope. AAVSO/VarAstro data rows follow the same discipline. Report-text
  changes are allowed only where intended (badge/note), and the diff must be *only* that text.
- **Science-changing edit** (alters which comps / magnitudes): byte-identity will not hold.
  Acceptance becomes a **small, bounded, explainable** diff - quantify how many targets change
  and `max |Deltamag|` / `max |Deltalc_rms|`, and confirm it is `<<` the photometric error. If the
  change is large, stop and re-think (e.g. the reverted proximity tie-break churned 143/143
  targets - that failed the bar).

### Re-baseline events (intentional SHA moves)

| date | old SHA | new SHA | reason |
|------|---------|---------|--------|
| 2026-06-09 | `770966c3...` (core 283) | `edbd97e7...` (426 incl. comp_qa) | CQ-C fix-once order-independent comp_qa locus; core LC/comp_quality/comparison unchanged; bounded diff 1 flag / 1 n_clean / 0 trust |
| 2026-06-11 | `203254fd...` / `95a5515a...` (historical pre-drift) | `3f7c9e7a...` core (2806) / `d5b72d08...` full (4285) | Sparse-fallback default ON + benign code drift (`comp_path`, BJD ~1e-9, `err` QC); rich anchor inert (0 recovery); two-run repro on `draft_000387` re-cut |
| 2026-06-11 | `f4bcc0ee...` / `bd0b1792...` (draft_385 truncated, RETIRED) | `203254fd...` core (2806) / `95a5515a...` full (4285) | Chi_and_H full zaloha anchor (draft_386; confirmed draft_387; G<=16) |
| 2026-06-11 | `d246a5be...` / `30a2f461...` (draft_382 TAP G<=19.5, RETIRED) | `f4bcc0ee...` core (1098) / `bd0b1792...` full (1113) | Chi_and_H zaloha cut from truncated run (RETIRED same day) |
| 2026-06-10 | `770966c3...` / `edbd97e7...` (deleted draft_366) | `d246a5be...` core (2810) / `30a2f461...` full (4291) | Chi_and_H TAP field DB cut (RETIRED 2026-06-11) |

## Config <-> UI parity

Every parameter lives in three places, kept in sync:

1. `config.py` (default + clamp) - and `config.json` for the shipped default.
2. The Settings UI (so a user can see and change it).
3. `VYVAR_PARAMS.md` (the registry: key -> default -> clamp -> UI location).

Run a parity check whenever parameters are added or changed: list keys in `config.py` not
surfaced in the UI (and vice-versa). Default-OFF experimental flags may stay UI-hidden, but
that choice is recorded in PARAMS, not left implicit.

> CONFIG-HUMAN-EDIT (2026-07-18): `config.json` is written as a **generated, grouped,
> commented JSONC-lite** document (loader tolerates `//` line comments; writer regenerates
> grouping + comments on every UI save). Per-key and per-section explanations come from the
> **registry `help` / `__meta__.phase_help`** fields - the single source of truth for config
> comments, dashboard tooltips and the guides. Edit key explanations in the registry, not by
> hand. Validate a hand-edited file with `python dev/scripts/validate_config.py`.

> Note: `config.json` also gets session/UI state rewritten each run (the CONFIG-CHURN issue).
> It still holds real overrides - **do not gitignore it**; the durable fix is a separate
> session-state store (ROADMAP).

> ENCODING-POLICY (2026-07-20): **ALL repo text files are ASCII-only.** This extends the
> existing "deliverables ASCII-only English / docs Czech-no-diacritics" convention to every
> tracked text file, **including** `CURSOR_RESULT_*.md`, `README.md`, `docs/README_FULL.md`,
> and `docs/README_CZ.md`.
> Em dash -> `-`, arrows -> `->` / `<-`, curly quotes -> straight, ellipsis -> `...`, etc.
> Cursor writes results in pure ASCII. Guard: `dev/tests/test_ascii_policy.py` (empty
> allowlist; justified exceptions only). Tool: `python dev/tools/ascii_migrate.py`
> (`--check` reports without writing). Append-only clarification for archives: byte-level
> ENCODING REPAIR of archived results (punctuation transcoding, zero semantic change) is
> exempt from the append-only rule; content edits remain forbidden.

## Shared-core / no duplicated logic

A capability has **one** implementation in a core module, called by both the pipeline and any
CLI / standalone tool. (E.g. `comp_qa_core`, `trust_flag_core`, `xval_run.py` harness,
`citations`.)
Never fork the math between a standalone script and the production stage.

## Gating discipline

New behaviour ships behind a config flag with a conservatively chosen default (OFF unless it is
proven to beat the current path). The flag also feeds the citation context, so a method is
cited only when it runs. A "silent no-op" flag (enabled but not wired) is a bug - wire it or
default it correctly (lesson: `psf_photometry_enabled` was a silent no-op until the reader
carried `psf_flux`).

## Verification gates

- **R1 - PDF overflow:** 0 violations (`verify_pdf_overflow.py`), always.
- **Reproducibility:** a standalone QA result and its productionized stage must produce the
  same numbers on the same draft.
- **Tests:** `pytest tests/` green before commit.
- **String literals in science modules:** any change to string literals in `src_py` science
  modules (including transliteration, wording, or formatting of emitted text) requires a
  `--full` byte-identity run before push; comments are exempt.
- **K2 / literature validation:** expected values anchor in the **spec/literature**, never in the
  code under test (no circular checks).
- **Cross-checks:** prefer an independent witness (sep) over re-running the same engine.
- **Cross-val storage:** offline harness only (`xval_run.py` -> `xval_out/` scratch or
  `validation/xval_ledger.csv` summary rows). No in-pipeline per-draft SEP stage; trust gate
  uses comp_qa + check-star + lc_quality only.

## Accepted lint style (Phase H, 2026-06-08)

VYVAR consciously does **not** enforce these ruff codes in production scope - they are style-only
and many `--unsafe-fixes` variants can hurt readability or risk subtle behavior change:

- **SIM102** (collapsible-if), **SIM103** (needless-bool), **SIM108** (if-else-to-ternary),
  **SIM113** (enumerate-for-loop)
- **RUF005** (collection-literal-concat), **RUF059** (unused-unpacked)

Future audits should treat them as **accepted**, not open work. Value-filtered cosmetic fixes
(SIM118 `.keys()` where safe, RUF022 sorted `__all__`, RUF007 `pairwise`, RUF034 dead-ternary)
are applied when byte-identical; ProcFrameStore keeps explicit `.keys()` (no `__iter__`).

**Enforced (2026-06-11):** `BLE001` + `E722` (broad/bare except) via `pyproject.toml`, pre-commit,
and `tests/test_ble001_regression.py`. New unmarked `except Exception` / bare `except:` fails CI.
Existing sites carry explicit `# noqa: BLE001`.

## Phase 2A do-no-harm validation (`err` / read noise)

When re-running Phase 2A against frozen draft LC baselines (scratch harness under `tmp/` or any
future tracked validator), pass **`db=VyvarDatabase(...)`** into `run_phase2a` /
`_phase2a_prepare_shared_state` - the same as `app.py` / `run_full_photometry_pipeline`. With
`db=None`, `resolve_read_noise` falls back to RN **10.0** e- while production uses DB RN (e.g.
**1.3** for Dablice); photon `err` inflates ~6-11% and produces false `err` diffs vs frozen LC.
Photon-error aperture always comes from proc CSV `aperture_r_px` when present
(`read_flux_from_csv`), not from SNR `apertures_px` dict.

**Proc CSV `mag` vs science flux (2026-07-07):** per-frame sidecar column `mag` is Gaia catalog
`g_mag` (constant per `catalog_id` across frames). Differential photometry and sandbox/diagnostic
harnesses must derive instrumental magnitudes from **`dao_flux`** (`read_flux_from_csv`,
`photometry_core.py:1291`) - never from proc `mag`.

## Cursor / Claude workflow

- **Claude** specs, audits (read-only, with `file:line`), designs, and reviews; recommends
  dispositions; writes the docs.
- **Cursor** implements and commits/pushes on the Windows dev repo (`C:\ASTRO\python\VYVAR\`).
- Each step is **verified before commit** (numbers reproduced, byte-identity / bounded-diff
  checked, overflow 0), and the commit message states what changed and why.
- **Language:** Cursor <-> Claude communication and all handoff artifacts (`CURSOR_TASK.md`,
  specs, code, commit messages, project docs) are in **English**. Milan <-> Claude conversation
  is in **Czech/Slovak**.
