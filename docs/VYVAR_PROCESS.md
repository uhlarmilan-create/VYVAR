# VYVAR — Process & conventions

How we work. The aim is that every change is **understood in its impact**, reflected in
**config and UI together**, and **documented** — so nothing drifts silently. This file
supersedes the old TODO-DEV-PROCESS item.

---

## Definition of Done (every change)

A change is not done until all six hold:

1. **Audit the existing code first.** Read and note (with `file:line`) what is already there
   in the area you are touching, *before* deciding the change. Do not build on an unaudited
   base. (Lesson 2026-06-02: the comp-selection audit came late and revealed the design only
   after work depended on it — audit first next time.)
2. **Impact analysis.** State what the change touches — numbers / reports / exports /
   downstream stages — and whether the photometry output is expected **byte-identical** or is
   an **intended** change. If intended, write the re-validation plan.
3. **Config.** Any new parameter is added to `config.py` with a sensible default and a clamp.
4. **UI parity.** The same parameter is surfaced in the Settings UI (label, help, range), and
   recorded in `docs/VYVAR_PARAMS.md`. *A parameter that exists only in config is not done.*
5. **Docs + citation.** Update `docs/VYVAR_STATE/ROADMAP/DECISIONS/JOURNAL` as appropriate; close
   any stale TODO. If the change adds a *method*, add a `CITATIONS.bib` entry and wire it into
   the conditional emitter (gated to when the method actually runs).
6. **Verification.** Reproduce the expected numbers; for read-only changes prove byte-identity
   (SHA-256 over LC / comp_quality / comparison_stars / exports); keep **0 PDF overflow** (R1);
   `pytest` green.

## External data (Brno) — PSF / NEIGHBOR-SUB gate

Before publishable PSF or NEIGHBOR-SUB on incoming external data: characterize plate scale,
pixel sampling, ePSF-vs-star Moffat mismatch (decisive), and crowding (`compute_crowding_index`).
Fine scale + mismatch ~1.0 is the validated regime (draft 367). Coarse / under-sampled data
(mismatch > ~3%) falls back to SAFE_LOW_YIELD — bright-neighbour blends **REFUSE**, not silent
deblend. PSF is validated publication-grade on synthetic fine-scale truth only; enable on real
data only after this gate passes. Standing rule: `docs/VYVAR_DECISIONS.md` (Brno section).

## Synthetic harness discipline

When the inject-and-recover harness **fails to reproduce** a real-field effect after ~2-3 fix
attempts, stop guessing prototype tweaks. Decompose on the **real production functions** (as V3d
bias decomposition did) and/or consult the literature. Example (2026-06-09): mid-mag PSF bias was
solved by SNLS/ZTF sky-only weighting practice (Astier 2013; Lacroix 2025), not by more sky
estimator prototypes alone.

## Byte-identity discipline

- **Read-only change** (QA, metadata, reports-only): the numeric photometry must be
  **byte-identical**. Prove it with a SHA-256 over `lightcurve_*.csv`, `comp_quality_*.json`,
  `comparison_stars_per_target.csv`, and the AAVSO/VarAstro data rows. Report-text changes are
  allowed only where intended (badge/note), and the diff must be *only* that text.
- **Science-changing edit** (alters which comps / magnitudes): byte-identity will not hold.
  Acceptance becomes a **small, bounded, explainable** diff — quantify how many targets change
  and `max |Δmag|` / `max |Δlc_rms|`, and confirm it is `≪` the photometric error. If the
  change is large, stop and re-think (e.g. the reverted proximity tie-break churned 143/143
  targets — that failed the bar).

## Config ↔ UI parity

Every parameter lives in three places, kept in sync:

1. `config.py` (default + clamp) — and `config.json` for the shipped default.
2. The Settings UI (so a user can see and change it).
3. `VYVAR_PARAMS.md` (the registry: key → default → clamp → UI location).

Run a parity check whenever parameters are added or changed: list keys in `config.py` not
surfaced in the UI (and vice-versa). Default-OFF experimental flags may stay UI-hidden, but
that choice is recorded in PARAMS, not left implicit.

> Note: `config.json` also gets session/UI state rewritten each run (the CONFIG-CHURN issue).
> It still holds real overrides — **do not gitignore it**; the durable fix is a separate
> session-state store (ROADMAP).

## Shared-core / no duplicated logic

A capability has **one** implementation in a core module, called by both the pipeline and any
CLI / standalone tool. (E.g. `comp_qa_core`, `trust_flag_core`, `xval_run.py` harness,
`citations`.)
Never fork the math between a standalone script and the production stage.

## Gating discipline

New behaviour ships behind a config flag with a conservatively chosen default (OFF unless it is
proven to beat the current path). The flag also feeds the citation context, so a method is
cited only when it runs. A "silent no-op" flag (enabled but not wired) is a bug — wire it or
default it correctly (lesson: `psf_photometry_enabled` was a silent no-op until the reader
carried `psf_flux`).

## Verification gates

- **R1 — PDF overflow:** 0 violations (`verify_pdf_overflow.py`), always.
- **Reproducibility:** a standalone QA result and its productionized stage must produce the
  same numbers on the same draft.
- **Tests:** `pytest tests/` green before commit.
- **Cross-checks:** prefer an independent witness (sep) over re-running the same engine.
- **Cross-val storage:** offline harness only (`xval_run.py` → `xval_out/` scratch or
  `validation/xval_ledger.csv` summary rows). No in-pipeline per-draft SEP stage; trust gate
  uses comp_qa + check-star + lc_quality only.

## Accepted lint style (Phase H, 2026-06-08)

VYVAR consciously does **not** enforce these ruff codes in production scope — they are style-only
and many `--unsafe-fixes` variants can hurt readability or risk subtle behavior change:

- **SIM102** (collapsible-if), **SIM103** (needless-bool), **SIM108** (if-else-to-ternary),
  **SIM113** (enumerate-for-loop)
- **RUF005** (collection-literal-concat), **RUF059** (unused-unpacked)

Future audits should treat them as **accepted**, not open work. Value-filtered cosmetic fixes
(SIM118 `.keys()` where safe, RUF022 sorted `__all__`, RUF007 `pairwise`, RUF034 dead-ternary)
are applied when byte-identical; ProcFrameStore keeps explicit `.keys()` (no `__iter__`).

## Cursor / Claude workflow

- **Claude** specs, audits (read-only, with `file:line`), designs, and reviews; recommends
  dispositions; writes the docs.
- **Cursor** implements and commits/pushes on the Windows dev repo (`C:\ASTRO\python\VYVAR\`).
- Each step is **verified before commit** (numbers reproduced, byte-identity / bounded-diff
  checked, overflow 0), and the commit message states what changed and why.
- **Language:** Cursor ↔ Claude communication and all handoff artifacts (`CURSOR_TASK.md`,
  specs, code, commit messages, project docs) are in **English**. Milan ↔ Claude conversation
  is in **Czech/Slovak**.
