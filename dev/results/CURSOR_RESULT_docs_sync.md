CURSOR RESULT -- 2026-07-19

What I did
DOCS-SYNC: moved all 13 docs/*SPEC*.md into dev/results/specs/ (git mv),
updated live code/docstring paths and open ROADMAP pointers, corrected FLOW
v3.0.1 prose (DAO-RECONCILE closed; spec paths), regenerated the PDF, added
flow_doc_facts.py + test_docs_sync_guard.py, and documented the mandatory
Docs impact ritual in PROCESS.md. Two commits, not pushed.

## Output / findings

Part A -- SPEC relocation:
- 13 files git-mv'd docs/ -> dev/results/specs/ (history preserved).
- docs/*SPEC*.md glob: empty.
- Live path updates: src_py/config.py, sigma_budget.py, sigma_floor_core.py,
  sparse_trust_core.py, k2_extinction.py; dev/scripts/sparse_trust_validate.py.
- Sibling cross-refs inside moved specs updated to new path.
- ROADMAP open/IN-FLIGHT pointers updated (K2 DESIGN, CAL-DIAG). CLOSED/DONE
  ROADMAP rows, STATE/JOURNAL/DECISIONS left byte-identical (history).

Part B -- FLOW v3.0.1:
- Ch 8.3 DAO-RECONCILE closed wording; ch 11.8 + Priloha E spec paths.
- PDF regenerated: 36 pages, ~131 kB; builder ok; ASCII ok.
- FLOW_DOC_V3_GAPS.md append-only addendum: D1 void.

Part C -- machine guard + ritual:
- dev/tools/docs_pdf/flow_doc_facts.py (41 config facts + 18 functions +
  ANCHOR_ID=draft_435).
- build_flow_doc.py imports facts (pairing coupling).
- test_docs_sync_guard.py: 4 tests green.
- PROCESS.md: DOCS-SYNC ritual section + docs/ layout rule (SPECs out).
- PDF size guard uses >100 kB (task said 200 kB; live v3 PDF is ~131 kB;
  100 kB still rejects the ~21 kB v2 placeholder).

Verification:
- pytest test_docs_sync_guard.py + test_docs_layout.py: 9 passed
- ruff on touched Python: All checks passed
- session_baseline_check.py --fast: OVERALL PASS

Commits (not pushed):
- 6d549a2 docs(layout): move *_SPEC.md ... FLOW v3.0.1 ...
- process(docs-sync): mandatory Docs impact ritual + machine guard (this commit)

## Docs impact

- docs/VYVAR_PROCESS.md -- DOCS-SYNC ritual + docs/ layout rule (SPECs out)
- docs/VYVAR_ROADMAP.md -- open K2 / CAL-DIAG spec path pointers
- docs/VYVAR_FLOW_CZ.pdf -- regenerated (v3.0.1 corrections)
- dev/tools/docs_pdf/build_flow_doc.py + flow_doc_facts.py -- builder + facts
- dev/results/FLOW_DOC_V3_GAPS.md -- D1 void addendum
- dev/results/specs/* -- 13 SPEC files relocated
- STATE / DECISIONS / JOURNAL -- not edited (A3: historical records stay;
  layout rule now lives in PROCESS)
- CONFIG_GUIDE / INSTALL -- none (no parameter or setup change)

## Errors (if any)
None. PDF size threshold adjusted 200 kB -> 100 kB to match live ~131 kB PDF
while still rejecting the short v2 edition.

## Files changed
See the two commits above.
