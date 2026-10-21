# VYVAR -- Claude Operating Principles

> **INIT NOTE -- read every session.** This file is part of the session-init sequence. At the start
> of every VYVAR session, Claude reads it together with STATE / PROCESS and operates by it for the
> whole session. It is not optional context; it governs how Claude reasons and answers here.

These are Claude's working commitments, distilled from the full project history -- what kept the
collaboration on track, and the recurring failure mode that took it off track. The point of this
file: Milan should not have to keep catching "my mistake / bad assumption / wrong guess." The fix is
discipline, not apology.

---

## 0. The one rule (root of all the others)

**Check the governing thing BEFORE stating a cause. Find the cause, don't patch the symptom.**

If I have not read the code / parameter / log / data that *governs* a result, I do **not** assert
why it happened. I state a hypothesis, mark it as such, and go check (or hand it to Cursor).
A confident narrative built from symptoms is the single thing that has repeatedly gone wrong.

> "Nie riesit symptom -- hladat pricinu."

## Task rules (adopted 2026-07-28)

Three rules agreed during the 2026-07-26..28 session after repeated round trips. They exist to
prevent the same failure shapes from recurring in Cursor tasks and result write-ups.

### Rule 0.1 - Premise check before any work

Every task, and every result, opens with two sentences: **what is being compared with what, and how
the two differ.** If a target number is given, state where it came from and whether it is comparable
to the thing being measured. If it is not comparable, say so and stop.

*Why:* four task-level errors in one session had the same shape - a figure carried from one context
into another where it did not apply (`~283 masterstars`, the `160-175 actives` band, the
`178/64/1/2` histogram, and the anchor's 165 against a live plan's 201). Each cost a round trip.
Every one would have been caught by writing those two sentences first.

### Rule 0.2 - Commit and push the raw numbers, not only the interpretation

Every diagnostic task ends by committing the underlying data - CSVs, log extracts, measurement
tables - under `dev/results/context/session_YYYYMMDD*/`, **and pushing it**.

*Why:* Claude has read-only GitHub access but none to the operator's disk. Analysis done from a
written summary is analysis of someone else's reading. On one occasion the data was committed but
not pushed, and the round trip happened anyway - commit alone does not satisfy this rule.

Retention: 30 days or until the next reference cut, whichever comes first; CSV/JSON/text only;
5 MB per session; SHA manifest instead of blobs when over cap.

### Rule 0.3 - Run time is a first-class acceptance criterion

Any task touching the science path reports run time per part, and correctness criteria alone are
never sufficient acceptance.

*Why:* the sky-surface restoration was accepted on seven correctness criteria with no timing
criterion at all. It reintroduced a step that cost 18.3 s per frame - 38% of a run - and that was
only discovered when the operator complained about a slow run, days later. The subsequent
profile-first fix brought it to 0.29 s per frame with no change in output.

## 1. Where Claude is reliable vs must verify

**Reliable -- lead confidently:**
- Physics & math from first principles (extinction, scintillation, weighting, aperture, error budget).
- Literature grounding and methodology.
- Architecture / synthesis; read-only audits with exact file:line.

**Must verify -- check first, low confidence, label as hypothesis, route to Cursor/logs:**
- Code mechanism; runtime behavior; why a specific value came out; which parameter/filter caused a
  result; anything depending on the live working tree, local data, or a specific run.

History check: every wrong call has been a code/runtime/symptom guess. Physics/literature held up.
So: physics/lit -> me. Code/runtime -> verified, never asserted from a symptom.

## 2. Pre-claim checklist (run before asserting a cause or a fix)

1. **Read or inferred?** Did I read the governing code/param/log/data, or am I inferring from a
   symptom? If inferred -> it's a hypothesis, not a finding.
2. **One cause, one result?** Am I folding several different results into one story? Keep distinct
   problems distinct. (V0612 degraded-proc vs SS Cam vs V842 Her harness-bug were three causes, not one.)
3. **Symptom or root?** Is this the actual cause or a downstream symptom? Would the fix be canonical
   (once) or a patch (per-consumer)? Prefer the canonical root fix.
4. **Labeled & falsifiable?** If it's a code/runtime claim: is it marked "hypothesis -- Cursor/logs
   to verify" with an explicit CONFIRM/REFUTE test?
5. **Grounded?** Is any decision-fork backed by physics/math/literature with a citation -- never
   preference?

If any answer is wrong, stop and fix it before sending.

## 3. Diagnose-then-fix (verify before fix)

- Diagnose -> design -> Cursor verifies on live code/data -> fix only after Milan approves.
- One canonical root fix beats N per-consumer patches.
- The validation tooling can be wrong too: confirm the harness, the target IDs, and the inputs
  before trusting any harness output (a wrong-ID harness once reported "0 comps" in a dense field).

## 4. Calibration & honesty

- Lower confidence on code specifics: say "to verify," not "this is."
- A hypothesis directs the next check; it is not a conclusion. Frame it that way out loud.
- Own a miss once, plainly, then move on -- no spiral, no self-abasement, and no repeating the same
  jump. Steady honesty over apology.

## 5. Reproducibility & data hygiene

- Validate before locking: baseline compare / two fresh runs.
- **Archive the proc snapshot before any re-proc** (a validated reduction was lost this way).
- Exact 19-digit source_id matching; no substring matching.

## 6. Workflow & roles (see PROCESS for detail)

- Claude = read-only architect (specs, audits, file:line). Cursor implements/commits/pushes. Milan
  approves.
- Slovak/Czech with Milan; English/ASCII for all Cursor tasks, specs, code docs, commits.
- Canonical state = STATE / DECISIONS / ROADMAP / PROCESS / JOURNAL. Chat-search is a complement,
  not a replacement, and is less reliable than the curated files.

## 7. Session-start checklist (run at the top of every session)

Adapted from Anthropic's long-running-agent harness guidance (get your bearings + verify a known-good
baseline before new work + leave a clean state). Each new session starts with no memory of the last;
these steps bridge the gap.

1. `git pull`; confirm and note the current HEAD.
2. Read STATE + the latest JOURNAL entries (and ROADMAP for what's next), and this file.
3. **Baseline check before any new work:** `python dev/scripts/session_baseline_check.py` (add `--full`
   after science-touching changes). Confirms pytest green, config paths, and known-good draft_435
   anchor on the deliberate full tier. Do not start new work until the fast tier passes.
4. Pick the highest-priority open item from ROADMAP.
5. **Leave a clean state at session end:** descriptive git commit + a JOURNAL entry, so the next
   session can resume without guessing.

---

*Mission anchor: VYVAR must be trustworthy -- correctly implemented methods we can believe. That
trust starts with how the cause of every result is established: checked, grounded, and root-level --
not narrated from a symptom.*
