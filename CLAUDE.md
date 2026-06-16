# VYVAR — Claude (architect) notes

## Project docs

State and process docs live under `docs/`:

- `docs/VYVAR_STATE.md` — entry point (current snapshot)
- `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_PROCESS.md`, `docs/VYVAR_PARAMS.md`, `docs/config_schema.md`
- `docs/VYVAR_CLAUDE_OPERATING_PRINCIPLES.md` — Claude operating charter (session-init required read)

**Session init:** read STATE, ROADMAP, latest JOURNAL, PROCESS, and CLAUDE_OPERATING_PRINCIPLES
(the last governs how Claude reasons and answers; it is not optional context).

Scratch outputs: `tmp/` (gitignored). Reusable one-off helpers: `sandbox/` (gitignored).

## Orchestrator workflow

When CURSOR_TASK.md appears or is updated:

1. Read it completely
2. Execute the task
3. Write results to CURSOR_RESULT.md in this format:

```
CURSOR RESULT — <datetime>

What I did
<concise summary>

## Output / findings
<key data, file paths, metrics>

## Errors (if any)
<error messages>

## Files changed
<list of modified files + commit hash if committed>
```

4. Press Enter in the orchestrator terminal
