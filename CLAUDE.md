# VYVAR — Claude (architect) notes

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
