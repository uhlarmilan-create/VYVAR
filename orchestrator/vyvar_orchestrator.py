"""
VYVAR Orchestrator — bridges Claude (claude.ai) and Cursor via files.
Milan runs this script; it handles the Claude API loop automatically.

Usage:
    python orchestrator/vyvar_orchestrator.py
    Then type your task and press Enter.
"""
import anthropic
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).parent.parent
STATE_FILE  = REPO_ROOT / "docs" / "VYVAR_STATE.md"
TASK_FILE   = REPO_ROOT / "CURSOR_TASK.md"
RESULT_FILE = REPO_ROOT / "CURSOR_RESULT.md"
LOG_FILE    = REPO_ROOT / "orchestrator" / "session.log"

SYSTEM_PROMPT = """You are Claude, architect of the VYVAR astronomical photometry pipeline.
You work in a team: Milan (product owner), Cursor (implementor), Claude (you = architect).

Your job in this orchestrator:
1. Read docs/VYVAR_STATE.md for context
2. Break down Milan's task into steps
3. For each step, output EXACTLY one of these response types:

TYPE A — task for Cursor:
CURSOR_TASK:
<your instruction in English for Cursor>
END_TASK

TYPE B — need Milan's decision:
NEED_DECISION:
<your question for Milan in Czech/Slovak>
END_DECISION

TYPE C — done:
DONE:
<summary in Czech/Slovak of what was accomplished>
END_DONE

Rules:
- Always read CURSOR_RESULT before deciding next step
- Update your mental model of VYVAR_STATE after each Cursor result
- Ask Milan only when truly necessary (architectural decisions, ambiguous requirements)
- Be concise in CURSOR_TASK — Cursor knows the codebase
- Language: CURSOR_TASK in English, NEED_DECISION and DONE in Czech/Slovak
"""

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def read_file(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""

def write_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")

def wait_for_cursor_result(timeout_min: int = 30) -> str:
    """Wait until CURSOR_RESULT.md appears or is updated."""
    log("⏳ Čakám na Cursor výsledok... (stlač Enter keď Cursor skončí)")
    # Simple approach: wait for user to press Enter
    # (Cursor doesn't auto-notify yet — TODO-ORCHESTRATOR phase 2)
    input()
    result = read_file(RESULT_FILE)
    if not result:
        log("⚠️  CURSOR_RESULT.md je prázdny")
    return result

def parse_response(text: str) -> tuple[str, str]:
    """Returns (type, content) where type is CURSOR_TASK/NEED_DECISION/DONE."""
    for rtype in ["CURSOR_TASK", "NEED_DECISION", "DONE"]:
        start = f"{rtype}:"
        end   = f"END_{rtype.split('_')[0] if '_' in rtype else rtype}"
        # handle END_TASK, END_DECISION, END_DONE
        end_map = {"CURSOR_TASK": "END_TASK", "NEED_DECISION": "END_DECISION", "DONE": "END_DONE"}
        end = end_map[rtype]
        if start in text and end in text:
            content = text.split(start, 1)[1].split(end, 1)[0].strip()
            return rtype, content
    return "UNKNOWN", text

def run_orchestrator():
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    log("=" * 60)
    log("VYVAR Orchestrator — štart")
    log(f"Repo: {REPO_ROOT}")
    log("=" * 60)

    # Read current state
    state = read_file(STATE_FILE)
    log(f"docs/VYVAR_STATE.md načítaný ({len(state)} znakov)")

    # Get task from Milan
    print("\n🔭 Zadaj úlohu pre VYVAR pipeline:")
    task = input(">>> ").strip()
    if not task:
        log("Žiadna úloha — koniec")
        return

    log(f"Úloha: {task}")

    # Build conversation history
    messages = [
        {
            "role": "user",
            "content": f"""Current docs/VYVAR_STATE.md:
---
{state}
---

Milan's task: {task}

Start with step 1."""
        }
    ]

    iteration = 0
    max_iterations = 20

    while iteration < max_iterations:
        iteration += 1
        log(f"\n--- Iterácia {iteration} ---")

        # Call Claude API
        log("🤔 Claude analyzuje...")
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        claude_text = response.content[0].text
        log(f"Claude odpoveď ({len(claude_text)} znakov)")

        # Parse response type
        rtype, content = parse_response(claude_text)

        if rtype == "CURSOR_TASK":
            log("📋 CURSOR_TASK → zapisujem do CURSOR_TASK.md")
            task_content = f"""# CURSOR TASK — {datetime.now().strftime('%Y-%m-%d %H:%M')}
Iteration: {iteration}

{content}

---
After completing this task, write your results to CURSOR_RESULT.md
"""
            write_file(TASK_FILE, task_content)
            print("\n📋 Cursor task zapísaný do CURSOR_TASK.md")
            print("   → Otvor CURSOR_TASK.md v Cursore a spusti task")
            print("   → Keď Cursor skončí a zapíše CURSOR_RESULT.md, stlač Enter")

            cursor_result = wait_for_cursor_result()
            log(f"✅ Cursor výsledok prijatý ({len(cursor_result)} znakov)")

            # Add to conversation
            messages.append({"role": "assistant", "content": claude_text})
            messages.append({
                "role": "user",
                "content": f"Cursor result:\n{cursor_result}"
            })

        elif rtype == "NEED_DECISION":
            log("❓ NEED_DECISION → čakám na Milana")
            print("\n❓ Claude potrebuje tvoje rozhodnutie:")
            print(f"   {content}")
            decision = input("Tvoja odpoveď: ").strip()
            log(f"Milan rozhodol: {decision}")

            messages.append({"role": "assistant", "content": claude_text})
            messages.append({
                "role": "user",
                "content": f"Milan's decision: {decision}"
            })

        elif rtype == "DONE":
            log("✅ DONE")
            print("\n✅ Hotovo!")
            print(f"\n{content}")
            break

        else:
            log(f"⚠️  Neznámy typ odpovede: {rtype}")
            log(f"Raw: {claude_text[:200]}")
            messages.append({"role": "assistant", "content": claude_text})
            messages.append({
                "role": "user",
                "content": "Please respond with one of: CURSOR_TASK, NEED_DECISION, or DONE."
            })

    if iteration >= max_iterations:
        log("⚠️  Max iterácií dosiahnutý")

    log("Orchestrator — koniec session")

if __name__ == "__main__":
    run_orchestrator()
