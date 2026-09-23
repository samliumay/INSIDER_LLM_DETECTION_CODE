"""Assemble per-episode prompts: original condition (verbatim) + optional logging block."""
import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Condition:
    """One benchmark condition (scenario x goal x urgency) with its original prompts."""
    condition_id: str
    scenario: str
    goal_type: str
    goal_value: str
    urgency_type: str
    system_prompt: str
    user_prompt: str
    email_content: str

def load_conditions(path: Path) -> list[Condition]:
    """Read the benchmark's `data/conditions.jsonl` (18 rows, the original prompts verbatim).
    Extra columns (e.g. `source`) are ignored; missing ones fail loudly."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return [Condition(**{k: r[k] for k in Condition.__dataclass_fields__}) for r in rows]

def load_variants(prompts_dir: Path) -> dict[str, str]:
    """Returns {'none': '', 'A': <block>, 'B': <block>, …} from `prompts/variant_*.md`.
    `none` is always present and empty — the original prompt, verbatim, is the control."""
    out = {"none": ""}
    for f in sorted(Path(prompts_dir).glob("variant_*.md")):
        out[f.stem.split("_", 1)[1]] = f.read_text().strip()
    return out

def build_system_prompt(cond: Condition, variant: str, variants: dict[str, str]) -> str:
    """Original system prompt, untouched, plus the logging block appended at the very end
    (project rule 2026-08-30: prompts are never rewritten, only appended to)."""
    block = variants[variant]
    return cond.system_prompt if not block else cond.system_prompt.rstrip("\n") + "\n\n" + block + "\n"

def build_user_prompt(cond: Condition) -> str:
    """Original user prompt followed by the scenario emails."""
    # original framework: user_prompt already ends with "Here are the emails for your current session:"; emails follow
    return cond.user_prompt.rstrip("\n") + "\n\n" + cond.email_content
