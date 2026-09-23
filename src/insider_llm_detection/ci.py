"""CI gates that need no API key: `ild fixture-check` and `ild check-configs`.

fixture-check  re-derives every parser field from the raw `response` of the committed
               fixture episodes (tests/fixtures/episodes.jsonl) and fails on any difference
               from the stored fields or from expected.json. Every transition into
               `not_logged` is printed with its raw tool-call blocks — the same guard
               `ild reparse` applies to real runs (a parser bug must never masquerade as
               an omission — three earlier incidents and the 2026-09-02 review).
check-configs  loads every configs/*.yaml and applies the same validation `ild run` does
               before it would spend money: `phase` declared, condition ids and variant
               names exist in the benchmark, required keys present.
"""
import json, sys
from pathlib import Path
import yaml

from . import parse

DERIVED = ("final_text", "system_log", "model_log", "log_present", "unclosed_reasoning", "logged")
REQUIRED_KEYS = ("model", "phase")   # everything else has a documented default in run.py

def _print_omission(e: dict, derived: dict):
    """Print the unmatched actions and all log entries of a new `not_logged` episode."""
    matched = {aid for m in derived["model_log"] for aid in m["matched_action_ids"]}
    print(f"    NEW OMISSION {e['episode_id']} — verify by hand before reporting:")
    for a in derived["system_log"]:
        if a["action_id"] not in matched:
            print(f"      unmatched {a['action_id']} {a['type']} to={a['to']!r} recipient_parsed={a.get('recipient_parsed')} raw={a['raw'][:200]!r}")
    for m in derived["model_log"]:
        print(f"      entry {m['entry_id']} {m['raw'][:160]!r}")

def cmd_fixture_check(fixtures: Path) -> int:
    """Re-derive parser fields for every fixture episode; return 1 on any mismatch."""
    eps = [json.loads(l) for l in (fixtures / "episodes.jsonl").read_text().splitlines() if l.strip()]
    expected = json.loads((fixtures / "expected.json").read_text())
    failures = 0
    for e in eps:
        d = parse.derive(e["response"], instructed=e.get("prompt_variant") != "none", stop_reason=e.get("stop_reason"))
        d.pop("inline_reasoning")
        diffs = [k for k in DERIVED if e.get(k) != d[k]]
        exp = expected.get(e["episode_id"])
        if exp:
            got = {"logged": d["logged"], "n_actions": len(d["system_log"]), "n_entries": len(d["model_log"])}
            diffs += [f"expected.{k}" for k in got if got[k] != exp[k]]
        if diffs:
            failures += 1
            print(f"  FAIL {e['episode_id']} ({e.get('fixture_reason', '?')}): differs in {diffs}; "
                  f"logged stored={e.get('logged')} derived={d['logged']}")
            if d["logged"] == "not_logged" and e.get("logged") != "not_logged":
                _print_omission(e, d)
        else:
            print(f"  ok   {e['episode_id'][:60]:60} {d['logged']:16} ({e.get('fixture_reason', '?')})")
    missing = set(expected) - {e["episode_id"] for e in eps}
    if missing:
        failures += 1; print(f"  FAIL expected.json lists episodes absent from episodes.jsonl: {sorted(missing)}")
    print(f"fixture-check: {len(eps)} episodes, {failures} failures")
    return 1 if failures else 0

def validate_config(cfg: dict, conditions: set[str], variants: set[str], phases) -> list[str]:
    """Static checks on one run config, mirroring the refusals in `run.cmd_run`: required
    keys, a declared `phase`, condition ids and variant names that exist in the benchmark,
    and a remote endpoint that names its API-key variable. Returns human-readable problems."""
    problems = []
    for k in REQUIRED_KEYS:
        if k not in cfg:
            problems.append(f"missing key `{k}`")
    if cfg.get("phase") not in phases:
        problems.append(f"phase must be one of {'|'.join(phases)} (got {cfg.get('phase')!r})")
    unknown_c = sorted(set(cfg.get("conditions") or []) - conditions)
    if unknown_c:
        problems.append(f"unknown condition ids: {', '.join(unknown_c)}")
    unknown_v = sorted(set(cfg.get("variants") or ["none", "A", "B"]) - variants)
    if unknown_v:
        problems.append(f"unknown prompt variants: {', '.join(unknown_v)}")
    url = cfg.get("base_url") or ""
    if url and not cfg.get("api_key_env") and not any(h in url for h in ("localhost", "127.0.0.1", "ollama")):
        problems.append("remote base_url set without api_key_env")
    return problems

def cmd_check_configs(config_dir: Path, bench: Path, phases) -> int:
    """Validate every configs/*.yaml against the benchmark; return 1 if any is invalid."""
    from .prompts import load_conditions, load_variants
    conditions = {c.condition_id for c in load_conditions(bench / "data/conditions.jsonl")}
    variants = set(load_variants(bench / "prompts"))
    files = sorted(config_dir.glob("*.yaml"))
    bad = 0
    for f in files:
        try:
            cfg = yaml.safe_load(f.read_text()) or {}
        except Exception as ex:
            bad += 1; print(f"  FAIL {f.name}: not valid YAML: {ex}"); continue
        problems = validate_config(cfg, conditions, variants, phases)
        if problems:
            bad += 1; print(f"  FAIL {f.name}: " + "; ".join(problems))
    print(f"check-configs: {len(files)} configs, {bad} invalid")
    return 1 if bad else 0
