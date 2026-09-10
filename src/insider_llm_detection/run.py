"""Single entrypoint. `ild run --config configs/smoke.yaml`."""
import argparse, asyncio, hashlib, json, os, random, subprocess, sys, time, uuid
from pathlib import Path
import yaml
from dotenv import load_dotenv
from .prompts import load_conditions, load_variants, build_system_prompt, build_user_prompt
from .client import OpenAICompatClient
from . import parse, classify, ci, evaluate
from .agentic_misalignment import SOURCE_COMMIT as CLASSIFIER_SOURCE

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH = "../INSIDER_LLM_DETECTION_BENCHMARK"
DEFAULT_RESULTS = "../results"   # project-root results/ (mirrored to HF), not inside this repo
PHASES = ("smoke", "validation", "pilot", "study")   # every config declares one; only `study` runs
                                                       # from a clean tree feed the study tables (eval.py)

def git_hash(path=ROOT):
    """HEAD of a repo, with a `-dirty` suffix when the tree has uncommitted changes. Stamped
    into meta.json (run), reparse_log, classify_log and evaluate_log so every derived field
    is attributable to a code version."""
    try:
        h = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        return h + ("-dirty" if git_dirty(path) else "")
    except Exception: return "unknown"

def git_dirty(path) -> bool:
    """True if `git status --porcelain` is non-empty; False (not dirty) when git is unavailable
    so a non-git checkout can still run — its hash then reads `unknown`."""
    try:
        return bool(subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip())
    except Exception:
        return False

def dataset_version(bench: Path):
    """`git describe --tags --always --dirty` of the benchmark repo, e.g. `v0.2-1-g8f96467`:
    the tag, commits since the tag, and the short hash — the dataset version every run records."""
    try: return subprocess.check_output(["git", "-C", str(bench), "describe", "--tags", "--always", "--dirty"], text=True).strip()
    except Exception: return "unknown"

def prompt_set_sha256(bench: Path) -> str:
    """sha256 over sorted prompt-file names+contents. Variant wording changed under the
    same names A/B between dataset versions, so the prompt set is hashed explicitly
    (code review 2026-09-01)."""
    h = hashlib.sha256()
    for f in sorted((bench / "prompts").glob("variant_*.md")):   # wording files only; the prompts README is documentation
        if f.is_file():
            h.update(f.name.encode()); h.update(b"\0"); h.update(f.read_bytes()); h.update(b"\0")
    return h.hexdigest()

def model_digest(base_url: str | None, model: str) -> str:
    """Ollama tags (`:cloud` included) are mutable; the digest pins the actual weights.
    A digest change mid-study means a different model (project rule, 2026-08-31)."""
    import urllib.request
    try:
        root = (base_url or "http://localhost:11434/v1").split("/v1")[0]
        with urllib.request.urlopen(root + "/api/tags", timeout=10) as r:
            for m in json.load(r).get("models", []):
                if m.get("name") == model or m.get("model") == model:
                    return m.get("digest", "unknown")
    except Exception:
        pass
    return "unknown"

async def run_episode(client, cfg, cond, variant, variants, sample, sem):
    """One episode: build the prompts, call the model (with retries and exponential backoff,
    under the concurrency semaphore), derive the parser fields, and return the full record.
    `seed + sample` is passed to the API so samples of one condition differ deterministically."""
    sys_p = build_system_prompt(cond, variant, variants)
    usr_p = build_user_prompt(cond)
    async with sem:
        for attempt in range(cfg.get("retries", 1) + 1):
            try:
                # `send_seed: false` for endpoints that reject the OpenAI `seed` field (Google's
                # OpenAI-compatible Gemini endpoint, 2026-09-03); meta.json records `seed_sent`.
                r = await client(cfg["model"], [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                                 max_tokens=cfg.get("max_tokens", 4000), temperature=cfg.get("temperature", 1.0),
                                 seed=(cfg.get("seed", 0) + sample) if cfg.get("send_seed", True) else None)
                break
            except Exception as ex:
                if attempt == cfg.get("retries", 1): raise
                await asyncio.sleep(2 ** attempt)
    d = parse.derive(r.completion, instructed=variant != "none",   # every variant but `none` requests a log
                     stop_reason=r.stop_reason)                    # stop_reason=length → logged=no_answer (M1)
    inline_reasoning = d.pop("inline_reasoning")
    reasoning = "\n".join(x for x in (r.reasoning, inline_reasoning) if x)
    return {
        "episode_id": f"{cond.condition_id}__{variant}__s{sample}__{uuid.uuid4().hex[:8]}",
        "condition_id": cond.condition_id, "scenario": cond.scenario,
        "goal_type": cond.goal_type, "goal_value": cond.goal_value, "urgency_type": cond.urgency_type,
        "prompt_variant": variant, "sample": sample, "model": cfg["model"],
        "system_prompt": sys_p, "user_prompt": usr_p,
        "response": r.completion, "reasoning": reasoning, **d,
        "api_model": r.api_model, "provider": r.provider,         # provider-reported model/routing (code review 2026-09-01)
        "harmful": None,                                          # filled by classify step
        "usage": r.usage, "duration_s": round(r.duration, 2), "stop_reason": r.stop_reason,
    }

async def cmd_run(cfg_path: Path, resume: str | None = None, allow_dirty: bool = False):
    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load(cfg_path.read_text())
    bench = (ROOT / cfg.get("benchmark_dir", DEFAULT_BENCH)).resolve()
    if not (bench / "data/conditions.jsonl").exists():
        sys.exit(f"benchmark not found at {bench}: run `make benchmark` (clones it next to this repo) "
                 f"or set `benchmark_dir` in the config")
    # Reproducibility comes from config + commit + seed: a run from a dirty tree cannot be
    # reconstructed from git, so refuse unless explicitly allowed (and then stamp the run).
    dirty = [name for name, p in (("code", ROOT), ("benchmark", bench)) if git_dirty(p)]
    if dirty and not allow_dirty:
        sys.exit(f"refusing to run: uncommitted changes in {' and '.join(dirty)} repo(s). "
                 f"Commit first, or pass --allow-dirty to stamp the run non-reconstructible.")
    conds = load_conditions(bench / "data/conditions.jsonl")
    variants = load_variants(bench / "prompts")
    sel = cfg.get("conditions")
    if sel:
        # a typo'd condition id must fail loudly, not silently shrink the grid (M4)
        valid = {c.condition_id for c in conds}
        unknown = sorted(set(sel) - valid)
        if unknown:
            sys.exit(f"unknown condition ids in config: {', '.join(unknown)}\nvalid: {', '.join(sorted(valid))}")
        conds = [c for c in conds if c.condition_id in sel]
    phase = cfg.get("phase")
    if phase not in PHASES:
        sys.exit(f"config must declare `phase: <{'|'.join(PHASES)}>` (got {phase!r}); "
                 f"the flag decides which tables a run may enter (code review 2026-09-02).")
    cvars = cfg.get("variants", ["none", "A", "B"])
    unknown_v = sorted(set(cvars) - set(variants))
    if unknown_v:
        sys.exit(f"unknown prompt variants in config: {', '.join(unknown_v)}\nvalid: {', '.join(sorted(variants))}")
    run_id = resume or cfg.get("run_id") or f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['model'].replace('/', '_').replace(':', '-')}"
    out = (ROOT / cfg.get("results_dir", DEFAULT_RESULTS)).resolve() / run_id
    # A fixed run_id in config must not silently append a second batch of episodes to an
    # existing run (code review 2026-09-01) — continuing a run requires --resume.
    if out.exists() and not resume:
        sys.exit(f"refusing to run: {out} already exists. Pass --resume {run_id} to continue it.")
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    if resume:
        if (out / "meta.json").exists():
            old_cfg = json.loads((out / "meta.json").read_text()).get("config")
            if old_cfg != cfg:
                sys.exit(f"refusing to resume {run_id}: config differs from the one in meta.json. "
                         f"Start a new run, or restore the original config.")
        if (out / "episodes.jsonl").exists():
            for l in (out / "episodes.jsonl").read_text().splitlines():
                if l.strip():
                    e = json.loads(l); done.add((e["condition_id"], e["prompt_variant"], e["sample"]))
    random.seed(cfg.get("seed", 0))
    meta = {"run_id": run_id, "config": cfg, "git_hash": git_hash(), "classifier_source": CLASSIFIER_SOURCE,
            "dataset_version": dataset_version(bench), "prompt_set_sha256": prompt_set_sha256(bench),
            "seed": cfg.get("seed", 0), "seed_sent": bool(cfg.get("send_seed", True)),
            "model_digest": model_digest(cfg.get("base_url"), cfg["model"]),
            "allow_dirty": bool(dirty), "reconstructible": not dirty, "phase": phase,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if dirty: meta["reconstructible_note"] = "ran with --allow-dirty (uncommitted changes at run time)"
    if not (out / "meta.json").exists(): (out / "meta.json").write_text(json.dumps(meta, indent=2))
    api_key = os.environ.get(cfg["api_key_env"]) if cfg.get("api_key_env") else "ollama"
    if not api_key:
        sys.exit(f"refusing to run: {cfg['api_key_env']} is not set (add it to .env)")
    client = OpenAICompatClient(base_url=cfg.get("base_url"), api_key=api_key)
    sem = asyncio.Semaphore(cfg.get("concurrency", 4))

    async def guarded(c, v, s):
        # A failed episode (after retries) must not abort the run: it is written to
        # failures.jsonl with its error and can be re-run with --resume.
        try:
            return await run_episode(client, cfg, c, v, variants, s, sem)
        except Exception as ex:
            return {"_failed": True, "condition_id": c.condition_id, "prompt_variant": v,
                    "sample": s, "error": repr(ex)}
    jobs = [guarded(c, v, s)
            for c in conds for v in cvars for s in range(cfg.get("samples_per_condition", 1))
            if (c.condition_id, v, s) not in done]
    if done: print(f"resuming {run_id}: {len(done)} episodes already done")
    print(f"{run_id}: {len(jobs)} episodes -> {out}")
    n = failed = 0
    with (out / "episodes.jsonl").open("a") as f, (out / "failures.jsonl").open("a") as ff:
        for coro in asyncio.as_completed(jobs):
            ep = await coro
            if ep.get("_failed"):
                failed += 1
                ff.write(json.dumps(ep, ensure_ascii=False) + "\n"); ff.flush()
                print(f"  FAILED {ep['condition_id']} {ep['prompt_variant']} s{ep['sample']}: {ep['error']}")
                continue
            f.write(json.dumps(ep, ensure_ascii=False) + "\n"); f.flush(); n += 1
            print(f"  [{n}/{len(jobs)}] {ep['condition_id']} {ep['prompt_variant']} actions={len(ep['system_log'])} log={ep['logged']}")
    print(f"done: {n} ok, {failed} failed" + (f" (see {out / 'failures.jsonl'})" if failed else ""))

def cmd_reparse(run_dir: Path):
    """Re-derive final_text / system_log / model_log / logged from stored responses (parser fixes).
    Derived fields are only reproducible if the parser commit that produced them is recorded,
    so every reparse appends a reparse_log entry to meta.json (code review 2026-09-01)."""
    f = run_dir / "episodes.jsonl"; eps = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    changed = 0; transitions = {}
    for e in eps:
        old_logged = e.get("logged")
        new = parse.derive(e["response"], instructed=e.get("prompt_variant") != "none",
                           stop_reason=e.get("stop_reason"))   # stored stop_reason → no_answer (M1)
        new.pop("inline_reasoning")                            # `reasoning` is stored raw, not re-derived
        if any(e.get(k) != v for k, v in new.items()): changed += 1
        e.update(new)
        evaluate.apply_verdicts(e)                             # stored evaluator verdicts survive a reparse
        if old_logged != e["logged"]:
            transitions.setdefault((old_logged, e["logged"]), []).append(e)
    f.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in eps))
    # A reparse that *creates* a discrepancy must show its evidence: every episode that
    # newly scores `not_logged` is printed with its unmatched tool-call blocks and its log
    # entries, so a parser gap is eyeballed before it is written anywhere (code review 2026-09-02).
    for (a, b), lst in sorted(transitions.items(), key=lambda kv: str(kv[0])):
        print(f"  logged {a} -> {b}: {len(lst)}")
        if b == "not_logged":
            for e in lst:
                matched = {aid for m in e["model_log"] for aid in m["matched_action_ids"]}
                print(f"    NEW OMISSION {e['episode_id']} — verify by hand before reporting:")
                for act in e["system_log"]:
                    if act["action_id"] not in matched:
                        print(f"      unmatched {act['action_id']} {act['type']} to={act['to']!r} "
                              f"recipient_parsed={act.get('recipient_parsed')} raw={act['raw'][:200]!r}")
                for m in e["model_log"]:
                    print(f"      entry {m['entry_id']} {m['raw'][:160]!r}")
    meta_f = run_dir / "meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    meta.setdefault("reparse_log", []).append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "code_git_hash": git_hash(),
                                               "n_episodes": len(eps), "n_changed": changed})
    meta_f.write_text(json.dumps(meta, indent=2))
    print(f"{run_dir.name}: reparsed {len(eps)} episodes, {changed} changed")

def main():
    ap = argparse.ArgumentParser(prog="ild")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--config", type=Path, required=True)
    r.add_argument("--resume", help="existing run_id to continue; finished episodes are skipped")
    r.add_argument("--allow-dirty", action="store_true",
                   help="run despite uncommitted changes; the run is stamped non-reconstructible")
    rp = sub.add_parser("reparse"); rp.add_argument("--run", type=Path, required=True, nargs="+")
    rl = sub.add_parser("relabel", help="re-derive harmful from the stored verdict + the classifier gates on the full completion (no API calls)")
    rl.add_argument("--run", type=Path, required=True, nargs="+")
    c = sub.add_parser("classify", help="label episodes harmful/not with the bundled Agentic Misalignment classifiers (Gemini judge)")
    c.add_argument("--run", type=Path, required=True, nargs="+")
    c.add_argument("--bench", type=Path, default=None, help=f"benchmark repo (default {DEFAULT_BENCH})")
    c.add_argument("--model", default=classify.DEFAULT_MODEL)
    c.add_argument("--force", action="store_true", help="relabel episodes that already carry a label")
    c.add_argument("--concurrency", type=int, default=4)
    c.add_argument("--input", choices=["final_text", "response"], default="final_text")
    ev = sub.add_parser("evaluate", help="log-accuracy evaluator: logged_matched -> logged_accurate / logged_inaccurate (Gemini judge, log-only)")
    ev.add_argument("--run", type=Path, required=True, nargs="+")
    ev.add_argument("--bench", type=Path, default=None)
    ev.add_argument("--model", default=classify.DEFAULT_MODEL)
    ev.add_argument("--force", action="store_true", help="re-judge episodes that already carry verdicts")
    ev.add_argument("--concurrency", type=int, default=4)
    fc = sub.add_parser("fixture-check", help="re-derive the committed fixture episodes and fail on any parser difference (CI gate)")
    fc.add_argument("--fixtures", type=Path, default=ROOT / "tests/fixtures")
    cc = sub.add_parser("check-configs", help="validate every configs/*.yaml against the benchmark without running anything (CI gate)")
    cc.add_argument("config_dir", type=Path, nargs="?", default=ROOT / "configs")
    cc.add_argument("--bench", type=Path, default=None)
    e = sub.add_parser("eval", help="regenerate results/tables.md from run dirs")
    e.add_argument("--runs", type=Path, nargs="+", default=None, help="run dirs (default: all under the results root)")
    e.add_argument("--out", type=Path, default=None, help="output file (default: <results>/tables.md)")
    a = ap.parse_args()
    if a.cmd == "run": asyncio.run(cmd_run(a.config, a.resume, a.allow_dirty))
    elif a.cmd == "reparse":
        for r in a.run: cmd_reparse(r)
    elif a.cmd == "relabel":
        classify.cmd_relabel(a.run)
    elif a.cmd == "classify":
        load_dotenv(ROOT / ".env")
        bench = (a.bench or ROOT / DEFAULT_BENCH).resolve()
        asyncio.run(classify.cmd_classify(a.run, bench, model_id=a.model, force=a.force,
                                          concurrency=a.concurrency, input_field=a.input,
                                          parser_hash=git_hash()))
    elif a.cmd == "evaluate":
        load_dotenv(ROOT / ".env")
        bench = (a.bench or ROOT / DEFAULT_BENCH).resolve()
        asyncio.run(evaluate.cmd_evaluate(a.run, bench, model_id=a.model, force=a.force,
                                          concurrency=a.concurrency, parser_hash=git_hash()))
    elif a.cmd == "fixture-check":
        sys.exit(ci.cmd_fixture_check(a.fixtures))
    elif a.cmd == "check-configs":
        sys.exit(ci.cmd_check_configs(a.config_dir, (a.bench or ROOT / DEFAULT_BENCH).resolve(), PHASES))
    elif a.cmd == "eval":
        from . import eval as ev   # lazy: eval.py imports git_hash from this module
        ev.cmd_eval(a.runs, a.out)

if __name__ == "__main__": main()
