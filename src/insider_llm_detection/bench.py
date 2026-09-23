"""Bundled benchmark copy: the benchmark files the code reads, kept inside this repo.

`benchmark/` holds a copy of the benchmark repo's run inputs (conditions, variant prompts,
log-accuracy evaluator prompt) so a fresh clone of the code runs on its own. `MANIFEST.json`
records the source revision and a sha256 per file; `check` fails on any drift, `sync`
refreshes the copy from a benchmark checkout.
"""
import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH = ROOT / "benchmark"
MANIFEST = "MANIFEST.json"
# Every benchmark file the code reads (prompts.py, run.py, classify.py, evaluate.py, ci.py), plus the license.
FILES = ("data/conditions.jsonl",
         "prompts/variant_A.md", "prompts/variant_B.md", "prompts/variant_C.md", "prompts/variant_D.md",
         "labels/log_accuracy_evaluator_prompt.md",
         "LICENSE")


def _sha(p: Path) -> str:
    """sha256 of one file."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def manifest(bench: Path) -> dict | None:
    """The copy's MANIFEST.json, or None for an external benchmark checkout."""
    f = bench / MANIFEST
    return json.loads(f.read_text()) if f.exists() else None


def version(bench: Path) -> str:
    """Benchmark revision stamped into meta.json as `dataset_version`: the manifest's
    `version` for the bundled copy, `git describe` for an external checkout."""
    m = manifest(bench)
    if m:
        return m["version"]
    try:
        return subprocess.check_output(["git", "-C", str(bench), "describe", "--tags", "--always", "--dirty"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def check(bench: Path = DEFAULT_BENCH) -> list[str]:
    """Compare the copy with its manifest. Returns the problems (empty = identical)."""
    m = manifest(bench)
    if not m:
        return [f"{bench / MANIFEST} missing"]
    problems = []
    for rel, sha in m["files"].items():
        p = bench / rel
        if not p.exists():
            problems.append(f"{rel}: missing")
        elif _sha(p) != sha:
            problems.append(f"{rel}: changed since sync (edit the benchmark repo, then `ild sync-benchmark`)")
    return problems


def cmd_check(bench: Path = DEFAULT_BENCH) -> int:
    """`ild check-benchmark`: print the result of `check`; exit code 1 on drift."""
    problems = check(bench)
    for p in problems:
        print(f"  FAIL {p}")
    m = manifest(bench) or {}
    print(f"check-benchmark: {len(m.get('files', {}))} files, version {m.get('version', '?')}, {len(problems)} problems")
    return 1 if problems else 0


def cmd_sync(source: Path, dest: Path = DEFAULT_BENCH) -> int:
    """`ild sync-benchmark --from <checkout>`: copy FILES from a clean benchmark checkout
    into `dest` and rewrite the manifest. Refuses uncommitted changes in those files."""
    source = source.resolve()
    dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--", *FILES],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit(f"refusing to sync: uncommitted changes in the benchmark files:\n{dirty}")
    files = {}
    for rel in FILES:
        src, dst = source / rel, dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files[rel] = _sha(dst)
    git = lambda *a: subprocess.check_output(["git", "-C", str(source), *a], text=True).strip()
    m = {"source": git("remote", "get-url", "origin"), "commit": git("rev-parse", "HEAD"),
         "version": git("describe", "--tags", "--always"), "synced_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "files": files}
    (dest / MANIFEST).write_text(json.dumps(m, indent=2) + "\n")
    print(f"synced {len(files)} files from {source} at {m['version']} into {dest}")
    return 0
