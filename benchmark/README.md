# Bundled benchmark inputs

A copy of the files this code reads from the [benchmark repo](https://huggingface.co/datasets/logicBombExe/INSIDER_LLM_DETECTION_BENCHMARK), so a fresh clone runs on its own.

| File | Used by |
|---|---|
| `data/conditions.jsonl` | the 18 conditions (`ild run`, `ild classify`, `ild check-configs`) |
| `prompts/variant_{A,B,C,D}.md` | the logging instructions appended to the system prompt |
| `labels/log_accuracy_evaluator_prompt.md` | the log-accuracy judge (`ild evaluate`) |
| `LICENSE` | MIT notice for the Agentic Misalignment material |

`MANIFEST.json` records the source commit and a sha256 per file. Do not edit these files here: change the benchmark repo, then run `ild sync-benchmark` (or `make sync-benchmark`). `ild check-benchmark` fails if the copy differs from the manifest.
