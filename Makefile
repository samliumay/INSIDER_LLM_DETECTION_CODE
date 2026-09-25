.PHONY: data sync-benchmark smoke baseline eval test publish-results
RESULTS_REPO ?= logicBombExe/INSIDER_LLM_DETECTION_RESULTS
CONFIG ?= configs/smoke.yaml

data:            ## verify the bundled benchmark copy (benchmark/) against its manifest
	uv run ild check-benchmark

sync-benchmark:  ## refresh benchmark/ from a clean benchmark checkout (default ../INSIDER_LLM_DETECTION_BENCHMARK)
	uv run ild sync-benchmark $(if $(FROM),--from $(FROM),)

smoke:           ## run the smoke-test config end to end
	uv run ild run --config $(CONFIG)

baseline:        ## model-log-only and random detectors on a results dir — NOT YET IMPLEMENTED
	@echo "ild baseline is not implemented yet : model-log-only and random detectors" && exit 1

eval:            ## regenerate ../results/tables.md and INDEX.md from all runs
	uv run ild eval

publish-results: ## mirror ../results/ to the HF results dataset repo (hf CLI must be logged in)
	hf upload $(RESULTS_REPO) ../results . --repo-type dataset --commit-message "results: $$(date -u +%Y-%m-%dT%H:%M:%SZ)"

test:
	uv run python -m pytest tests/ -q
