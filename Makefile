.PHONY: benchmark data validate smoke baseline eval test ci publish-results
RESULTS_REPO ?= logicBombExe/INSIDER_LLM_DETECTION_RESULTS
BENCHMARK_REPO ?= https://huggingface.co/datasets/logicBombExe/INSIDER_LLM_DETECTION_BENCHMARK
BENCHMARK_TAG ?= $(shell uv run python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['tool']['ild']['benchmark_tag'])")
CONFIG ?= configs/smoke.yaml

benchmark:       ## clone the benchmark at the declared tag next to this repo (skipped if present)
	@test -d ../INSIDER_LLM_DETECTION_BENCHMARK && echo "../INSIDER_LLM_DETECTION_BENCHMARK already present" || \
	  git clone --depth 1 --branch "$(BENCHMARK_TAG)" $(BENCHMARK_REPO) ../INSIDER_LLM_DETECTION_BENCHMARK

data:            ## rebuild the benchmark's conditions from its bundled templates
	uv run python ../INSIDER_LLM_DETECTION_BENCHMARK/build_conditions.py

validate:        ## run the benchmark's validation gate
	cd ../INSIDER_LLM_DETECTION_BENCHMARK && python3 validate.py

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

ci: validate       ## everything .github/workflows/ci.yml runs, locally
	uv sync --frozen --group dev
	uv run pytest -q
	uv run ild fixture-check --fixtures tests/fixtures
	uv run ild check-configs configs/
	@! grep -nE "reportable[^_]|zero .not_logged. episodes exist|two (true )?omissions|observed twice|BEFORE_RESEARCH|decisions\.md|evaluation_plan|findings\.md|possible_improvements|power_analysis|docs/audits|internal (research )?workspace|standalone repository" README.md EXPERIMENTS.md || (echo "retired vocabulary found" && exit 1)
