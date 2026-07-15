# AutoRed-Final — Agent Notes

AutoRed is a research codebase for prompt-injection / access-code-extraction experiments against LLM defenses. It couples a planner LLM, a generator LLM, a DistilBERT stop-point judge, and a multi-layer extractor/verifier.

## Canonical Sources of Truth

- **Live system state:** `docs/current_implementation.md`
- **Architecture / phase plan:** `docs/current_implementation_plan.md`
- **Runtime entry point:** `experiment/llama_3_8b_vllm.py`
- **Backend API:** `server/main.py` (FastAPI + WebSocket)
- **Frontend:** `ui/` (Vite + React + Tailwind)
- **Schema stubs:** `schemas/`
- **Run artifacts:** `results/` and `results/benchmarks/<id>/merged_summary.json`

There is no root README, no `pyproject.toml`, and no test runner config. `requirements.txt` is the authoritative dependency list; `uv.lock` exists but `scripts/training/README.md` references a non-existent `requirements_qlo.txt`.

## Environment Setup

```bash
# Python env (CUDA 12.4 PyTorch is pinned)
pip install -r requirements.txt

# Or with uv
uv pip install -r requirements.txt

# Frontend env
cd ui
npm install
```

Key environmental quirks:

- **Always set `VLLM_USE_V1=0`** before running the runtime. The code expects the vLLM V0 engine; V1 triggers `torch.compile` and extra GPU memory use that usually OOMs.
- **CUDA is required.** The runtime loads `meta-llama/Meta-Llama-3-8B-Instruct` and `Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2` through vLLM.
- The victim/target LLM defaults to `meta-llama/Meta-Llama-3-8B-Instruct`. Override with `--victim-model-id <hf-model-id>` (runtime CLI) or `AUTORED_VICTIM_MODEL_ID` (server / worker processes). Models that ship custom Python code (e.g. `internlm/internlm2-chat-7b`) also require `--trust-remote-code` (or `AUTORED_TRUST_REMOTE_CODE=1`). For newer Mistral checkpoints that ship the Mistral-format tokenizer files, set `--tokenizer-mode mistral` (or `AUTORED_TOKENIZER_MODE=mistral`); otherwise leave it as `auto`.
- **GPU-heavy work belongs on the HPC cluster.** Single experiments, benchmarks, extractor benchmarks, and any command that loads vLLM / CUDA models are meant to run on the cluster. Do not run them on a local workstation. Local machines should only be used for model-free workflows (UI development, backend browsing with `AUTORED_LOAD_MODELS=0`, or parsing/merging scripts).
- For offline/air-gapped HPC runs: set `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`.
- If vLLM fails during memory profiling with `AssertionError: Error in memory profiling`, try `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or `export AUTORED_SKIP_VLLM_MEMORY_PROFILE=1` before launching. The latter disables the memory-increase assertion inside vLLM's worker.
- If the DistilBERT judge / access-code predictor OOMs after the victim LLM loads, lower vLLM's GPU memory fraction with `--gpu-memory-utilization 0.45` (or `AUTORED_GPU_MEMORY_UTILIZATION=0.45`). The default is `0.50`.

## Running the System

### Single experiment

```bash
VLLM_USE_V1=0 python experiment/llama_3_8b_vllm.py \
  --mode experiment \
  --rounds 20 \
  --dataset-size 1000 \
  --planner-path experiment/results/planner_sft_v2 \
  --generator-path experiment/results/generator_sft_v2 \
  --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2
```

### Benchmark

```bash
VLLM_USE_V1=0 python experiment/llama_3_8b_vllm.py \
  --mode benchmark \
  --rounds 70 \
  --dataset-size 1000 \
  --planner-path experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
  --generator-path experiment/results/generator_sft_v2 \
  --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
  --dataset-path data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl \
  --benchmark-output results/benchmarks/my_run/worker_0.json

# Merge multi-worker results
python scripts/merge_benchmarks.py \
  --output results/benchmarks/my_run/merged_summary.json \
  --worker-results results/benchmarks/my_run/worker_*.json
```

The 4-GPU batched benchmark is orchestrated by `hpc/autored_benchmark_4gpu_vllm.sh` (note the hardcoded `PROJECT_ROOT=/nlsasfs/home/isea/isea38/AutoRed-Final`; change it for your cluster).

To benchmark a deterministic slice of the loaded dataset instead of a random sample, add `--start-idx N` (0-based, inclusive). For example, `--start-idx 1000 --rounds 1000` processes indices 1000-1999. If `--start-idx` is omitted, the benchmark falls back to random sampling as before.

The HPC wrapper `hpc/autored_benchmark_4gpu_vllm.sh` now takes named options for every parameter:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./hpc/autored_benchmark_4gpu_vllm.sh \
  --rounds 1000 \
  --planner-path experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
  --generator-path experiment/results/generator_sft_v2 \
  --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
  --dataset-path data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl \
  --dataset-size 1000 \
  --output-dir results/benchmarks/Mistral_subset2_$(date +%F_%H-%M-%S)_4g \
  --victim-model-id internlm/internlm2-chat-7b \
  --trust-remote-code \
  --start-idx 0 \
  --attempts 20
```

If `--output-dir` is omitted, it defaults to `results/benchmarks/batched_${NUM_ROUNDS}r_4gpu`. `--attempts` (alias `--max-attempts`) controls the per-scenario attempt limit and defaults to 20. Use `--trust-remote-code` for models such as `internlm/internlm2-chat-7b` that ship custom Python files. Use `--tokenizer-mode mistral` only for newer Mistral checkpoints that ship Mistral-format tokenizer files; otherwise leave it as `auto`.

### Auto-update KB / DB / RAG

The runtime can automatically keep the knowledge stores fresh after each run or benchmark:

```bash
# Default: append per-run records and rebuild aggregate indices after a benchmark
VLLM_USE_V1=0 AUTORED_UPDATE_KB=all python experiment/llama_3_8b_vllm.py --mode benchmark ...

# Only cheap per-run appends (success/failure JSONL + SQLite trajectory DB)
VLLM_USE_V1=0 AUTORED_UPDATE_KB=run python experiment/llama_3_8b_vllm.py --mode benchmark ...

# Disable
VLLM_USE_V1=0 AUTORED_UPDATE_KB=off python experiment/llama_3_8b_vllm.py --mode benchmark ...
```

`--update-kb` accepts `off | run | benchmark | all` and overrides the env var. The implementation lives in `experiment/kb_updater.py`:

- **Per-run append** writes attempt-level records to `data/autored_successes_v1.jsonl`, `data/autored_failures_v1.jsonl`, and `data/autored_kb.db`.
- **Benchmark rebuild** regenerates `data/strategy_knowledge_base.json`, `data/oracle_rules.json`, and the FAISS RAG index (`data/rag/success_defenses.index` + `success_metadata.json`) from accumulated records.
- **Multi-worker caveat:** the expensive rebuild is skipped when `--num-workers > 1` to avoid concurrent writes; worker per-run appends still run. Rebuild manually after `scripts/merge_benchmarks.py`.

### Backend + UI

```bash
# Server (loads models on startup unless disabled)
AUTORED_SERVER_MODE=1 python -m uvicorn server.main:app

# Or skip model loading and just browse runs
AUTORED_SERVER_MODE=1 AUTORED_LOAD_MODELS=0 python -m uvicorn server.main:app

# Frontend dev server
cd ui
npm install
npm run dev
```

Server endpoints include `/api/runs`, `/api/benchmarks/{id}`, `/api/run/{run_id}`, WebSocket on `/ws/run/{run_id}`, and CSV/HTML exports on `/api/export/{run_id}/csv` and `/api/export/{run_id}/html`.

### Local viewing without GPU

For a laptop/local workstation that only needs to browse existing runs (no model inference):

```bash
pip install -r requirements_local.txt

cd ui && npm install

# In one terminal: start the backend without loading models
AUTORED_LOAD_MODELS=0 python -m uvicorn server.main:app --host 127.0.0.1 --port 8000

# In another terminal: start the Vite dev server
cd ui
npm run dev
```

`requirements_local.txt` excludes `torch`, `vLLM`, `transformers`, `sentence-transformers`, CUDA wheels, and the rest of the HPC training stack; it only installs FastAPI, uvicorn, and their runtime dependencies.

## Important Runtime Quirks

- **`experiment/llama_3_8b_vllm.py` loads a defense dataset at module import unless `AUTORED_SERVER_MODE=1` is set.** `server/main.py` sets this env var before importing anything from `experiment/`.
- The planner and generator are separate LoRA adapters loaded onto the same vLLM base model when an `adapter_config.json` is present; otherwise each path is loaded as a standalone vLLM instance.
- Hardcoded default paths in `experiment/llama_3_8b_vllm.py` include `pre_trained/pi_reward_model` (judge), `experiment/access_code_predictor`, `experiment/results/planner_sft_v2`, and `experiment/results/generator_sft_v2`.
- `--update-kb` / `AUTORED_UPDATE_KB` controls the new post-run KB/DB/RAG updater. Default is `all` (per-run append + benchmark rebuild), but it skips the expensive rebuild in multi-worker mode to avoid races.
- The judge is a **stop-point classifier** (`ATTACK` vs `ATTEMPT`), not a success verifier. Final success is decided by extraction + verification against the victim model.
- vLLM 0.8.5 may **silently ignore a PEFT LoRA adapter** even when `lora_request` is supplied. If planner outputs in the benchmark are prompt echoes or free-text plans instead of the trained XML, pre-merge the adapter into a full model:
  ```bash
  python scripts/merge_adapter_to_full.py \
    --base-model Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
    --adapter experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
    --output-dir experiment/results/planner_sft_v2_contract_anchor/checkpoint-27_merged
  ```
  The runtime automatically uses `<adapter_path>_merged` if it exists. Verify LoRA behavior with `scripts/tests/test_vllm_planner_lora.py`.
- To merge **both** planner and generator into one combined full model (recommended for benchmarks, avoids vLLM LoRA entirely):
  ```bash
  python scripts/merge_adapter_to_full.py \
    --base-model Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
    --adapter experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
    --adapter experiment/results/generator_sft_v2 \
    --output-dir experiment/results/planner_generator_combined_merged
  ```
  Test with `scripts/tests/test_combined_model.py`. Then run the benchmark with both `--planner-path` and `--generator-path` pointing to the combined merged directory.
- The shared planner/generator vLLM instance uses `max_model_len=2048`, `enable_prefix_caching=True`, and `gpu_memory_utilization` controlled by `AUTORED_SHARED_GPU_MEMORY_UTILIZATION` (default 0.55). The victim default in the HPC wrapper is now 0.40 (`--gpu-memory-utilization`).

## Training / Dataset Pipeline

- Primary trainer: `scripts/training/train_qlo.py` (QLoRA SFT).
- Dataset builders live under `scripts/dataset_tools/`:
  - `build_planner_sft_v2.py` → `data/planner_sft_dataset_v2.jsonl`
  - `build_generator_sft_v2.py` → `data/generator_sft_dataset_v2.jsonl`
- SLURM templates are in `hpc/` (e.g., `train_planner_sft.slurm`, `train_generator_sft.slurm`).

## Verification / Isolation Tests

There is no pytest suite. The project uses isolation smoke tests:

```bash
python scripts/tests/test_planner_v2.py
python scripts/tests/test_generator_v2.py
python scripts/tests/test_kb_updater.py
```

These only validate that the adapters emit correctly shaped planner XML / clean generator text; they do not exercise the full pipeline.

## Large Local Artifacts

Model weights and datasets are gitignored. Expected local pieces are listed in `largefiles.txt`. Notable ones:

- `models/defense_classifier/` and `models/ranker_deberta_v1/`
- `pre_trained/pi_reward_model/`
- `experiment/access_code_predictor/`
- `experiment/results/planner_sft_v2*/`
- `experiment/results/generator_sft_v2/`
- `data/*--largeFile.jsonl`
- `data/TensorTrust_subsets/*.jsonl`
- `data/rag/success_defenses.index` + `success_metadata.json`

## Workflow Conventions

- Use `docs/current_implementation.md` for the single source of truth on which checkpoint is "current" for planner, generator, judge, and access-code predictor.
- Run artifacts are locked to the current `git_commit` hash captured at runtime.
- Benchmark summaries are separate from per-run trace archives; both must be present for full UI analysis.
- Do not modify `experiment/llama_3_8b_vllm.py` default paths lightly — many HPC scripts and the server assume those defaults.
