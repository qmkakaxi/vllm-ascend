# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM Ascend is a **hardware plugin** for running vLLM on Ascend NPU (Huawei). It registers via vLLM's pluggable platform interface (`vllm.platform_plugins`) and does not fork vLLM itself. It supports A2/A3/A5/310P hardware.

## Build & Install

```bash
# Full build (C++ extensions + Python package, requires NPU toolchain)
pip install -e ./

# CPU-only / no-NPU environment (skip C++ kernel compilation)
COMPILE_CUSTOM_KERNELS=0 pip install -e ./

# Build only the C++ extensions (for dev iteration)
bash csrc/build.sh
```

Build requirements: Python 3.10-3.11, PyTorch 2.9.0, torch-npu 2.9.0, CANN 8.5.1.

## Testing

```bash
# Run all unit tests (no NPU required for most UT)
pytest -sv tests/ut/

# Run a specific unit test file
pytest -sv tests/ut/ops/test_prepare_finalize.py

# Run a specific test case
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs

# Run e2e tests (requires NPU hardware)
pytest -sv tests/e2e/singlecard/test_piecewise_res_consistency.py
```

- Unit tests: `tests/ut/`
- System/e2e tests: `tests/e2e/` (require NPU hardware)

## Lint & Format

```bash
# Python lint + format
ruff check vllm_ascend/
ruff format vllm_ascend/

# Full pre-commit check (all file types, run before pushing)
bash format.sh ci

# Local pre-commit (auto-fix)
bash format.sh
```

The lint stack includes ruff, clang-format, codespell, typos, shellcheck, markdownlint, and project-specific custom checks.

## Architecture

### Two-Layer Plugin Architecture

The package `vllm_ascend/` is a **hardware plugin** loaded by vLLM at startup. It has two layers:

1. **Platform layer** (`vllm_ascend/platform.py`): `NPUPlatform` class registered via the `vllm.platform_plugins` entry point. Handles device detection, memory, scheduling, and dispatches to NPU-specific components.

2. **Patch layer** (`vllm_ascend/patch/`): Monkey-patches upstream vLLM modules to override behavior for NPU.
   - `patch/platform/` — Scheduler, KV cache, distributed, profiling patches
   - `patch/worker/` — Model-specific patches (DeepSeek, Qwen3, Minimax, etc.)
   - Patches are applied via `vllm_ascend.utils.adapt_patch()` at plugin load time

### Key Directories

| Directory | Purpose |
|---|---|
| `vllm_ascend/ops/` | NPU custom ops in Python (RMSNorm, MLA, RoPE, fused MoE, etc.) |
| `vllm_ascend/worker/` | Model runner (`model_runner_v1.py`) and `worker.py` |
| `vllm_ascend/worker/v2/` | vLLM v2 model runner |
| `vllm_ascend/_310p/` | Ascend 310P-specific overrides (separate model runner, ops, worker) |
| `vllm_ascend/distributed/` | HCCL communicator, KV transfer, parallel state |
| `vllm_ascend/quantization/` | NPU quant methods (W8A8, W4A8, etc.) |
| `vllm_ascend/attention/` | Attention backend configurations |
| `vllm_ascend/lora/` | LoRA adapter support for NPU |
| `vllm_ascend/eplb/` | Expert-parallel load balancing |
| `vllm_ascend/envs.py` | **All** environment variables must be defined here via `env_variables` dict |
| `csrc/` | C++ extensions (kernels, attention, MoE dispatch, MC2, torch bindings) |
| `vllm_ascend/_cann_ops_custom/` | Custom CANN (Ascend Compute Architecture) operators |

### C++ Extension Structure (`csrc/`)

The C++ code compiles into two targets:
- `vllm_ascend_C` — pybind11 module with torch binding (`csrc/torch_binding.cpp`) + device-side utilities
- `vllm_ascend_kernels` — AscendC kernels (compiled KERNEL_FILES in `csrc/kernels/` + attention/mla/moe operators)

Key C++ subdirectories:
- `csrc/kernels/` — BGMV/SGMV expand/shrink kernels for quantization
- `csrc/attention/` — Lightning indexer, sparse flash attention
- `csrc/mc2/` — Dispatch/combine kernels for MoE expert parallelism
- `csrc/moe/`, `csrc/gmm/` — MoE fused ops
- `csrc/batch_matmul_transpose/`, `csrc/mla_preprocess/` — MLA (multi-head latent attention) kernels
- `csrc/aclnn_torch_adapter/` — ACLNN-to-PyTorch bridging

### Adding a New Model

1. Create a patch file in `vllm_ascend/patch/worker/` (e.g., `patch_new_model.py`)
2. Monkey-patch the upstream model's `forward` or other methods
3. Register any new custom ops in `vllm_ascend/ops/` if needed
4. Do **not** add new model files directly — use patching

### Platform Detection

Heterogeneous chip support (A2/A3/A5/310P) is handled via:
- `SOC_VERSION` env var or auto-detection via `npu-smi info` at build time
- `vllm_ascend/platform.py` maps SOC versions to `AscendDeviceType` enum
- 310P has its own sub-directory `_310p/` with overrides
- Compile-time: `CMakeLists.txt` conditionally enables/disables kernels per chip

## Commit Convention

- Follow [Conventional Commits](https://www.conventionalcommits.org/) with valid types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`
- **All commits must be signed off**: `git commit -s`
- PR titles: `[Type][Module] Description` (e.g., `[BugFix] Fix CPU binding logic`)
