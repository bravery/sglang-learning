# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a **personal learning repository** for studying SGLang 0.5.7 source code. It contains:
- The complete SGLang 0.5.7 source code (in `sglang-0.5.7/`)
- Structured learning notes organized by module (in `notes/`)
- Learning progress tracking and problem logs

**This is NOT the official SGLang repository.** The official repo is at https://github.com/sgl-project/sglang

## Working with Learning Notes

### Note Structure
Each note file in `notes/` follows this template:
- **Overview section** - Module purpose and source location
- **Core concepts** - Key ideas and data structures
- **Code analysis placeholders** - To be filled during study
- **Performance considerations**
- **Personal learning journal** - Daily notes and discoveries

### Updating Notes
When helping update learning notes:
1. Fill in code analysis sections with specific file paths and line numbers from `sglang-0.5.7/`
2. Add concrete examples from the source code
3. Update the learning journal sections with dates
4. Cross-reference between related modules
5. Add diagrams using Mermaid syntax when helpful

### Progress Tracking
- Update `LEARNING_LOG.md` when completing study sessions
- Add questions to `QUESTIONS.md` when encountering unclear concepts
- Update the progress table in `README.md` when completing modules

## SGLang Source Code Structure

The actual SGLang code is in `sglang-0.5.7/python/sglang/`:

```
sglang/
├── srt/                      # SGLang Runtime - Core system
│   ├── managers/             # Request scheduling, batching (35 files, scheduler split across multiple mixins)
│   ├── model_executor/       # Model inference execution
│   ├── mem_cache/            # RadixAttention KV cache (20 files)
│   ├── constrained/          # Structured output with FSM
│   ├── entrypoints/          # HTTP/gRPC servers
│   │   ├── http_server.py    # FastAPI-based HTTP server (62KB)
│   │   ├── grpc_server.py    # gRPC server (40KB)
│   │   └── engine.py         # Core engine (39KB)
│   ├── sampling/             # Sampling strategies
│   ├── distributed/          # Distributed inference (TP/PP/EP/DP)
│   ├── hardware_backend/     # Multi-hardware support
│   ├── layers/               # Model layer implementations
│   ├── tokenizer/            # Tokenization
│   ├── lora/                 # LoRA adapter support
│   └── speculative/          # Speculative decoding
├── lang/                     # High-level language API
├── jit_kernel/               # JIT-compiled kernels
└── multimodal_gen/           # Vision-language models

sgl-kernel/                   # C++/CUDA kernels (separate package)
sgl-model-gateway/            # Model metadata service
```

### Entry Points
- **Server launch**: `python/sglang/launch_server.py` - Routes to HTTP/gRPC/encoder-only modes
- **HTTP mode**: `srt/entrypoints/http_server.py`
- **gRPC mode**: `srt/entrypoints/grpc_server.py`
- **Encoder-only**: `srt/disaggregation/encode_server.py`

### Architecture Pattern
```
Client Request
  → Entrypoint (HTTP/gRPC)
    → Manager (Scheduler with mixins for profiling/metrics/output processing)
      → Model Executor (forward pass)
        → Memory Cache (RadixAttention prefix sharing)
          → Sampling
            → Response
```

### Key Design Patterns

**Mixin Pattern for Scheduler**: Complex scheduler functionality is split across multiple mixin files:
- `scheduler_profiler_mixin.py`
- `scheduler_metrics_mixin.py`
- `scheduler_output_processor_mixin.py`
- `scheduler_dp_attn_mixin.py`
- `scheduler_update_weights_mixin.py`

**RadixAttention Innovation**: Uses radix tree data structure for automatic KV cache prefix sharing across requests, enabling 3-5x faster inference.

**Constrained Generation**: Multiple backend support (xgrammar, llguidance, outlines) with FSM-based token filtering for structured outputs.

## Development Commands

### Setting Up Development Environment

From the SGLang source directory:
```bash
cd sglang-0.5.7/python
pip install -e ".[dev]"
```

### Running Tests

Tests use pytest with async support enabled:
```bash
# Run all tests
cd sglang-0.5.7
pytest test/

# Run specific test file
pytest test/srt/test_specific.py

# Run tests with specific markers
pytest -v test/

# Configuration is in test/pytest.ini with asyncio_mode=auto
```

### Running Examples

```bash
cd sglang-0.5.7

# Run basic examples
python examples/quick_start.py

# Launch server (requires model)
python python/sglang/launch_server.py --model-path <model>

# HTTP mode (default)
python python/sglang/launch_server.py --model-path <model>

# gRPC mode
python python/sglang/launch_server.py --model-path <model> --grpc-mode

# Encoder-only mode
python python/sglang/launch_server.py --model-path <model> --encoder-only
```

### Debugging

Use Python debugger to trace execution:
```bash
cd sglang-0.5.7
python -m pdb python/sglang/launch_server.py --model-path <model>
```

### Build System

- **Python package**: Uses setuptools with setuptools-scm for versioning
- **C++/CUDA kernels**: CMake-based build in `sgl-kernel/`
- **Dependencies**: See `python/pyproject.toml` for full list
  - PyTorch 2.9.1, Transformers 4.57.1
  - flashinfer 0.5.3, xgrammar 0.1.27
  - FastAPI, gRPC, Triton

## Key Concepts to Understand

### RadixAttention (Core Innovation)
- Prefix caching using radix tree structure
- Automatic sharing of common prefixes across requests
- Located in `srt/mem_cache/radix_cache.py`
- Enables 3-5x inference speedup

### Continuous Batching
- Dynamic request entry/exit during generation
- Zero-overhead scheduling with batch overlap
- See `srt/batch_overlap/` and batch-related scheduler mixins

### Structured Output
- JSON schema validation
- Regular expression constraints
- Grammar-based decoding
- Located in `srt/constrained/`

### Distributed Inference
- Tensor Parallelism (TP), Pipeline Parallelism (PP)
- Expert Parallelism (EP) for MoE models
- Data Parallelism (DP)
- Located in `srt/distributed/`

## Recommended Learning Path

When helping someone learn SGLang, follow the progression outlined in `QUICK_START.md`:

### Week 1: Basics
1. Project overview and architecture (`notes/00-*.md`, `notes/01-*.md`)
2. Entry points - trace from `launch_server.py` through HTTP/gRPC servers
3. Basic request flow

### Week 2: Core Execution
1. Scheduler system in `srt/managers/` (note the mixin pattern)
2. Model executor in `srt/model_executor/`
3. Sampling strategies in `srt/sampling/`

### Week 3: Key Features
1. **RadixAttention** in `srt/mem_cache/` - This is critical to understand
2. Constrained generation in `srt/constrained/`
3. Batch optimization

### Week 4+: Advanced
1. Distributed systems
2. Multi-modal support
3. JIT kernels
4. Language API

## Important Files to Reference

When explaining specific features, point to these key files:

**Server Configuration**: `srt/server_args.py` (5,182 lines - comprehensive config)

**Scheduling**: Look for `scheduler*.py` files in managers, note mixin composition

**RadixAttention**: `srt/mem_cache/radix_cache.py`, `srt/mem_cache/memory_pool.py`

**Constrained Generation**: `srt/constrained/fsm_cache.py`, `srt/constrained/jump_forward.py`

**Model Execution**: `srt/model_executor/model_runner.py`, `srt/model_executor/forward_batch_info.py`

## Comparative Analysis

When discussing SGLang vs other frameworks (vLLM, TGI), key differentiators:
- **RadixAttention** vs vLLM's PagedAttention (prefix sharing vs paging)
- **Zero-overhead scheduling** with batch overlap
- **Advanced structured output** with multiple FSM backends
- **Production deployment** at scale (400,000+ GPUs, trillions of tokens daily)

Use `QUESTIONS.md` to track framework comparisons and design trade-offs.

## Multi-Hardware Support

SGLang runs on diverse hardware:
- NVIDIA GPUs (GB200, B300, H100, A100)
- AMD GPUs (MI355, MI300)
- Intel Xeon CPUs
- Google TPUs (via SGLang-Jax backend)
- Ascend NPUs

Backend implementations are in `srt/hardware_backend/`.
