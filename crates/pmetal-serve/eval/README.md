# pmetal-serve eval harness

Evaluation kit for validating pmetal as an OpenAI-compatible drop-in replacement
for `mlx-vlm` / `vllm-mlx` and measuring its performance advantage on Apple Silicon.

## Prerequisites

```bash
# MacPorts Python deps (MacPorts is the system package manager on this machine)
sudo port install py314-requests py314-openai py314-psutil

# OR for a quick one-off experiment:
# /opt/local/bin/pip-3.14 install openai requests psutil
```

---

## 1. Conformance — `run_conformance.sh`

Boots pmetal, then runs the vllm-mlx OpenAI compatibility test suite against it
(text + streaming only — vision deferred).

```bash
# Build release first (once)
cargo build -p pmetal --features serve --release

# Run conformance (text-only subset of vllm-mlx's own test script)
bash crates/pmetal-serve/eval/run_conformance.sh \
    --model mlx-community/Qwen3-4B-Instruct-4bit

# Custom port / binary
bash crates/pmetal-serve/eval/run_conformance.sh \
    --model mlx-community/Qwen3-4B-Instruct-4bit \
    --port 8080 \
    --pmetal-bin ./target/release/pmetal
```

The script sets `--no-image --no-video` on the vllm-mlx test runner, so only the
text / streaming / completions / models / health sections run.  All should pass.

Tests exercised:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (HTTP + OpenAI client)
- `POST /v1/completions` (legacy)
- `POST /v1/chat/completions` with `stream: true`

---

## 2. Behavioral parity — `promptfoo.yaml`

Compares pmetal vs. vllm-mlx output quality on a set of greedy (temperature=0,
seed=42) text prompts using [promptfoo](https://promptfoo.dev).

```bash
# Install promptfoo
npm install -g promptfoo

# Start pmetal on 8000
./target/release/pmetal serve --model mlx-community/Qwen3-4B-Instruct-4bit --port 8000 &

# Start vllm-mlx on 8001
vllm-mlx serve mlx-community/Qwen3-4B-Instruct-4bit --port 8001 &

# Run parity eval
promptfoo eval --config crates/pmetal-serve/eval/promptfoo.yaml

# View results in browser
promptfoo view
```

---

## 3. Performance benchmark — `bench.py`

Measures TTFT (time-to-first-token), decode throughput, and peak RSS across
multiple concurrency levels.

```bash
# pmetal only (single server)
/opt/local/bin/python3.14 crates/pmetal-serve/eval/bench.py \
    --pmetal-url http://localhost:8000

# Head-to-head vs. vllm-mlx (start both servers first)
/opt/local/bin/python3.14 crates/pmetal-serve/eval/bench.py \
    --pmetal-url    http://localhost:8000 \
    --baseline-url  http://localhost:8001 \
    --baseline-name vllm-mlx \
    --concurrency 1 4 8 \
    --max-tokens 256 \
    --prompts 20
```

Output:
- p50 / p95 TTFT and total latency per concurrency level
- Mean decode tok/s
- Peak RSS (requires `py314-psutil`)
- Side-by-side speedup table when a baseline is provided

---

## Interpretation guide

| Metric    | What it tells you |
|-----------|-------------------|
| **TTFT**  | First-token latency — dominated by prompt processing (prefill). Lower is better. |
| **Decode tok/s** | Autoregressive generation speed. Higher is better. Rust server overhead shows here vs. Python GIL. |
| **RSS**   | Peak resident memory. Should be lower for pmetal (no Python runtime). |
| **Conformance** | Whether the API wire format is compatible enough for the test client. |

---

## Future options

**vllm-mlx `BaseEngine` plugin (not built — deferred)**

`vllm-mlx` has a clean engine-abstraction seam:
- `vllm_mlx/engine/base.py`: `BaseEngine(ABC)` with `generate`/`stream_generate`/`chat`
- `pyproject.toml`: `mlx = "vllm_mlx.plugin:mlx_platform_plugin"` entry point

A future `PmetalEngine(BaseEngine)` could call pmetal over HTTP (or via `crates/pmetal-py`
PyO3 bindings) to run it as a co-equal backend inside vllm-mlx.  This enables consumer
code that calls `vllm-mlx`'s Python API directly (not just the HTTP endpoint) to switch
backends without code changes.

Deferred because it couples pmetal to vllm-mlx's evolving plugin ABI and Python
lifecycle — contrary to the "Rust-only in the hot path" goal for this milestone.
