#!/usr/bin/env bash
# run_conformance.sh — Boot pmetal, run the vllm-mlx OpenAI compatibility tests.
#
# Usage:
#   ./eval/run_conformance.sh --model mlx-community/Qwen3-4B-Instruct-4bit
#   ./eval/run_conformance.sh --model /path/to/local-model --port 8080
#
# Requirements (MacPorts):
#   port install py314-requests py314-openai
#   pip314 not needed — all deps via MacPorts
#
# The vllm-mlx repo is expected at ../../../../../../Apple/Metal/vllm-mlx
# relative to this file (i.e. ~/Workspaces/Apple/Metal/vllm-mlx).  Override
# with VLLM_MLX_DIR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL=""
PORT=8000
SERVER_URL="http://localhost:${PORT}"
PMETAL_BIN="${REPO_ROOT}/target/release/pmetal"
VLLM_MLX_DIR="${VLLM_MLX_DIR:-${HOME}/Workspaces/Apple/Metal/vllm-mlx}"
PYTHON="${PYTHON:-/opt/local/bin/python3.14}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";      shift 2 ;;
        --port)        PORT="$2";       SERVER_URL="http://localhost:${PORT}"; shift 2 ;;
        --pmetal-bin)  PMETAL_BIN="$2"; shift 2 ;;
        --python)      PYTHON="$2";     shift 2 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    echo "ERROR: --model <model-id-or-path> is required" >&2
    exit 1
fi

if [[ ! -x "${PMETAL_BIN}" ]]; then
    echo "pmetal binary not found at ${PMETAL_BIN}; building release..." >&2
    cargo build --manifest-path "${REPO_ROOT}/Cargo.toml" \
        -p pmetal --features serve --release
fi

# ── Boot pmetal server ────────────────────────────────────────────────────────
echo "==> Starting pmetal serve --model ${MODEL} --port ${PORT}"
"${PMETAL_BIN}" serve --model "${MODEL}" --port "${PORT}" &
PMETAL_PID=$!
trap 'echo "Stopping pmetal (pid ${PMETAL_PID})"; kill "${PMETAL_PID}" 2>/dev/null; wait "${PMETAL_PID}" 2>/dev/null' EXIT

# Wait for the server to become healthy (up to 60 s)
echo "==> Waiting for server to become ready..."
for i in $(seq 1 60); do
    if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo "==> Server ready after ${i}s"
        break
    fi
    sleep 1
    if [[ ${i} -eq 60 ]]; then
        echo "ERROR: pmetal server did not become ready in 60 s" >&2
        exit 1
    fi
done

# ── Run the vllm-mlx compatibility test suite ─────────────────────────────────
COMPAT_SCRIPT="${VLLM_MLX_DIR}/examples/test_openai_compatibility.py"
if [[ ! -f "${COMPAT_SCRIPT}" ]]; then
    echo "ERROR: test_openai_compatibility.py not found at ${COMPAT_SCRIPT}" >&2
    echo "       Set VLLM_MLX_DIR to the vllm-mlx repo root." >&2
    exit 1
fi

echo ""
echo "==> Running OpenAI compatibility tests (text + streaming only, no vision)"
echo ""
"${PYTHON}" "${COMPAT_SCRIPT}" \
    --server-url "${SERVER_URL}" \
    --no-image \
    --no-video
COMPAT_EXIT=$?

# ── Run the simple demo script ────────────────────────────────────────────────
DEMO_SCRIPT="${VLLM_MLX_DIR}/examples/demo_openai_text.py"
if [[ -f "${DEMO_SCRIPT}" ]]; then
    echo ""
    echo "==> Running demo_openai_text.py"
    echo ""
    "${PYTHON}" "${DEMO_SCRIPT}" --server-url "${SERVER_URL}" || true
fi

echo ""
if [[ ${COMPAT_EXIT} -eq 0 ]]; then
    echo "✓ All conformance tests passed."
else
    echo "✗ Some conformance tests failed (exit ${COMPAT_EXIT})."
fi

exit ${COMPAT_EXIT}
