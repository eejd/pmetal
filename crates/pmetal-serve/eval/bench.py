#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
bench.py — pmetal vs. mlx-vlm/vllm-mlx performance benchmark.

Measures TTFT (time-to-first-token), decode throughput (tok/s), total latency,
and peak RSS for both servers over the OpenAI chat/completions API.

Usage:
    # Against pmetal only:
    python eval/bench.py --pmetal-url http://localhost:8000

    # Head-to-head (start vllm-mlx separately first):
    python eval/bench.py \
        --pmetal-url  http://localhost:8000 \
        --baseline-url http://localhost:8001 \
        --baseline-name vllm-mlx

    # More thorough:
    python eval/bench.py --pmetal-url http://localhost:8000 \
        --concurrency 1 4 8 --max-tokens 256 --prompts 20

Requirements:
    /opt/local/bin/pip-3.14 install openai requests psutil
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

try:
    import requests
    from openai import OpenAI
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n  Run: /opt/local/bin/pip-3.14 install openai requests")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Benchmark prompts ─────────────────────────────────────────────────────────

PROMPTS = [
    "Explain the concept of attention in transformer models in 3 sentences.",
    "What is the difference between supervised and unsupervised learning?",
    "Write a haiku about Metal GPU shaders.",
    "List 5 key properties of Rust's ownership system.",
    "Summarise the history of Apple Silicon in 2 sentences.",
    "What does TTFT stand for and why does it matter for LLM serving?",
    "Describe the difference between a tokenizer and a detokenizer.",
    "In one sentence: why is KV cache important for autoregressive decoding?",
    "What are the main advantages of using safetensors over pickle?",
    "Explain what RoPE positional embeddings do.",
    "What is continuous batching in the context of LLM servers?",
    "Name three common quantisation schemes for LLM weights.",
    "What does GGUF stand for and what formats does it support?",
    "In one line, why is the GIL a problem for Python-based LLM servers?",
    "What is speculative decoding and when does it help throughput?",
    "Explain what MLX is and who maintains it.",
    "What is unified memory and why does it matter on Apple Silicon?",
    "Describe prefix caching in LLM inference.",
    "What is LoRA and what problem does it solve?",
    "Why might Rust outperform Python for a high-concurrency HTTP server?",
]


@dataclass
class RequestResult:
    prompt: str
    ttft_ms: float           # time to first token (streaming)
    total_ms: float          # total wall-clock latency
    completion_tokens: int   # tokens returned
    decode_tps: float        # completion_tokens / (total_ms - ttft_ms)
    error: Optional[str] = None


@dataclass
class BenchSummary:
    server: str
    concurrency: int
    n: int
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    total_p50_ms: float = 0.0
    total_p95_ms: float = 0.0
    decode_tps_mean: float = 0.0
    errors: int = 0
    results: list[RequestResult] = field(default_factory=list)

    def print(self):
        ok = [r for r in self.results if r.error is None]
        print(f"\n{'─'*60}")
        print(f"  Server : {self.server}")
        print(f"  Conc.  : {self.concurrency}  |  N={self.n}  |  Errors={self.errors}")
        if ok:
            print(f"  TTFT   : p50={self.ttft_p50_ms:.1f}ms  p95={self.ttft_p95_ms:.1f}ms")
            print(f"  Total  : p50={self.total_p50_ms:.1f}ms  p95={self.total_p95_ms:.1f}ms")
            print(f"  Decode : {self.decode_tps_mean:.1f} tok/s (mean)")
        print(f"{'─'*60}")


def run_one(client: OpenAI, model: str, prompt: str, max_tokens: int) -> RequestResult:
    """Single streaming request; measures TTFT and total latency."""
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    completion_tokens = 0

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if t_first is None and chunk.choices and chunk.choices[0].delta.content:
                t_first = time.perf_counter()
            # count tokens from usage chunk when available
            if hasattr(chunk, "usage") and chunk.usage is not None:
                completion_tokens = chunk.usage.completion_tokens or completion_tokens
            elif chunk.choices and chunk.choices[0].delta.content:
                completion_tokens += 1  # fallback: count non-empty deltas
    except Exception as exc:
        t_end = time.perf_counter()
        return RequestResult(
            prompt=prompt[:40],
            ttft_ms=0,
            total_ms=(t_end - t_start) * 1000,
            completion_tokens=0,
            decode_tps=0,
            error=str(exc),
        )

    t_end = time.perf_counter()
    ttft_ms = ((t_first or t_end) - t_start) * 1000
    total_ms = (t_end - t_start) * 1000
    decode_time_s = max((total_ms - ttft_ms) / 1000, 1e-6)
    decode_tps = completion_tokens / decode_time_s

    return RequestResult(
        prompt=prompt[:40],
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        completion_tokens=completion_tokens,
        decode_tps=decode_tps,
    )


def probe_model(url: str) -> str:
    """GET /v1/models → first model id, or 'unknown'."""
    try:
        r = requests.get(f"{url}/v1/models", timeout=5)
        data = r.json().get("data", [])
        return data[0]["id"] if data else "unknown"
    except Exception:
        return "unknown"


def bench_server(
    server_label: str,
    url: str,
    concurrency: int,
    prompts: list[str],
    max_tokens: int,
) -> BenchSummary:
    model = probe_model(url)
    client = OpenAI(base_url=f"{url}/v1", api_key="not-needed")
    summary = BenchSummary(server=f"{server_label} ({model})", concurrency=concurrency, n=len(prompts))

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(run_one, client, model, p, max_tokens): p for p in prompts}
        for fut in as_completed(futs):
            summary.results.append(fut.result())

    ok = [r for r in summary.results if r.error is None]
    summary.errors = len(summary.results) - len(ok)

    if ok:
        ttfts = sorted(r.ttft_ms for r in ok)
        totals = sorted(r.total_ms for r in ok)
        n = len(ok)
        summary.ttft_p50_ms = statistics.median(ttfts)
        summary.ttft_p95_ms = ttfts[int(n * 0.95)] if n > 1 else ttfts[-1]
        summary.total_p50_ms = statistics.median(totals)
        summary.total_p95_ms = totals[int(n * 0.95)] if n > 1 else totals[-1]
        summary.decode_tps_mean = statistics.mean(r.decode_tps for r in ok)

    return summary


def server_rss_mb(url: str) -> Optional[float]:
    """Try to get RSS of the process listening on the server port (best-effort)."""
    if not HAS_PSUTIL:
        return None
    try:
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        port = parsed.port
        for proc in psutil.process_iter(["pid", "connections", "memory_info"]):
            try:
                conns = proc.connections(kind="inet")
                if any(c.laddr.port == port and c.status == "LISTEN" for c in conns):
                    return proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    return None


def print_comparison(pmetal: list[BenchSummary], baseline: list[BenchSummary]):
    """Side-by-side comparison table."""
    print("\n" + "=" * 72)
    print("  HEAD-TO-HEAD COMPARISON")
    print("=" * 72)
    print(f"  {'Conc':>4}  {'Metric':>22}  {'pmetal':>12}  {'baseline':>12}  {'speedup':>8}")
    print(f"  {'-'*4}  {'-'*22}  {'-'*12}  {'-'*12}  {'-'*8}")
    for pm, bl in zip(pmetal, baseline):
        c = pm.concurrency
        rows = [
            ("TTFT p50 (ms)",    pm.ttft_p50_ms,     bl.ttft_p50_ms,     True),
            ("TTFT p95 (ms)",    pm.ttft_p95_ms,     bl.ttft_p95_ms,     True),
            ("Total p50 (ms)",   pm.total_p50_ms,    bl.total_p50_ms,    True),
            ("Decode tok/s",     pm.decode_tps_mean, bl.decode_tps_mean, False),
        ]
        for label, pm_val, bl_val, lower_better in rows:
            if bl_val > 0:
                speedup = bl_val / pm_val if lower_better else pm_val / bl_val
                marker = "✓" if speedup >= 1.0 else "✗"
                su_str = f"{marker} {speedup:.2f}×"
            else:
                su_str = "—"
            print(f"  {c:>4}  {label:>22}  {pm_val:>12.1f}  {bl_val:>12.1f}  {su_str:>8}")
        print()
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="pmetal vs. mlx-vlm/vllm-mlx serving benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pmetal-url",    default="http://localhost:8000",
                        help="pmetal server URL (default: %(default)s)")
    parser.add_argument("--baseline-url",  default="",
                        help="Baseline server URL for comparison (e.g. vllm-mlx)")
    parser.add_argument("--baseline-name", default="baseline",
                        help="Label for the baseline server (default: %(default)s)")
    parser.add_argument("--concurrency",   type=int, nargs="+", default=[1, 4],
                        help="Concurrency levels to test (default: 1 4)")
    parser.add_argument("--max-tokens",    type=int, default=128,
                        help="Max tokens per request (default: %(default)s)")
    parser.add_argument("--prompts",       type=int, default=len(PROMPTS),
                        help=f"Number of prompts to use (default: {len(PROMPTS)})")
    args = parser.parse_args()

    prompts = PROMPTS[: args.prompts]

    # ── Health check ──────────────────────────────────────────────────────────
    for url, label in [(args.pmetal_url, "pmetal")]  + (
        [(args.baseline_url, args.baseline_name)] if args.baseline_url else []
    ):
        try:
            r = requests.get(f"{url}/health", timeout=5)
            r.raise_for_status()
            print(f"✓ {label} is reachable at {url}")
        except Exception as e:
            sys.exit(f"✗ Cannot reach {label} at {url}: {e}")

    rss_pmetal   = server_rss_mb(args.pmetal_url)
    rss_baseline = server_rss_mb(args.baseline_url) if args.baseline_url else None

    pmetal_summaries:   list[BenchSummary] = []
    baseline_summaries: list[BenchSummary] = []

    for c in args.concurrency:
        print(f"\n{'='*60}")
        print(f"  Concurrency = {c}")
        print(f"{'='*60}")

        print(f"\n  → pmetal ({args.pmetal_url})")
        pm = bench_server("pmetal", args.pmetal_url, c, prompts, args.max_tokens)
        pm.print()
        pmetal_summaries.append(pm)

        if args.baseline_url:
            print(f"\n  → {args.baseline_name} ({args.baseline_url})")
            bl = bench_server(args.baseline_name, args.baseline_url, c, prompts, args.max_tokens)
            bl.print()
            baseline_summaries.append(bl)

    # ── RSS ───────────────────────────────────────────────────────────────────
    if rss_pmetal is not None:
        print(f"\n  Peak RSS — pmetal: {rss_pmetal:.0f} MB", end="")
        if rss_baseline is not None:
            print(f"  |  {args.baseline_name}: {rss_baseline:.0f} MB", end="")
        print()

    # ── Comparison table ──────────────────────────────────────────────────────
    if baseline_summaries:
        print_comparison(pmetal_summaries, baseline_summaries)
    else:
        print("\n(No baseline specified — run with --baseline-url for head-to-head comparison)")


if __name__ == "__main__":
    main()
