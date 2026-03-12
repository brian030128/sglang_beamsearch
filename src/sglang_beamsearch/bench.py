"""Benchmark sglang_beamsearch plugin.

Uses the same inputs as beam_engine/tests/benchmark_beam_search.py:
  - Model: meta-llama/Llama-3.1-8B
  - Batch size: 4 sequences
  - Prompt lengths: [512, 640, 768, 896] tokens (random IDs 1000–30000, seed=42)
  - Beam width: 4
  - Output length: 100 tokens
  - Warmup: 3 iters, Benchmark: 10 iters

Usage:
    # Terminal 1: Launch SGLang server with plugin
    python launch.py --model-path meta-llama/Llama-3.1-8B --port 8000

    # Terminal 2: Run benchmark
    python -m sglang_beamsearch.bench --port 8000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Same benchmark parameters as beam_engine/tests/benchmark_beam_search.py
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.1-8B"
BATCH_SIZE = 4
PROMPT_LENS = [512, 640, 768, 896]
BEAM_WIDTH = 4
OUTPUT_LEN = 100
NUM_WARMUP = 3
NUM_ITERS = 10
SEED = 42


def make_prompts(rng: np.random.Generator) -> list[list[int]]:
    """Generate BATCH_SIZE prompts with distinct lengths and random content.

    Identical to beam_engine's _make_prompts().
    """
    return [
        rng.integers(1000, 30000, size=length).tolist()
        for length in PROMPT_LENS
    ]


def run_beam_search_request(
    url: str,
    prompt_token_ids: list[int],
    beam_width: int,
    max_new_tokens: int,
    timeout: int = 300,
) -> dict:
    """Send a single beam search request to SGLang server."""
    payload = {
        "input_ids": prompt_token_ids,
        "sampling_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": max_new_tokens + 1,
            "custom_params": {
                "__beam_search__": {
                    "beam_width": beam_width,
                    "max_new_tokens": max_new_tokens,
                }
            },
        },
    }
    resp = requests.post(f"{url}/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def run_batch(
    url: str,
    prompts: list[list[int]],
    beam_width: int,
    max_new_tokens: int,
) -> tuple[float, list[dict]]:
    """Send all prompts sequentially and return (total_time_s, results).

    Times the full batch end-to-end.
    """
    t0 = time.perf_counter()
    results = []
    for prompt_ids in prompts:
        r = run_beam_search_request(url, prompt_ids, beam_width, max_new_tokens)
        results.append(r)
    elapsed = time.perf_counter() - t0
    return elapsed, results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sglang_beamsearch (same inputs as beam_engine benchmark)"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--max-new-tokens", type=int, default=OUTPUT_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-warmup", type=int, default=NUM_WARMUP)
    parser.add_argument("--num-iters", type=int, default=NUM_ITERS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    # Generate prompts (same as beam_engine)
    rng = np.random.default_rng(args.seed)
    prompt_lens = PROMPT_LENS[: args.batch_size]
    prompts = [
        rng.integers(1000, 30000, size=length).tolist()
        for length in prompt_lens
    ]

    input_lens_str = ", ".join(str(len(p)) for p in prompts)
    total_input = sum(len(p) for p in prompts)

    print("=" * 65)
    print(f"  Benchmark: sglang_beamsearch  (beam search, width={args.beam_width})")
    print(f"  Server:     {url}")
    print(f"  Batch size: {args.batch_size} sequences (sequential requests)")
    print(f"  Beam width: {args.beam_width}")
    print(f"  Input lens: [{input_lens_str}] tokens  (total {total_input})")
    print(f"  Output:     {args.max_new_tokens} tokens/seq")
    print(f"  Warmup:     {args.num_warmup}  |  Bench iters: {args.num_iters}")
    print("=" * 65)

    # Check server health
    try:
        r = requests.get(f"{url}/health", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"\nError: SGLang server not available at {url}: {e}")
        print("Start the server with: python launch.py --model-path <model> --port 8000")
        return

    # Verification pass
    print("\n[sglang_beamsearch] Verification pass (first 50 chars per sequence):")
    _, results = run_batch(url, prompts, args.beam_width, args.max_new_tokens)
    for i, r in enumerate(results):
        text = r.get("text", "")
        preview = text[:50] + "..." if len(text) > 50 else text
        print(f"  seq[{i}] (prompt_len={len(prompts[i])}): {preview!r}")

    # Warmup
    print(f"\n[sglang_beamsearch] Warming up ({args.num_warmup} iters)...")
    for w in range(args.num_warmup):
        t, _ = run_batch(url, prompts, args.beam_width, args.max_new_tokens)
        print(f"  warmup {w + 1}: {t * 1e3:.1f} ms")

    # Benchmark
    print(f"\n[sglang_beamsearch] Benchmarking ({args.num_iters} iters)...")
    latencies: list[float] = []
    for i in range(args.num_iters):
        t, _ = run_batch(url, prompts, args.beam_width, args.max_new_tokens)
        latencies.append(t)
        print(f"  iter {i + 1:2d}/{args.num_iters}: total={t * 1e3:.1f} ms")

    avg_ms = np.mean(latencies) * 1e3
    std_ms = np.std(latencies) * 1e3
    min_ms = np.min(latencies) * 1e3
    max_ms = np.max(latencies) * 1e3

    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  sglang_beamsearch  (beam_width={args.beam_width}, batch={args.batch_size})")
    print(f"    Avg latency:  {avg_ms:8.1f} ms  (std={std_ms:.1f})")
    print(f"    Min latency:  {min_ms:8.1f} ms")
    print(f"    Max latency:  {max_ms:8.1f} ms")
    print("=" * 65)


if __name__ == "__main__":
    main()
