"""Benchmark script for sglang_beamsearch plugin.

Usage:
    # Terminal 1: Launch SGLang server with plugin
    python -c "
    from sglang_beamsearch import load_beam_search_plugin
    load_beam_search_plugin()
    import sglang.srt.server
    sglang.srt.server.main()
    " --model-path <model> --port 8000

    # Terminal 2: Run benchmark
    python -m sglang_beamsearch.bench --port 8000 --beam-width 4 --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import json
import time

import requests


def run_beam_search_request(
    url: str,
    prompt: str,
    beam_width: int,
    max_new_tokens: int,
) -> dict:
    """Send a beam search request to SGLang server."""
    payload = {
        "text": prompt,
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

    t0 = time.perf_counter()
    resp = requests.post(f"{url}/generate", json=payload, timeout=300)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    resp.raise_for_status()
    result = resp.json()
    result["elapsed_ms"] = elapsed_ms
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark sglang_beamsearch")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument(
        "--prompt",
        default="The future of artificial intelligence is",
        help="Prompt text (used for all requests)",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    # Warmup
    print(f"Warming up with 1 request...")
    try:
        run_beam_search_request(url, args.prompt, args.beam_width, 8)
    except Exception as e:
        print(f"Warmup failed: {e}")
        print("Is the SGLang server running with the beam search plugin?")
        return

    # Benchmark
    print(f"\nRunning {args.num_prompts} beam search request(s):")
    print(f"  beam_width={args.beam_width}")
    print(f"  max_new_tokens={args.max_new_tokens}")
    print(f"  prompt={args.prompt!r}")
    print()

    total_ms = 0
    for i in range(args.num_prompts):
        result = run_beam_search_request(
            url, args.prompt, args.beam_width, args.max_new_tokens
        )
        elapsed = result["elapsed_ms"]
        total_ms += elapsed
        print(f"  Request {i + 1}: {elapsed:.1f} ms")
        if "text" in result:
            text = result["text"]
            if len(text) > 200:
                text = text[:200] + "..."
            print(f"    Output: {text}")

    avg_ms = total_ms / args.num_prompts
    print(f"\nAverage: {avg_ms:.1f} ms over {args.num_prompts} request(s)")
    print(f"Total:   {total_ms:.1f} ms")


if __name__ == "__main__":
    main()
