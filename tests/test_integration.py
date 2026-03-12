"""Integration tests for sglang_beamsearch plugin.

These tests require a running SGLang server with the beam search plugin loaded.
Set the SGLANG_BEAM_TEST_URL env var to the server URL (default: http://localhost:8000).

Run:
    SGLANG_BEAM_TEST_URL=http://localhost:8000 pytest tests/test_integration.py -v
"""

from __future__ import annotations

import os
import time

import pytest
import requests

SGLANG_URL = os.environ.get("SGLANG_BEAM_TEST_URL", "http://localhost:8000")


def _server_available() -> bool:
    try:
        r = requests.get(f"{SGLANG_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason=f"SGLang server not available at {SGLANG_URL}",
)


def _generate(prompt: str, beam_width: int, max_new_tokens: int, timeout: int = 120) -> dict:
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
    resp = requests.post(f"{SGLANG_URL}/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _generate_normal(prompt: str, max_new_tokens: int, timeout: int = 60) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
    }
    resp = requests.post(f"{SGLANG_URL}/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# Basic functionality
# --------------------------------------------------------------------------- #


class TestBeamSearchBasic:
    def test_server_responds(self):
        """Verify the server is up and responding."""
        r = requests.get(f"{SGLANG_URL}/health", timeout=5)
        assert r.status_code == 200

    def test_normal_request_still_works(self):
        """Non-beam requests should work normally after plugin is loaded."""
        result = _generate_normal("Hello, world!", max_new_tokens=10)
        assert "text" in result
        assert len(result["text"]) > 0

    def test_beam_search_returns_output(self):
        """Beam search request should return generated text."""
        result = _generate(
            "The capital of France is",
            beam_width=2,
            max_new_tokens=16,
        )
        assert "text" in result
        assert len(result["text"]) > 0

    def test_beam_search_width_4(self):
        """Test with beam_width=4."""
        result = _generate(
            "Once upon a time",
            beam_width=4,
            max_new_tokens=32,
        )
        assert "text" in result
        assert len(result["text"]) > 0


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


class TestBeamSearchDeterminism:
    def test_same_prompt_same_output(self):
        """Beam search should be deterministic for the same prompt."""
        prompt = "The meaning of life is"
        r1 = _generate(prompt, beam_width=2, max_new_tokens=20)
        r2 = _generate(prompt, beam_width=2, max_new_tokens=20)
        assert r1["text"] == r2["text"], (
            f"Non-deterministic output:\n  run1: {r1['text']!r}\n  run2: {r2['text']!r}"
        )


# --------------------------------------------------------------------------- #
# Multiple concurrent requests
# --------------------------------------------------------------------------- #


class TestBeamSearchConcurrent:
    def test_two_sequential_requests(self):
        """Two beam search requests in sequence should both complete."""
        r1 = _generate("The sky is", beam_width=2, max_new_tokens=10)
        r2 = _generate("The ocean is", beam_width=2, max_new_tokens=10)
        assert "text" in r1
        assert "text" in r2
        # Different prompts should give different outputs
        # (not guaranteed but very likely)


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #


class TestBeamSearchBenchmark:
    @pytest.mark.parametrize("beam_width", [1, 2, 4])
    def test_benchmark_beam_widths(self, beam_width):
        """Benchmark beam search with different beam widths."""
        prompt = "The future of artificial intelligence is"
        max_new_tokens = 64
        num_runs = 3

        # Warmup
        _generate(prompt, beam_width=beam_width, max_new_tokens=8)

        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            result = _generate(prompt, beam_width=beam_width, max_new_tokens=max_new_tokens)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)
            assert "text" in result

        avg_ms = sum(latencies) / len(latencies)
        min_ms = min(latencies)
        max_ms = max(latencies)
        print(
            f"\n  beam_width={beam_width}, max_new_tokens={max_new_tokens}: "
            f"avg={avg_ms:.1f}ms, min={min_ms:.1f}ms, max={max_ms:.1f}ms"
        )

    def test_benchmark_token_lengths(self):
        """Benchmark beam search with different output lengths."""
        prompt = "The future of artificial intelligence is"
        beam_width = 4

        for max_new_tokens in [16, 64, 128]:
            # Warmup
            _generate(prompt, beam_width=beam_width, max_new_tokens=8)

            t0 = time.perf_counter()
            result = _generate(prompt, beam_width=beam_width, max_new_tokens=max_new_tokens)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            assert "text" in result
            print(
                f"\n  max_new_tokens={max_new_tokens}, beam_width={beam_width}: "
                f"{elapsed_ms:.1f}ms"
            )
