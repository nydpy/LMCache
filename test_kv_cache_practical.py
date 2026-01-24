#!/usr/bin/env python3
"""
Practical KV cache test - actually store and retrieve from LMCache.

This test demonstrates:
1. First requests compute KV and store to cache (cache MISS)
2. Second requests load KV from cache (cache HIT)
3. Speedup from cache hits

Usage: python test_kv_cache_practical.py
"""

import os
import time
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from vllm import LLM, SamplingParams

print("="*60)
print("PRACTICAL KV CACHE TEST")
print("="*60)

print("\nLoading model with prefix caching enabled...")
llm = LLM(
    model="Qwen/Qwen2-0.5B",
    max_model_len=2048,
    gpu_memory_utilization=0.5,
    enable_prefix_caching=True,
)
print("Model ready!\n")

sampling = SamplingParams(max_tokens=30, temperature=0.3)

# Long shared prefix (will be cached)
shared_prefix = """You are a helpful AI assistant with extensive knowledge.

User: My name is Alice and I'm a software engineer.
Assistant: Nice to meet you, Alice! Software engineering is a great field.

User: I work at a startup called TechCorp in San Francisco.
Assistant: San Francisco is a great tech hub! What does TechCorp do?

User: We build AI tools for developers.
Assistant: That's exciting! AI tools are transforming how developers work.

User: I prefer Python for most of my work.
Assistant: Python is excellent for AI development with its rich ecosystem.

User: """

# Different questions using the SAME prefix
questions = [
    "What is my name?",
    "Where do I work?",
    "What programming language do I prefer?",
    "What does my company build?",
]

print("="*60)
print("ROUND 1: First requests (cache MISS - computing KV)")
print("="*60)

times_round1 = []
for q in questions:
    prompt = shared_prefix + q
    start = time.time()
    out = llm.generate([prompt], sampling)[0].outputs[0].text
    elapsed = time.time() - start
    times_round1.append(elapsed)
    print(f"\nQ: {q}")
    print(f"A: {out.strip()}")
    print(f"Time: {elapsed*1000:.1f}ms")

print("\n" + "="*60)
print("ROUND 2: Same requests (cache HIT - loading KV)")
print("="*60)

times_round2 = []
for q in questions:
    prompt = shared_prefix + q
    start = time.time()
    out = llm.generate([prompt], sampling)[0].outputs[0].text
    elapsed = time.time() - start
    times_round2.append(elapsed)
    print(f"\nQ: {q}")
    print(f"A: {out.strip()}")
    print(f"Time: {elapsed*1000:.1f}ms")

print("\n" + "="*60)
print("ROUND 3: Different suffix, same prefix (partial cache HIT)")
print("="*60)

new_questions = [
    "Tell me about yourself.",
    "What city am I in?",
    "Do I like Java or Python?",
]

times_round3 = []
for q in new_questions:
    prompt = shared_prefix + q
    start = time.time()
    out = llm.generate([prompt], sampling)[0].outputs[0].text
    elapsed = time.time() - start
    times_round3.append(elapsed)
    print(f"\nQ: {q}")
    print(f"A: {out.strip()}")
    print(f"Time: {elapsed*1000:.1f}ms")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)

avg_r1 = sum(times_round1) / len(times_round1) * 1000
avg_r2 = sum(times_round2) / len(times_round2) * 1000
avg_r3 = sum(times_round3) / len(times_round3) * 1000

print(f"\nRound 1 (cache miss):        {avg_r1:.1f}ms average")
print(f"Round 2 (full cache hit):    {avg_r2:.1f}ms average")
print(f"Round 3 (prefix cache hit):  {avg_r3:.1f}ms average")

if avg_r2 > 0:
    speedup_r2 = avg_r1 / avg_r2
    print(f"\nSpeedup R1→R2: {speedup_r2:.2f}x")

if avg_r3 > 0:
    speedup_r3 = avg_r1 / avg_r3
    print(f"Speedup R1→R3: {speedup_r3:.2f}x")

print("\n" + "="*60)
print("WHAT THIS SHOWS")
print("="*60)
print("""
- Round 1: First time seeing this prefix, must compute KV
- Round 2: Exact same prompts, KV loaded from cache (fastest)
- Round 3: Same prefix, new questions, prefix KV cached

With prefix caching:
- Shared conversation context is computed ONCE
- All subsequent requests reuse cached KV
- Only new tokens need attention computation
""")
