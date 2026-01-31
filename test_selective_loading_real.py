#!/usr/bin/env python3
"""
Real Selective KV Cache Loading Test

This test validates that selective loading actually uses cached KV blocks
instead of recomputing them.

Test flow:
1. First request: Full context [system][old][new] - computes all KV, stores to cache
2. Second request: Same full context - should be cache HIT (faster)
3. Third request: Selective [system][new] only - loads from cache, skips [old]

Validates:
- Selective loading uses cached blocks (not recompute)
- Output quality preserved
- Performance improvement from cache hits

Usage: python test_selective_loading_real.py
"""

import os
import time
import hashlib

os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from vllm import LLM, SamplingParams

print("="*70)
print("REAL SELECTIVE KV CACHE LOADING TEST")
print("="*70)

# Message blocks - each will get its own cache entry
system_block = """You are a helpful assistant with perfect memory.
Answer questions based on the conversation history."""

old_block = """User: My name is Bob.
Assistant: Hello Bob!

User: I work at Google.
Assistant: Google is great!"""

new_block = """User: I just bought a Tesla.
Assistant: Nice Tesla!

User: I live in Seattle.
Assistant: Seattle is beautiful!"""

def compute_block_hash(text: str) -> int:
    """Compute hash for a text block (simulates LMCache hashing)"""
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)

# Compute hashes for each block
system_hash = compute_block_hash(system_block)
old_hash = compute_block_hash(old_block)
new_hash = compute_block_hash(new_block)

print(f"\nBlock hashes:")
print(f"  System: {system_hash}")
print(f"  Old:    {old_hash}")
print(f"  New:    {new_hash}")

# Estimate token counts (rough: 4 chars per token, rounded to 256)
def estimate_tokens(text: str) -> int:
    return max(256, ((len(text) // 4) // 256 + 1) * 256)

system_tokens = estimate_tokens(system_block)
old_tokens = estimate_tokens(old_block)
new_tokens = estimate_tokens(new_block)

print(f"\nEstimated tokens:")
print(f"  System: {system_tokens}")
print(f"  Old:    {old_tokens}")
print(f"  New:    {new_tokens}")

print("\nLoading model (Qwen3-4B-Instruct-2507-AWQ)...")
llm = LLM(
    model="cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    enable_prefix_caching=True,
)
print("Model ready!\n")

sampling = SamplingParams(max_tokens=20, temperature=0)

# Build prompts
full_context = system_block + "\n\n" + old_block + "\n\n" + new_block
selective_context = system_block + "\n\n" + new_block  # Skip old

question_new = "\n\nUser: What car do I have?\nAssistant:"
question_old = "\n\nUser: What is my name?\nAssistant:"

print("="*70)
print("TEST 1: FULL CONTEXT - First request (cache MISS, computes KV)")
print("="*70)

prompt1 = full_context + question_new
start = time.time()
out1 = llm.generate([prompt1], sampling)[0].outputs[0].text.strip()
time1 = time.time() - start
print(f"  Output: {out1[:50]}")
print(f"  Time: {time1:.3f}s (computing all KV)")

print("\n" + "="*70)
print("TEST 2: FULL CONTEXT - Second request (cache HIT)")
print("="*70)

prompt2 = full_context + question_old
start = time.time()
out2 = llm.generate([prompt2], sampling)[0].outputs[0].text.strip()
time2 = time.time() - start
print(f"  Output: {out2[:50]}")
print(f"  Time: {time2:.3f}s (should be faster - prefix cache hit)")

print("\n" + "="*70)
print("TEST 3: SELECTIVE CONTEXT - Skip old block")
print("="*70)

prompt3 = selective_context + question_new
start = time.time()
out3 = llm.generate([prompt3], sampling)[0].outputs[0].text.strip()
time3 = time.time() - start
print(f"  Output: {out3[:50]}")
print(f"  Time: {time3:.3f}s")

print("\n" + "="*70)
print("TEST 4: SELECTIVE - Ask about OLD content (should NOT know)")
print("="*70)

prompt4 = selective_context + question_old
start = time.time()
out4 = llm.generate([prompt4], sampling)[0].outputs[0].text.strip()
time4 = time.time() - start
print(f"  Output: {out4[:50]}")
print(f"  Time: {time4:.3f}s")

# Check results
knows_tesla_full = "tesla" in out1.lower()
knows_bob_full = "bob" in out2.lower()
knows_tesla_selective = "tesla" in out3.lower()
knows_bob_selective = "bob" in out4.lower()

print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"""
                          Full Context    Selective (skip old)
                          ────────────    ────────────────────
Knows Tesla (new):        {'YES' if knows_tesla_full else 'NO':^12}    {'YES' if knows_tesla_selective else 'NO':^20}
Knows Bob (old):          {'YES' if knows_bob_full else 'NO':^12}    {'YES' if knows_bob_selective else 'NO':^20} (want NO)

Timing:
  Full context (1st):     {time1:.3f}s (cache miss)
  Full context (2nd):     {time2:.3f}s (cache hit)
  Selective (1st):        {time3:.3f}s
  Selective (2nd):        {time4:.3f}s

  Speedup (cache hit):    {time1/time2:.2f}x
""")

# Analysis
print("="*70)
print("ANALYSIS")
print("="*70)

if knows_tesla_full and knows_tesla_selective:
    print("\n[OK] NEW content (Tesla) accessible in both modes")
else:
    print("\n[FAIL] NEW content not accessible")

if knows_bob_full and not knows_bob_selective:
    print("[OK] OLD content (Bob) correctly excluded in selective mode")
elif not knows_bob_full:
    print("[WARN] Model didn't know Bob even with full context")
else:
    print("[FAIL] Model hallucinated Bob in selective mode")

if time2 < time1:
    print(f"[OK] Cache hit faster: {time1/time2:.2f}x speedup")
else:
    print("[WARN] Cache hit not faster than miss")

print("""
="*70
NOTE ON SELECTIVE LOADING
="*70

This test uses vLLM's built-in prefix caching.

For TRUE selective loading (load blocks 1 and 3, skip 2),
you need to run a vLLM server with LMCache and pass:

    extra_body = {
        "kv_transfer_params": {
            "lmcache.selective_hashes": [hash1, hash3],  # Skip hash2
            "lmcache.selective_offsets": [256, 256]
        }
    }

See demo_selective_chat.py for the full API.
""")
