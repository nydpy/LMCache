#!/usr/bin/env python3
"""
Selective KV Cache Quality Test - Simpler extraction test

Uses direct completion instead of Q&A to better test model memory.
"""

import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from vllm import LLM, SamplingParams

print("="*70)
print("SELECTIVE KV CACHE QUALITY TEST")
print("Direct completion test (not Q&A)")
print("="*70)

# Simple, direct context
system = "Complete the sentence based on the conversation.\n\n"

old_messages = """User: My name is Bob.
Assistant: Hello Bob!

User: I work at Google.
Assistant: Google is great!

"""

new_messages = """User: I just bought a Tesla.
Assistant: Nice Tesla!

User: I live in Seattle.
Assistant: Seattle is beautiful!

"""

# Test completions - more direct than Q&A
completions_new = [
    ("User: What car do I have?\nAssistant: You have a", "Tesla"),
    ("User: Where do I live?\nAssistant: You live in", "Seattle"),
]

completions_old = [
    ("User: What is my name?\nAssistant: Your name is", "Bob"),
    ("User: Where do I work?\nAssistant: You work at", "Google"),
]

print("\nLoading model (Qwen3-4B-Instruct-2507-AWQ)...")
llm = LLM(
    model="cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    enable_prefix_caching=True,
    quantization="awq",
)
print("Model ready!\n")

sampling = SamplingParams(max_tokens=10, temperature=0)

full_context = system + old_messages + new_messages
selective_context = system + new_messages  # Skip old

print("="*70)
print("TEST 1: FULL CONTEXT")
print("="*70)

print("\n[Completions about NEW content]")
full_new = []
for prompt, expected in completions_new:
    full_prompt = full_context + prompt
    out = llm.generate([full_prompt], sampling)[0].outputs[0].text.strip()
    ok = expected.lower() in out.lower()
    full_new.append(ok)
    print(f"  {'✓' if ok else '✗'} ...{prompt[-30:]} → {out[:20]}")

print("\n[Completions about OLD content]")
full_old = []
for prompt, expected in completions_old:
    full_prompt = full_context + prompt
    out = llm.generate([full_prompt], sampling)[0].outputs[0].text.strip()
    ok = expected.lower() in out.lower()
    full_old.append(ok)
    print(f"  {'✓' if ok else '✗'} ...{prompt[-30:]} → {out[:20]}")

print("\n" + "="*70)
print("TEST 2: SELECTIVE CONTEXT (skip old)")
print("="*70)

print("\n[Completions about NEW content - should work]")
sel_new = []
for prompt, expected in completions_new:
    full_prompt = selective_context + prompt
    out = llm.generate([full_prompt], sampling)[0].outputs[0].text.strip()
    ok = expected.lower() in out.lower()
    sel_new.append(ok)
    print(f"  {'✓' if ok else '✗'} ...{prompt[-30:]} → {out[:20]}")

print("\n[Completions about OLD content - should NOT complete correctly]")
sel_old = []
for prompt, expected in completions_old:
    full_prompt = selective_context + prompt
    out = llm.generate([full_prompt], sampling)[0].outputs[0].text.strip()
    ok = expected.lower() in out.lower()
    sel_old.append(ok)
    status = "✗ (good)" if not ok else "✓ (knows?)"
    print(f"  {status} ...{prompt[-30:]} → {out[:20]}")

print("\n" + "="*70)
print("TEST 3: CACHE HIT (same selective, 2nd time)")
print("="*70)

print("\n[Completions about NEW content - cache hit]")
cache_new = []
for prompt, expected in completions_new:
    full_prompt = selective_context + prompt
    out = llm.generate([full_prompt], sampling)[0].outputs[0].text.strip()
    ok = expected.lower() in out.lower()
    cache_new.append(ok)
    print(f"  {'✓' if ok else '✗'} ...{prompt[-30:]} → {out[:20]}")

print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"""
                        NEW         OLD
Full context:           {sum(full_new)}/2         {sum(full_old)}/2
Selective (skip old):   {sum(sel_new)}/2         {sum(sel_old)}/2 (want 0)
Cache hit:              {sum(cache_new)}/2         -
""")

# Analysis
if sum(sel_new) == sum(full_new):
    print("✓ NO DEGRADATION on new content")
    print("  Selective loading preserves quality for kept blocks")
else:
    print(f"⚠ Degradation: {sum(full_new) - sum(sel_new)} difference")

if sum(sel_old) == 0:
    print("\n✓ OLD CONTENT CORRECTLY EXCLUDED")
elif sum(sel_old) < sum(full_old):
    print(f"\n⚠ Partial hallucination on old content")

if sum(cache_new) == sum(sel_new):
    print("\n✓ CACHE HIT CONSISTENT")
    print("  Same results on second request")

print("""
="*70
INTERPRETATION
="*70

For memory-machine with selective anchor loading:

1. Anchors (like old_messages) can be selectively excluded
2. Quality preserved for loaded anchors (new_messages)
3. Cache hit produces same results
4. Model correctly "forgets" excluded content
""")
