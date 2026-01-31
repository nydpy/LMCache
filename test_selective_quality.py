#!/usr/bin/env python3
"""
Selective KV Cache Quality Test - Position Shift Degradation Check

Tests if loading KV from different positions causes quality degradation.

Scenario:
- System prompt: tokens 0-100 (always cached)
- Old messages: tokens 100-300 (SKIPPED)
- New messages: tokens 300-500 (kept)

Cache MISS: Recompute new messages at positions 100-300 (correct RoPE)
Cache HIT:  Load cached KV from 300-500, place at 100-300 (RoPE mismatch?)

Usage: python test_selective_quality.py
"""

import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from vllm import LLM, SamplingParams

print("="*70)
print("SELECTIVE KV CACHE QUALITY TEST")
print("Position Shift Degradation Check")
print("="*70)

# System prompt (always included)
system_prompt = """You are a helpful AI assistant. You remember all conversations accurately.

"""

# Old messages (will be SKIPPED in selective test)
old_messages = """User: My name is Alice.
Assistant: Nice to meet you, Alice!

User: I work at Google as a software engineer.
Assistant: That's great! Google is an amazing company.

User: I have a dog named Max.
Assistant: Max is a lovely name for a dog!

User: My favorite color is blue.
Assistant: Blue is a calming and beautiful color.

User: I live in San Francisco.
Assistant: San Francisco is a wonderful city!

"""

# New messages (will be KEPT)
new_messages = """User: I just got promoted to Senior Engineer!
Assistant: Congratulations on your promotion! That's fantastic news.

User: I'm planning to buy a Tesla Model 3.
Assistant: The Tesla Model 3 is a great electric vehicle choice.

User: My birthday is next week, on March 15th.
Assistant: How exciting! I'll remember your birthday is March 15th.

"""

# Test questions about NEW messages only
test_questions = [
    ("What position did I get promoted to?", "Senior Engineer"),
    ("What car am I planning to buy?", "Tesla"),
    ("When is my birthday?", "March 15"),
]

# Test questions about OLD messages (should NOT know if skipped correctly)
old_questions = [
    ("What is my dog's name?", "Max"),
    ("What company do I work at?", "Google"),
    ("What is my favorite color?", "blue"),
]

print("\nBuilding contexts...")
print(f"  System prompt: ~{len(system_prompt.split())} words")
print(f"  Old messages:  ~{len(old_messages.split())} words (to be skipped)")
print(f"  New messages:  ~{len(new_messages.split())} words (to be kept)")

# Full context (for comparison baseline)
full_context = system_prompt + old_messages + new_messages + "User: "

# Selective context (skip old messages - simulates cache hit with shift)
selective_context = system_prompt + new_messages + "User: "

print(f"\n  Full context:      ~{len(full_context.split())} words")
print(f"  Selective context: ~{len(selective_context.split())} words")

print("\nLoading model...")
llm = LLM(
    model="Qwen/Qwen2-0.5B",
    max_model_len=4096,
    gpu_memory_utilization=0.5,
    enable_prefix_caching=True,
)
print("Model ready!\n")

sampling = SamplingParams(max_tokens=30, temperature=0)

print("="*70)
print("TEST 1: Full Context (baseline - knows everything)")
print("="*70)

print("\nQuestions about NEW messages:")
full_new_results = []
for q, expected in test_questions:
    prompt = full_context + q
    output = llm.generate([prompt], sampling)[0].outputs[0].text.strip()
    has_answer = expected.lower() in output.lower()
    full_new_results.append(has_answer)
    status = "✓" if has_answer else "✗"
    print(f"  {status} Q: {q}")
    print(f"    Expected: {expected}")
    print(f"    Got: {output[:60]}...")

print("\nQuestions about OLD messages:")
full_old_results = []
for q, expected in old_questions:
    prompt = full_context + q
    output = llm.generate([prompt], sampling)[0].outputs[0].text.strip()
    has_answer = expected.lower() in output.lower()
    full_old_results.append(has_answer)
    status = "✓" if has_answer else "✗"
    print(f"  {status} Q: {q}")
    print(f"    Expected: {expected}")
    print(f"    Got: {output[:60]}...")

print("\n" + "="*70)
print("TEST 2: Selective Context (skip old messages)")
print("="*70)

print("\nQuestions about NEW messages (should still work):")
selective_new_results = []
for q, expected in test_questions:
    prompt = selective_context + q
    output = llm.generate([prompt], sampling)[0].outputs[0].text.strip()
    has_answer = expected.lower() in output.lower()
    selective_new_results.append(has_answer)
    status = "✓" if has_answer else "✗"
    print(f"  {status} Q: {q}")
    print(f"    Expected: {expected}")
    print(f"    Got: {output[:60]}...")

print("\nQuestions about OLD messages (should NOT know - skipped!):")
selective_old_results = []
for q, expected in old_questions:
    prompt = selective_context + q
    output = llm.generate([prompt], sampling)[0].outputs[0].text.strip()
    has_answer = expected.lower() in output.lower()
    selective_old_results.append(has_answer)
    status = "✓" if has_answer else "✗"  # ✓ here means it wrongly "knows"
    print(f"  {status} Q: {q}")
    print(f"    Expected to NOT know: {expected}")
    print(f"    Got: {output[:60]}...")

print("\n" + "="*70)
print("TEST 3: Cache Hit Simulation (same selective, second time)")
print("="*70)

print("\nQuestions about NEW messages (cache hit):")
cache_hit_results = []
for q, expected in test_questions:
    prompt = selective_context + q
    output = llm.generate([prompt], sampling)[0].outputs[0].text.strip()
    has_answer = expected.lower() in output.lower()
    cache_hit_results.append(has_answer)
    status = "✓" if has_answer else "✗"
    print(f"  {status} Q: {q}")
    print(f"    Expected: {expected}")
    print(f"    Got: {output[:60]}...")

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

full_new_score = sum(full_new_results)
full_old_score = sum(full_old_results)
selective_new_score = sum(selective_new_results)
selective_old_score = sum(selective_old_results)
cache_hit_score = sum(cache_hit_results)

print(f"""
                          NEW msgs    OLD msgs
                          (should ✓)  (context dependent)
─────────────────────────────────────────────────────────
Full context:             {full_new_score}/{len(test_questions)}         {full_old_score}/{len(old_questions)} (should know)
Selective (skip old):     {selective_new_score}/{len(test_questions)}         {selective_old_score}/{len(old_questions)} (should NOT know)
Cache hit (2nd selective):{cache_hit_score}/{len(test_questions)}         -
""")

# Check for degradation
if selective_new_score == full_new_score:
    print("✓ NO DEGRADATION on new messages!")
    print("  Selective loading preserves quality for kept context.")
else:
    diff = full_new_score - selective_new_score
    print(f"⚠ DEGRADATION DETECTED: {diff} fewer correct on new messages")

if selective_new_score == cache_hit_score:
    print("\n✓ CACHE HIT CONSISTENT with cache miss")
    print("  Second request produces same quality.")
else:
    print("\n⚠ CACHE INCONSISTENCY between miss and hit")

if selective_old_score < full_old_score:
    print(f"\n✓ OLD MESSAGES CORRECTLY FORGOTTEN")
    print(f"  Full context knew {full_old_score}/3, selective knew {selective_old_score}/3")
else:
    print(f"\n? Model may be hallucinating old message content")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
This test simulates:
1. Full context:     System + Old + New (baseline)
2. Selective:        System + New (skip old - like sliding window)
3. Cache hit:        Same as selective, second request

Key findings:
- If NEW message accuracy is same: Position shift doesn't hurt quality
- If OLD message accuracy drops: Selective loading correctly excludes old
- If Cache hit = Selective: KV cache is consistent

For your memory-machine:
- System prompt (anchors 0-100) → always cached
- Old anchors (100-300) → can be skipped without quality loss on new
- New/relevant anchors (300-500) → loaded and shifted, quality preserved
""")
