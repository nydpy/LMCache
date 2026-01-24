#!/usr/bin/env python3
"""
KV cache test with single token output.

This isolates the prefill time by generating only 1 token.
Shows true KV cache speedup without generation overhead.

Usage: python test_kv_cache_one_token.py
"""

import os
import time

os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from vllm import LLM, SamplingParams

print("="*60)
print("KV CACHE TEST - SINGLE TOKEN OUTPUT")
print("="*60)

# Base document chunk
doc_chunk = """
=== USER PROFILE SECTION ===

Name: Alice Johnson
Age: 32
Location: San Francisco, California
Occupation: Senior Software Engineer at TechCorp Inc.
Education: MS Computer Science from Stanford University

Work History:
Current Role: Senior Software Engineer at TechCorp Inc. (2020-present)
- Leading the AI/ML platform team with 12 direct reports
- Architected the real-time inference system serving 10M requests/day
- Technologies: Python, PyTorch, Kubernetes, Redis, PostgreSQL

Previous Role: Software Engineer at DataSystems LLC (2017-2020)
- Built data pipelines processing 500TB of data daily
- Implemented machine learning models for fraud detection
- Technologies: Scala, Spark, Kafka, AWS

Technical Skills:
- Python (Expert): 8 years experience
- Java (Advanced): 5 years
- PyTorch, TensorFlow, FastAPI, Kubernetes

Projects:
- Real-time Recommendation Engine: 50M users, 15ms latency
- Automated Code Review: catches 40% of bugs

Preferences:
- Editor: VS Code with Vim keybindings
- OS: macOS for development

Hobbies:
- Rock climbing, Photography, Cooking
"""

# Multiply for longer context
MULTIPLIER = 6
long_document = "You are a helpful AI assistant.\n"
for i in range(MULTIPLIER):
    long_document += f"\n--- SECTION {i+1} ---\n"
    long_document += doc_chunk

long_document += "\n\nAnswer with just one word.\n\nUser: "

approx_tokens = len(long_document.split()) * 1.3
print(f"\nDocument: ~{int(approx_tokens)} tokens")

# Try 0.5B first (faster), fall back to 7B
model_name = "Qwen/Qwen2-0.5B"
print(f"Model: {model_name}")

print("\nLoading model...")
llm = LLM(
    model=model_name,
    max_model_len=4096,
    gpu_memory_utilization=0.5,
    enable_prefix_caching=True,
)
print("Model ready!\n")

# Only generate 1 token!
sampling = SamplingParams(max_tokens=1, temperature=0)

questions = [
    "What is Alice's first name?",
    "What city does Alice live in?",
    "What is Alice's job title?",
    "What editor does Alice use?",
]

print("="*60)
print("ROUND 1: Cache MISS (1 token output)")
print("="*60)

times_r1 = []
for q in questions:
    prompt = long_document + q
    start = time.time()
    out = llm.generate([prompt], sampling)[0].outputs[0].text
    elapsed = time.time() - start
    times_r1.append(elapsed)
    print(f"Q: {q}")
    print(f"A: [{out.strip()}]  Time: {elapsed*1000:.1f}ms")

print("\n" + "="*60)
print("ROUND 2: Cache HIT (1 token output)")
print("="*60)

times_r2 = []
for q in questions:
    prompt = long_document + q
    start = time.time()
    out = llm.generate([prompt], sampling)[0].outputs[0].text
    elapsed = time.time() - start
    times_r2.append(elapsed)
    print(f"Q: {q}")
    print(f"A: [{out.strip()}]  Time: {elapsed*1000:.1f}ms")

print("\n" + "="*60)
print("ROUND 3: New questions, Cache HIT (1 token output)")
print("="*60)

new_questions = [
    "Is Alice a man or woman?",
    "Does Alice use Mac or Windows?",
    "Is Alice senior or junior?",
    "Does Alice like Python or Java?",
]

times_r3 = []
for q in new_questions:
    prompt = long_document + q
    start = time.time()
    out = llm.generate([prompt], sampling)[0].outputs[0].text
    elapsed = time.time() - start
    times_r3.append(elapsed)
    print(f"Q: {q}")
    print(f"A: [{out.strip()}]  Time: {elapsed*1000:.1f}ms")

print("\n" + "="*60)
print("RESULTS (1 TOKEN OUTPUT)")
print("="*60)

avg_r1 = sum(times_r1) / len(times_r1) * 1000
avg_r2 = sum(times_r2) / len(times_r2) * 1000
avg_r3 = sum(times_r3) / len(times_r3) * 1000

print(f"\nDocument: ~{int(approx_tokens)} tokens, Output: 1 token")
print(f"\nRound 1 (cache miss):   {avg_r1:.1f}ms")
print(f"Round 2 (cache hit):    {avg_r2:.1f}ms")
print(f"Round 3 (cache hit):    {avg_r3:.1f}ms")

speedup = avg_r1 / avg_r3 if avg_r3 > 0 else 0
print(f"\nSpeedup: {speedup:.1f}x")
print(f"Time saved: {avg_r1 - avg_r3:.1f}ms per request")

print("\n" + "="*60)
print("WHAT THIS SHOWS")
print("="*60)
print(f"""
With only 1 token output, we isolate the PREFILL time:

  Cache MISS: {avg_r1:.0f}ms = prefill {int(approx_tokens)} tokens + generate 1 token
  Cache HIT:  {avg_r3:.0f}ms = load KV cache + generate 1 token

The {speedup:.0f}x speedup is almost pure prefill savings!

For classification tasks (yes/no, A/B/C/D):
  → KV cache gives massive speedup
  → Most time was computing attention, not generating
""")
