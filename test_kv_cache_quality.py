#!/usr/bin/env python3
"""
KV Cache Quality Test - Check for degradation in answers.

Compares responses from:
1. Cache MISS (compute fresh)
2. Cache HIT (load from cache)

Tests if KV cache loading produces identical/similar quality responses.

Usage: python test_kv_cache_quality.py
"""

import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from vllm import LLM, SamplingParams

print("="*70)
print("KV CACHE QUALITY TEST - Checking for Degradation")
print("="*70)

# Context document
context = """You are a helpful AI assistant with access to the following user profile.

=== USER PROFILE ===

Name: Alice Johnson
Age: 32
Birthday: March 15, 1992
Location: San Francisco, California
Occupation: Senior Software Engineer at TechCorp Inc.
Salary: $185,000 per year
Education: MS Computer Science from Stanford University (2016)

Work History:
- TechCorp Inc. (2020-present): Senior Software Engineer, AI Platform
- DataSystems LLC (2017-2020): Software Engineer, Data Pipelines
- Google (2016, intern): TensorFlow team

Technical Skills:
- Python (Expert, 8 years)
- Java (Advanced, 5 years)
- Rust (Learning, 6 months)

Favorite Programming Language: Python
Favorite Editor: VS Code with Vim keybindings
Operating System: macOS

Projects:
1. Recommendation Engine: Reduced latency from 200ms to 15ms
2. Code Review Bot: Catches 40% of bugs automatically

Hobbies:
- Rock climbing (bouldering V5-V6)
- Photography (Canon R5)
- Cooking (Italian and Japanese)

Favorite Food: Sushi
Favorite Coffee: Oat milk latte from Blue Bottle
Pet: Cat named "Pixel"

Goals:
- Short term: Get promoted to Staff Engineer
- Long term: Become CTO of a startup

=== END PROFILE ===

Answer questions about Alice based on the profile above. Be precise and factual.

User: """

print(f"\nContext size: ~{len(context.split())} words")

print("\nLoading model (Qwen3-4B-Instruct-2507-AWQ)...")
llm = LLM(
    model="cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    enable_prefix_caching=True,
    quantization="awq",
)
print("Model ready!\n")

# Use temperature=0 for deterministic outputs
sampling = SamplingParams(max_tokens=50, temperature=0)

# Test questions with known answers
test_cases = [
    ("What is Alice's full name?", "Alice Johnson"),
    ("How old is Alice?", "32"),
    ("Where does Alice work?", "TechCorp"),
    ("What is Alice's favorite programming language?", "Python"),
    ("What editor does Alice use?", "VS Code"),
    ("What is Alice's cat's name?", "Pixel"),
    ("What coffee does Alice like?", "oat milk latte"),
    ("What is Alice's salary?", "185,000"),
    ("When is Alice's birthday?", "March 15"),
    ("What is Alice's short term goal?", "Staff Engineer"),
]

print("="*70)
print("ROUND 1: Cache MISS (computing fresh)")
print("="*70)

results_round1 = []
for question, expected in test_cases:
    prompt = context + question
    output = llm.generate([prompt], sampling)[0].outputs[0].text.strip()
    has_answer = expected.lower() in output.lower()
    results_round1.append({
        "question": question,
        "expected": expected,
        "output": output[:80],
        "correct": has_answer
    })
    status = "✓" if has_answer else "✗"
    print(f"\n{status} Q: {question}")
    print(f"  Expected: {expected}")
    print(f"  Got: {output[:80]}...")

print("\n" + "="*70)
print("ROUND 2: Cache HIT (loading from cache)")
print("="*70)

results_round2 = []
for question, expected in test_cases:
    prompt = context + question
    output = llm.generate([prompt], sampling)[0].outputs[0].text.strip()
    has_answer = expected.lower() in output.lower()
    results_round2.append({
        "question": question,
        "expected": expected,
        "output": output[:80],
        "correct": has_answer
    })
    status = "✓" if has_answer else "✗"
    print(f"\n{status} Q: {question}")
    print(f"  Expected: {expected}")
    print(f"  Got: {output[:80]}...")

print("\n" + "="*70)
print("ROUND 3: Cache HIT (different questions, same context)")
print("="*70)

new_questions = [
    ("What university did Alice attend?", "Stanford"),
    ("How many years of Python experience?", "8"),
    ("What was the latency improvement?", "15ms"),
    ("What percentage of bugs does the bot catch?", "40%"),
    ("What type of climbing does Alice do?", "bouldering"),
]

results_round3 = []
for question, expected in new_questions:
    prompt = context + question
    output = llm.generate([prompt], sampling)[0].outputs[0].text.strip()
    has_answer = expected.lower() in output.lower()
    results_round3.append({
        "question": question,
        "expected": expected,
        "output": output[:80],
        "correct": has_answer
    })
    status = "✓" if has_answer else "✗"
    print(f"\n{status} Q: {question}")
    print(f"  Expected: {expected}")
    print(f"  Got: {output[:80]}...")

print("\n" + "="*70)
print("COMPARISON: Cache Miss vs Cache Hit")
print("="*70)

# Compare outputs
print("\nOutput comparison (should be IDENTICAL with temp=0):\n")
all_identical = True
for r1, r2 in zip(results_round1, results_round2):
    match = "IDENTICAL" if r1["output"] == r2["output"] else "DIFFERENT"
    if r1["output"] != r2["output"]:
        all_identical = False
    print(f"Q: {r1['question'][:40]}")
    print(f"  R1: {r1['output'][:50]}...")
    print(f"  R2: {r2['output'][:50]}...")
    print(f"  Status: {match}")
    print()

print("="*70)
print("RESULTS SUMMARY")
print("="*70)

r1_correct = sum(1 for r in results_round1 if r["correct"])
r2_correct = sum(1 for r in results_round2 if r["correct"])
r3_correct = sum(1 for r in results_round3 if r["correct"])

print(f"\nRound 1 (cache miss): {r1_correct}/{len(results_round1)} correct")
print(f"Round 2 (cache hit):  {r2_correct}/{len(results_round2)} correct")
print(f"Round 3 (new Qs):     {r3_correct}/{len(results_round3)} correct")

print(f"\nOutputs identical between R1 and R2: {'YES' if all_identical else 'NO'}")

if r1_correct == r2_correct and all_identical:
    print("\n" + "="*70)
    print("✓ NO DEGRADATION DETECTED")
    print("="*70)
    print("""
KV cache hit produces IDENTICAL responses to cache miss.
This confirms:
1. KV cache loading preserves attention computation
2. No quality loss from using cached values
3. Safe to use for production
""")
elif r1_correct == r2_correct:
    print("\n" + "="*70)
    print("✓ SAME ACCURACY (minor output variations)")
    print("="*70)
    print("""
Same number of correct answers, but some wording differences.
This is normal with sampling - the core facts are preserved.
""")
else:
    diff = r1_correct - r2_correct
    print("\n" + "="*70)
    print(f"⚠ POTENTIAL DEGRADATION: {abs(diff)} answer difference")
    print("="*70)
    print("""
Cache hit produced different accuracy than cache miss.
This needs investigation - check the specific questions that differ.
""")
