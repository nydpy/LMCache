#!/usr/bin/env python3
"""
Large context KV cache test - tests with ~4000+ tokens.

This demonstrates cache benefits with longer contexts.

Usage: python test_kv_cache_large.py
"""

import os
import time
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from vllm import LLM, SamplingParams

print("="*60)
print("LARGE CONTEXT KV CACHE TEST")
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
- Leading the AI/ML platform team
- Architected the real-time inference system serving 10M requests/day
- Mentoring junior engineers and conducting code reviews
- Technologies: Python, PyTorch, Kubernetes, Redis, PostgreSQL

Previous Role: Software Engineer at DataSystems LLC (2017-2020)
- Built data pipelines processing 500TB of data daily
- Implemented machine learning models for fraud detection
- Reduced processing time by 60% through optimization
- Technologies: Scala, Spark, Kafka, AWS

Technical Skills:
Programming Languages:
- Python (Expert): 8 years experience, primary language for ML work
- Java (Advanced): 5 years, used for backend services
- Scala (Intermediate): 3 years, big data processing
- JavaScript/TypeScript (Intermediate): Frontend and Node.js
- Rust (Learning): Interested in systems programming

Frameworks and Libraries:
- PyTorch: Deep learning, custom model architectures
- TensorFlow: Production deployment, TensorFlow Serving
- FastAPI: Building REST APIs
- React: Frontend development
- Apache Spark: Large-scale data processing
- Kubernetes: Container orchestration, Helm charts

Projects:
Project 1: Real-time Recommendation Engine
- Built recommendation system serving 50M users
- Reduced latency from 200ms to 15ms
- A/B testing showed 23% increase in engagement

Project 2: Automated Code Review System
- ML model to detect bugs and suggest improvements
- Trained on 10M code snippets from internal repos
- Catches 40% of bugs before human review

Preferences:
- Prefers morning meetings, coding in afternoon
- Editor: VS Code with Vim keybindings
- Terminal: iTerm2 with zsh
- OS: macOS for development, Linux for servers

Hobbies:
- Rock climbing (bouldering V5-V6)
- Photography (landscape and street)
- Reading (sci-fi and technical books)
- Cooking (Italian and Japanese cuisine)

Goals:
- Lead a major ML infrastructure project
- Publish a paper at a top ML conference
- Become a Staff Engineer or Engineering Manager
"""

# Multiply to get longer context
MULTIPLIER = 6  # Repeat 6 times for ~4000+ tokens
long_document = "You are a helpful AI assistant. Below is detailed information about a user.\n"
for i in range(MULTIPLIER):
    long_document += f"\n--- SECTION {i+1} ---\n"
    long_document += doc_chunk

long_document += "\n\n=== END OF DOCUMENT ===\n\nBased on ALL sections above, answer the user's question.\n\nUser: "

# Count approximate tokens
approx_tokens = len(long_document.split()) * 1.3  # rough estimate
print(f"\nDocument size: ~{int(approx_tokens)} tokens ({MULTIPLIER} sections)")

print("\nLoading model...")
llm = LLM(
    model="Qwen/Qwen2-0.5B",
    max_model_len=8192,
    gpu_memory_utilization=0.5,
    enable_prefix_caching=True,
)
print("Model ready!\n")

sampling = SamplingParams(max_tokens=50, temperature=0.3)

questions = [
    "What is Alice's current job title?",
    "What programming language is Alice most skilled in?",
    "What editor does Alice use?",
    "What are Alice's hobbies?",
]

print("="*60)
print("ROUND 1: First requests (CACHE MISS)")
print("="*60)

times_round1 = []
for q in questions:
    prompt = long_document + q
    start = time.time()
    out = llm.generate([prompt], sampling)[0].outputs[0].text
    elapsed = time.time() - start
    times_round1.append(elapsed)
    print(f"\nQ: {q}")
    print(f"A: {out.strip()[:80]}...")
    print(f"Time: {elapsed*1000:.1f}ms")

print("\n" + "="*60)
print("ROUND 2: Same requests (CACHE HIT)")
print("="*60)

times_round2 = []
for q in questions:
    prompt = long_document + q
    start = time.time()
    out = llm.generate([prompt], sampling)[0].outputs[0].text
    elapsed = time.time() - start
    times_round2.append(elapsed)
    print(f"\nQ: {q}")
    print(f"A: {out.strip()[:80]}...")
    print(f"Time: {elapsed*1000:.1f}ms")

print("\n" + "="*60)
print("ROUND 3: New questions (PREFIX CACHE HIT)")
print("="*60)

new_questions = [
    "Where did Alice go to school?",
    "What was the latency improvement in the recommendation project?",
    "What is Alice's long-term career goal?",
    "What technologies did Alice use at DataSystems?",
]

times_round3 = []
for q in new_questions:
    prompt = long_document + q
    start = time.time()
    out = llm.generate([prompt], sampling)[0].outputs[0].text
    elapsed = time.time() - start
    times_round3.append(elapsed)
    print(f"\nQ: {q}")
    print(f"A: {out.strip()[:80]}...")
    print(f"Time: {elapsed*1000:.1f}ms")

print("\n" + "="*60)
print("RESULTS")
print("="*60)

avg_r1 = sum(times_round1) / len(times_round1) * 1000
avg_r2 = sum(times_round2) / len(times_round2) * 1000
avg_r3 = sum(times_round3) / len(times_round3) * 1000

print(f"\nDocument: ~{int(approx_tokens)} tokens")
print(f"\nRound 1 (cache miss):        {avg_r1:.1f}ms average")
print(f"Round 2 (full cache hit):    {avg_r2:.1f}ms average")
print(f"Round 3 (prefix cache hit):  {avg_r3:.1f}ms average")

speedup_r2 = avg_r1 / avg_r2 if avg_r2 > 0 else 0
speedup_r3 = avg_r1 / avg_r3 if avg_r3 > 0 else 0

print(f"\nSpeedup R1→R2: {speedup_r2:.2f}x")
print(f"Speedup R1→R3: {speedup_r3:.2f}x")

time_saved_r2 = avg_r1 - avg_r2
time_saved_r3 = avg_r1 - avg_r3

print(f"\nTime saved per request (R2): {time_saved_r2:.1f}ms")
print(f"Time saved per request (R3): {time_saved_r3:.1f}ms")

print("\n" + "="*60)
print("IMPACT AT SCALE")
print("="*60)
print(f"""
For {int(approx_tokens)} token context, serving 1000 requests/day:

Without cache: {avg_r1:.0f}ms × 1000 = {avg_r1:.0f} seconds/day
With cache:    {avg_r3:.0f}ms × 1000 = {avg_r3:.0f} seconds/day

Time saved per day: {time_saved_r3:.0f} seconds
Time saved per month: {time_saved_r3*30/60:.1f} minutes
""")
