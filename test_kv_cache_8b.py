#!/usr/bin/env python3
"""
Large model (8B) KV cache test with 4-bit quantization.

Tests KV cache performance with Qwen2-7B-Instruct-AWQ (4-bit quantized).
Works on T4 GPU (16GB VRAM).

Usage: python test_kv_cache_8b.py
"""

import os
import time

os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from vllm import LLM, SamplingParams

print("="*60)
print("8B MODEL KV CACHE TEST (4-bit AWQ)")
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
- Reduced latency by 40% through custom CUDA kernels
- Mentoring junior engineers and conducting weekly code reviews
- Technologies: Python, PyTorch, Kubernetes, Redis, PostgreSQL

Previous Role: Software Engineer at DataSystems LLC (2017-2020)
- Built data pipelines processing 500TB of data daily
- Implemented machine learning models for fraud detection with 99.2% accuracy
- Reduced processing time by 60% through Spark optimization
- Led migration from on-premise to AWS cloud infrastructure
- Technologies: Scala, Spark, Kafka, AWS, Terraform

Technical Skills:
Programming Languages:
- Python (Expert): 8 years experience, primary language for ML work
- Java (Advanced): 5 years, used for backend microservices
- Scala (Intermediate): 3 years, big data processing with Spark
- C++ (Intermediate): CUDA kernels and performance-critical code
- Rust (Learning): Interested in systems programming

Frameworks and Libraries:
- PyTorch: Deep learning, custom model architectures, distributed training
- TensorFlow: Production deployment, TensorFlow Serving, TFLite
- FastAPI: Building high-performance REST APIs
- React: Frontend development with TypeScript
- Apache Spark: Large-scale data processing, MLlib
- Kubernetes: Container orchestration, Helm charts, Istio service mesh

Projects:
Project 1: Real-time Recommendation Engine
- Built recommendation system serving 50M users globally
- Reduced latency from 200ms to 15ms using FAISS indexing
- A/B testing showed 23% increase in user engagement
- Handles 100K QPS with 99.99% uptime

Project 2: Automated Code Review System
- ML model to detect bugs and suggest improvements
- Trained on 10M code snippets from internal repositories
- Catches 40% of bugs before human review
- Reduced code review time by 30%

Preferences:
- Prefers morning meetings (9-11am), deep coding in afternoon
- Editor: VS Code with Vim keybindings and custom extensions
- Terminal: iTerm2 with zsh, tmux for session management
- OS: macOS for development, Ubuntu for servers

Hobbies:
- Rock climbing: Bouldering V5-V6, lead climbing 5.11a
- Photography: Landscape and street, Canon R5
- Reading: Sci-fi (Asimov, Clarke) and technical books
- Cooking: Italian and Japanese cuisine, makes fresh pasta

Goals:
Short-term (1 year):
- Lead a major ML infrastructure project end-to-end
- Publish a paper at NeurIPS or ICML
- Get promoted to Staff Engineer

Long-term (5 years):
- Become VP of Engineering or CTO at a startup
- Start a technical YouTube channel with 100K subscribers
- Write a book on ML systems design
"""

# Multiply to get longer context
MULTIPLIER = 6
long_document = "You are a helpful AI assistant. Below is detailed information about a user.\n"
for i in range(MULTIPLIER):
    long_document += f"\n--- SECTION {i+1} ---\n"
    long_document += doc_chunk

long_document += "\n\n=== END OF DOCUMENT ===\n\nBased on ALL sections above, answer the user's question.\n\nUser: "

# Count approximate tokens
approx_tokens = len(long_document.split()) * 1.3
print(f"\nDocument size: ~{int(approx_tokens)} tokens ({MULTIPLIER} sections)")

# Use 4-bit AWQ quantized model for T4
model_name = "Qwen/Qwen2-7B-Instruct-AWQ"
print(f"Model: {model_name} (4-bit quantized)")

print("\nLoading model...")
start_load = time.time()

llm = LLM(
    model=model_name,
    max_model_len=8192,
    gpu_memory_utilization=0.85,
    enable_prefix_caching=True,
    quantization="awq",
    trust_remote_code=True,
)

load_time = time.time() - start_load
print(f"Model loaded in {load_time:.1f}s\n")

sampling = SamplingParams(max_tokens=50, temperature=0.3)

questions = [
    "What is Alice's current job title and company?",
    "What programming language is Alice most skilled in?",
    "What was the latency improvement in the recommendation project?",
    "What editor and terminal does Alice use?",
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
    print(f"A: {out.strip()[:100]}...")
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
    print(f"A: {out.strip()[:100]}...")
    print(f"Time: {elapsed*1000:.1f}ms")

print("\n" + "="*60)
print("ROUND 3: New questions (PREFIX CACHE HIT)")
print("="*60)

new_questions = [
    "Where did Alice go to school?",
    "What are Alice's short-term career goals?",
    "What hobbies does Alice have?",
    "What cloud platforms has Alice used?",
]

times_round3 = []
for q in new_questions:
    prompt = long_document + q
    start = time.time()
    out = llm.generate([prompt], sampling)[0].outputs[0].text
    elapsed = time.time() - start
    times_round3.append(elapsed)
    print(f"\nQ: {q}")
    print(f"A: {out.strip()[:100]}...")
    print(f"Time: {elapsed*1000:.1f}ms")

print("\n" + "="*60)
print("RESULTS")
print("="*60)

avg_r1 = sum(times_round1) / len(times_round1) * 1000
avg_r2 = sum(times_round2) / len(times_round2) * 1000
avg_r3 = sum(times_round3) / len(times_round3) * 1000

print(f"\nModel: {model_name}")
print(f"Document: ~{int(approx_tokens)} tokens")
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
print("COMPARISON: 0.5B vs 7B")
print("="*60)
print(f"""
Results comparison:

                    Qwen2-0.5B       Qwen2-7B-AWQ
                    (24 layers)      (28 layers, 4-bit)
------------------------------------------------------------
~2600 tokens
  Cache miss        84ms             {avg_r1:.0f}ms
  Cache hit         21ms             {avg_r3:.0f}ms
  Speedup           4x               {speedup_r3:.1f}x

Larger models benefit MORE from KV caching because:
- More layers = more attention computation saved
- More heads = more KV to cache/reuse
""")
