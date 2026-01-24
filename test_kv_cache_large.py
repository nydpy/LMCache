#!/usr/bin/env python3
"""
Large context KV cache test - tests with ~2000+ tokens.

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

# Generate a long document (~2000 tokens)
long_document = """You are a helpful AI assistant. Below is a detailed document about a user.

=== USER PROFILE ===

Name: Alice Johnson
Age: 32
Location: San Francisco, California
Occupation: Senior Software Engineer at TechCorp Inc.
Education: MS Computer Science from Stanford University

=== WORK HISTORY ===

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

Internship: Google (Summer 2016)
- Worked on the TensorFlow team
- Contributed to distributed training features
- Published internal paper on gradient compression

=== TECHNICAL SKILLS ===

Programming Languages:
- Python (Expert): 8 years experience, primary language for ML work
- Java (Advanced): 5 years, used for backend services
- Scala (Intermediate): 3 years, big data processing
- JavaScript/TypeScript (Intermediate): Frontend and Node.js
- Rust (Learning): Interested in systems programming
- Go (Basic): Some microservices work

Frameworks and Libraries:
- PyTorch: Deep learning, custom model architectures
- TensorFlow: Production deployment, TensorFlow Serving
- FastAPI: Building REST APIs
- React: Frontend development
- Apache Spark: Large-scale data processing
- Kubernetes: Container orchestration, Helm charts

Databases:
- PostgreSQL: Primary relational database
- Redis: Caching, session management
- MongoDB: Document storage
- Elasticsearch: Search and analytics
- Apache Cassandra: High-throughput writes

Cloud Platforms:
- AWS: EC2, S3, Lambda, SageMaker, EKS
- GCP: Compute Engine, BigQuery, Vertex AI
- Azure: Basic experience with Azure ML

=== PROJECTS ===

Project 1: Real-time Recommendation Engine
- Built recommendation system serving 50M users
- Reduced latency from 200ms to 15ms
- A/B testing showed 23% increase in engagement
- Tech: PyTorch, FAISS, Redis, Kubernetes

Project 2: Automated Code Review System
- ML model to detect bugs and suggest improvements
- Trained on 10M code snippets from internal repos
- Catches 40% of bugs before human review
- Tech: Transformers, CodeBERT, Python

Project 3: Data Pipeline Optimization
- Redesigned ETL pipeline for analytics team
- Reduced daily processing time from 8 hours to 45 minutes
- Saved $50K/month in compute costs
- Tech: Spark, Airflow, Delta Lake

=== PREFERENCES ===

Work Style:
- Prefers morning meetings, coding in afternoon
- Likes detailed code reviews
- Values documentation and clean code
- Enjoys pair programming sessions

Communication:
- Prefers Slack for quick questions
- Email for detailed discussions
- Weekly 1:1s with manager
- Monthly team presentations

Development Environment:
- Editor: VS Code with Vim keybindings
- Terminal: iTerm2 with zsh
- Version Control: Git with conventional commits
- OS: macOS for development, Linux for servers

=== INTERESTS ===

Technical Interests:
- Large Language Models and their applications
- Efficient ML inference at scale
- Developer tools and productivity
- Open source contribution

Hobbies:
- Rock climbing (bouldering V5-V6)
- Photography (landscape and street)
- Reading (sci-fi and technical books)
- Cooking (Italian and Japanese cuisine)

=== GOALS ===

Short-term (1 year):
- Lead a major ML infrastructure project
- Publish a paper at a top ML conference
- Mentor 2-3 junior engineers to promotion

Long-term (5 years):
- Become a Staff Engineer or Engineering Manager
- Start a technical blog with 10K followers
- Contribute significantly to an open source ML project

=== CONTACT ===

Email: alice.johnson@techcorp.com
GitHub: github.com/alicejohnson
LinkedIn: linkedin.com/in/alicejohnson
Twitter: @alice_codes

=== END OF PROFILE ===

Based on the above profile, answer the user's question.

User: """

# Count approximate tokens
approx_tokens = len(long_document.split()) * 1.3  # rough estimate
print(f"\nDocument size: ~{int(approx_tokens)} tokens")

print("\nLoading model...")
llm = LLM(
    model="Qwen/Qwen2-0.5B",
    max_model_len=4096,
    gpu_memory_utilization=0.5,
    enable_prefix_caching=True,
)
print("Model ready!\n")

sampling = SamplingParams(max_tokens=50, temperature=0.3)

questions = [
    "What is Alice's current job title?",
    "What programming language is Alice most skilled in?",
    "What was the latency improvement in the recommendation engine project?",
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
    "What cloud platforms does Alice know?",
    "What is Alice's long-term goal?",
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
If you serve 1000 requests/day with this context:

Without cache: {avg_r1:.0f}ms × 1000 = {avg_r1:.0f} seconds total
With cache:    {avg_r2:.0f}ms × 1000 = {avg_r2:.0f} seconds total

Daily time saved: {(avg_r1-avg_r2):.0f} seconds
Monthly time saved: {(avg_r1-avg_r2)*30:.0f} seconds = {(avg_r1-avg_r2)*30/60:.1f} minutes
""")
