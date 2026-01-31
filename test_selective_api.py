#!/usr/bin/env python3
"""
Selective KV Cache Loading API Test

Tests to verify selective KV cache loading actually uses cached blocks.

REQUIRES: vLLM server running with LMCache enabled

To start server:
    LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=5 \
    vllm serve cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit \
        --gpu-memory-utilization 0.85 --max-model-len 4096 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

Then run this test:
    python test_selective_api.py
"""

import hashlib
import time
import random
from typing import List, Dict, Any, Tuple

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("ERROR: openai package required. Install with: pip install openai")
    exit(1)

SERVER_URL = "http://localhost:8000/v1"

print("="*70)
print("SELECTIVE KV CACHE LOADING - VERIFICATION TEST")
print("="*70)

# Message blocks
system_msg = {"role": "system", "content": "You are a helpful assistant with perfect memory."}

old_msgs = [
    {"role": "user", "content": "My name is Bob."},
    {"role": "assistant", "content": "Hello Bob! Nice to meet you."},
    {"role": "user", "content": "I work at Google."},
    {"role": "assistant", "content": "Google is a great company!"},
]

new_msgs = [
    {"role": "user", "content": "I just bought a Tesla Model Y."},
    {"role": "assistant", "content": "Congratulations on your new Tesla!"},
    {"role": "user", "content": "I live in Seattle."},
    {"role": "assistant", "content": "Seattle is a beautiful city!"},
]


def compute_msg_hash(msg: Dict) -> int:
    """Compute hash for a message (role:content)"""
    data = f"{msg['role']}:{msg['content']}"
    return int(hashlib.sha256(data.encode()).hexdigest()[:16], 16)


def estimate_tokens(msg: Dict) -> int:
    """Estimate token count for a message"""
    text = f"{msg['role']}: {msg['content']}"
    return max(256, ((len(text) // 4) // 256 + 1) * 256)


def build_selective_params(messages: List[Dict]) -> Dict[str, Any]:
    """Build kv_transfer_params for selective loading"""
    hashes = [compute_msg_hash(msg) for msg in messages]
    offsets = [estimate_tokens(msg) for msg in messages]
    return {
        "lmcache.selective_hashes": hashes,
        "lmcache.selective_offsets": offsets,
    }


def make_request(
    client: OpenAI,
    messages: List[Dict],
    selective_params: Dict[str, Any] = None,
    label: str = ""
) -> Tuple[str, float]:
    """Make a request with optional selective loading params. Returns (content, time)"""

    kwargs = {
        "model": "cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit",
        "messages": messages,
        "max_tokens": 30,
        "temperature": 0,
    }

    if selective_params:
        kwargs["extra_body"] = {"kv_transfer_params": selective_params}

    print(f"\n{label}")
    print(f"  Messages: {len(messages)}")
    if selective_params:
        print(f"  Selective hashes: {len(selective_params['lmcache.selective_hashes'])}")
        print(f"  Total cached tokens: {sum(selective_params['lmcache.selective_offsets'])}")
    else:
        print(f"  Selective params: None (fresh compute)")

    start = time.time()
    try:
        response = client.chat.completions.create(**kwargs)
        elapsed = time.time() - start
        content = response.choices[0].message.content
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Response: {content[:60]}...")
        return content, elapsed
    except Exception as e:
        print(f"  ERROR: {e}")
        return "", 0.0


def main():
    print(f"\nConnecting to: {SERVER_URL}")
    client = OpenAI(base_url=SERVER_URL, api_key="dummy")

    try:
        models = client.models.list()
        print(f"Server ready. Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        print("\nMake sure vLLM server is running with LMCache:")
        print('  LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True vllm serve ... --kv-transfer-config \'{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}\'')
        return

    times = {}

    # =========================================================================
    # TEST 1: SELECTIVE MISS (new content, not cached yet)
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: SELECTIVE MISS (content NOT in cache yet)")
    print("="*70)

    # Create unique messages that haven't been cached
    unique_id = random.randint(10000, 99999)
    uncached_msgs = [
        {"role": "user", "content": f"My favorite number is {unique_id}."},
        {"role": "assistant", "content": f"Interesting! {unique_id} is your favorite number."},
    ]

    uncached_messages = [system_msg] + uncached_msgs + [
        {"role": "user", "content": "What is my favorite number?"}
    ]
    uncached_params = build_selective_params([system_msg] + uncached_msgs)

    out1, times['selective_miss'] = make_request(
        client,
        uncached_messages,
        selective_params=uncached_params,
        label="Request 1: SELECTIVE MISS (blocks not cached)"
    )

    # =========================================================================
    # TEST 2: SELECTIVE HIT (same content, now cached)
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: SELECTIVE HIT (same content, should be cached now)")
    print("="*70)

    out2, times['selective_hit'] = make_request(
        client,
        uncached_messages,
        selective_params=uncached_params,
        label="Request 2: SELECTIVE HIT (blocks now cached)"
    )

    # =========================================================================
    # TEST 3: FRESH COMPUTE (same content, NO selective params)
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 3: FRESH COMPUTE (same content, no selective params)")
    print("="*70)

    out3, times['fresh_compute'] = make_request(
        client,
        uncached_messages,
        selective_params=None,  # No selective - just vLLM prefix cache
        label="Request 3: FRESH COMPUTE (no selective params)"
    )

    # =========================================================================
    # TEST 4: Full context to cache all blocks
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 4: FULL CONTEXT (cache all blocks for later)")
    print("="*70)

    full_messages = [system_msg] + old_msgs + new_msgs + [
        {"role": "user", "content": "What car do I have?"}
    ]

    out4, times['full_miss'] = make_request(
        client, full_messages,
        label="Request 4: FULL CONTEXT (cache miss)"
    )

    # =========================================================================
    # TEST 5: Full context hit
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 5: FULL CONTEXT HIT")
    print("="*70)

    full_messages_q2 = [system_msg] + old_msgs + new_msgs + [
        {"role": "user", "content": "What is my name?"}
    ]
    out5, times['full_hit'] = make_request(
        client, full_messages_q2,
        label="Request 5: FULL CONTEXT HIT"
    )

    # =========================================================================
    # TEST 6: Selective loading (skip old, use cached new)
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 6: SELECTIVE (skip old, load new from cache)")
    print("="*70)

    selective_messages = [system_msg] + new_msgs
    selective_params = build_selective_params(selective_messages)

    selective_q = selective_messages + [
        {"role": "user", "content": "What car do I have?"}
    ]
    out6, times['selective_skip'] = make_request(
        client,
        selective_q,
        selective_params=selective_params,
        label="Request 6: SELECTIVE (skip old blocks)"
    )

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("TIMING COMPARISON")
    print("="*70)

    print(f"""
Test                              Time      Notes
─────────────────────────────────────────────────────────────────
1. Selective MISS (not cached):   {times['selective_miss']:.3f}s   Compute + store
2. Selective HIT  (cached):       {times['selective_hit']:.3f}s   Load from cache
3. Fresh compute (no params):     {times['fresh_compute']:.3f}s   vLLM prefix cache only
4. Full context MISS:             {times['full_miss']:.3f}s   Compute all
5. Full context HIT:              {times['full_hit']:.3f}s   Load all
6. Selective (skip old):          {times['selective_skip']:.3f}s   Load subset

ANALYSIS:
─────────────────────────────────────────────────────────────────""")

    # Analysis
    if times['selective_hit'] < times['selective_miss'] * 0.8:
        print(f"[OK] Selective HIT faster than MISS: {times['selective_miss']/times['selective_hit']:.2f}x speedup")
        print("     → LMCache is loading cached KV blocks")
    else:
        print(f"[??] Selective HIT not much faster than MISS")
        print(f"     MISS: {times['selective_miss']:.3f}s, HIT: {times['selective_hit']:.3f}s")
        print("     → May not be using cached blocks")

    if times['selective_hit'] < times['fresh_compute'] * 0.8:
        print(f"[OK] Selective HIT faster than fresh: {times['fresh_compute']/times['selective_hit']:.2f}x speedup")
    else:
        print(f"[??] Selective HIT similar to fresh compute")

    if times['full_hit'] < times['full_miss'] * 0.5:
        print(f"[OK] Full HIT much faster than MISS: {times['full_miss']/times['full_hit']:.2f}x speedup")
        print("     → Prefix caching working")

    # Quality check
    print(f"""
QUALITY CHECK:
─────────────────────────────────────────────────────────────────
Knows favorite number (test 1-3): {unique_id}
  Selective MISS: {'YES' if str(unique_id) in out1 else 'NO'}
  Selective HIT:  {'YES' if str(unique_id) in out2 else 'NO'}
  Fresh compute:  {'YES' if str(unique_id) in out3 else 'NO'}

Full context tests:
  Knows Tesla: {'YES' if 'tesla' in out4.lower() else 'NO'}
  Knows Bob:   {'YES' if 'bob' in out5.lower() else 'NO'}
  Selective knows Tesla: {'YES' if 'tesla' in out6.lower() else 'NO'}
""")

    print("="*70)


if __name__ == "__main__":
    main()
