#!/usr/bin/env python3
"""
Selective KV Cache Loading API Test - Using Real Block Hashes

This test uses the /cache/stored_hashes endpoint to get real block hashes
computed by LMCache, then uses those for selective loading.

REQUIRES: vLLM server with LMCache and kv_events enabled

To start server:
    LMCACHE_ENABLE_KV_EVENTS=true \
    LMCACHE_CHUNK_SIZE=256 \
    LMCACHE_LOCAL_CPU=True \
    vllm serve cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit \
        --gpu-memory-utilization 0.85 --max-model-len 4096 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

Then run this test:
    python test_selective_api.py
"""

import time
import requests
from typing import List, Dict, Any, Tuple, Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("ERROR: openai package required. Install with: pip install openai")
    exit(1)

SERVER_URL = "http://localhost:8000"
OPENAI_URL = f"{SERVER_URL}/v1"
LMCACHE_URL = SERVER_URL  # Internal API on same port

print("="*70)
print("SELECTIVE KV CACHE LOADING - REAL HASH TEST")
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


def get_stored_hashes() -> Optional[Dict]:
    """Get stored block hashes from LMCache internal API."""
    try:
        resp = requests.get(f"{LMCACHE_URL}/cache/stored_hashes", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"  Error getting hashes: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        print(f"  Error getting hashes: {e}")
        return None


def make_request(
    client: OpenAI,
    messages: List[Dict],
    selective_params: Dict[str, Any] = None,
    label: str = ""
) -> Tuple[str, float]:
    """Make a request with optional selective loading params."""

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
        print(f"  Selective hashes: {selective_params.get('lmcache.selective_hashes', [])[:3]}...")
        print(f"  Selective offsets: {selective_params.get('lmcache.selective_offsets', [])}")

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
    print(f"\nConnecting to: {OPENAI_URL}")
    client = OpenAI(base_url=OPENAI_URL, api_key="dummy")

    try:
        models = client.models.list()
        print(f"Server ready. Models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        return

    times = {}

    # =========================================================================
    # TEST 1: Full context - Store all KV
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: FULL CONTEXT (store all KV blocks)")
    print("="*70)

    full_messages = [system_msg] + old_msgs + new_msgs + [
        {"role": "user", "content": "What car do I have?"}
    ]

    out1, times['full_store'] = make_request(
        client, full_messages,
        label="Request 1: Store full context"
    )

    # Get stored hashes
    print("\n  Fetching stored hashes...")
    hashes_data = get_stored_hashes()

    if hashes_data and hashes_data.get("events"):
        events = hashes_data["events"]
        print(f"  Got {len(events)} store events")

        all_hashes = []
        all_offsets = []
        for event in events:
            block_hashes = event.get("block_hashes", [])
            block_size = event.get("block_size", 256)
            all_hashes.extend(block_hashes)
            all_offsets.extend([block_size] * len(block_hashes))

        print(f"  Total blocks: {len(all_hashes)}")
        print(f"  Block hashes (first 5): {all_hashes[:5]}")
        print(f"  Block size: {events[0].get('block_size') if events else 'N/A'}")
    else:
        print("  WARNING: No hashes returned!")
        print("  Make sure LMCACHE_ENABLE_KV_EVENTS=true")
        if hashes_data:
            print(f"  Response: {hashes_data}")
        all_hashes = []
        all_offsets = []

    # =========================================================================
    # TEST 2: Full context again - Cache hit
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: FULL CONTEXT AGAIN (cache hit)")
    print("="*70)

    full_messages_q2 = [system_msg] + old_msgs + new_msgs + [
        {"role": "user", "content": "What is my name?"}
    ]
    out2, times['full_hit'] = make_request(
        client, full_messages_q2,
        label="Request 2: Full context cache hit"
    )

    # =========================================================================
    # TEST 3: Selective loading with real hashes
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 3: SELECTIVE LOADING (using real hashes)")
    print("="*70)

    if all_hashes:
        # Use only some of the hashes (skip middle blocks)
        # Take first half and last quarter to simulate selective loading
        num_blocks = len(all_hashes)
        if num_blocks >= 4:
            # Skip some middle blocks
            selected_indices = list(range(num_blocks // 4)) + list(range(3 * num_blocks // 4, num_blocks))
            selected_hashes = [all_hashes[i] for i in selected_indices]
            selected_offsets = [all_offsets[i] for i in selected_indices]
        else:
            selected_hashes = all_hashes
            selected_offsets = all_offsets

        selective_params = {
            "lmcache.selective_hashes": selected_hashes,
            "lmcache.selective_offsets": selected_offsets,
        }

        # Use same messages but with selective params
        selective_messages = [system_msg] + new_msgs + [
            {"role": "user", "content": "What car do I have?"}
        ]

        out3, times['selective'] = make_request(
            client,
            selective_messages,
            selective_params=selective_params,
            label="Request 3: Selective loading with real hashes"
        )
    else:
        print("  SKIPPED - No hashes available")
        out3 = ""
        times['selective'] = 0

    # =========================================================================
    # TEST 4: Fresh compute (no selective params, same shorter context)
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 4: FRESH COMPUTE (same context, no selective params)")
    print("="*70)

    fresh_messages = [system_msg] + new_msgs + [
        {"role": "user", "content": "What car do I have?"}
    ]

    out4, times['fresh'] = make_request(
        client,
        fresh_messages,
        selective_params=None,
        label="Request 4: Fresh compute (no cache params)"
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
1. Full context (store):          {times['full_store']:.3f}s   Compute + store all
2. Full context (hit):            {times['full_hit']:.3f}s   Load from cache
3. Selective (real hashes):       {times['selective']:.3f}s   Load subset
4. Fresh compute:                 {times['fresh']:.3f}s   No cache params

ANALYSIS:
─────────────────────────────────────────────────────────────────""")

    if times['full_hit'] < times['full_store'] * 0.5:
        speedup = times['full_store'] / times['full_hit']
        print(f"[OK] Full cache HIT faster than MISS: {speedup:.2f}x speedup")

    if times['selective'] > 0 and times['selective'] < times['fresh'] * 0.8:
        speedup = times['fresh'] / times['selective']
        print(f"[OK] Selective faster than fresh: {speedup:.2f}x speedup")
        print("     → LMCache selective loading with real hashes WORKS!")
    elif times['selective'] > 0:
        print(f"[??] Selective ({times['selective']:.3f}s) similar to fresh ({times['fresh']:.3f}s)")
        print("     → Check if hashes match stored blocks")

    print(f"""
QUALITY CHECK:
─────────────────────────────────────────────────────────────────
Full context:
  Knows Tesla: {'YES' if 'tesla' in out1.lower() else 'NO'}
  Knows Bob:   {'YES' if 'bob' in out2.lower() else 'NO'}

Selective (new content only):
  Knows Tesla: {'YES' if 'tesla' in out3.lower() else 'NO'}
""")

    print("="*70)


if __name__ == "__main__":
    main()
