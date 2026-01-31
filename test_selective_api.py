#!/usr/bin/env python3
"""
Selective KV Cache Loading API Test

This test uses the LMCache selective loading API via vLLM server.

REQUIRES: vLLM server running with LMCache enabled

To start server:
    vllm serve cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.85

Then run this test:
    python test_selective_api.py

Test flow:
1. Send full context [system][old][new] - stores all blocks in cache
2. Send selective request with only [system][new] hashes - loads from cache
3. Verify: knows new content, doesn't know old content
"""

import hashlib
import time
from typing import List, Dict, Any

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("ERROR: openai package required. Install with: pip install openai")
    exit(1)

# Server URL - adjust if different
SERVER_URL = "http://localhost:8000/v1"

print("="*70)
print("SELECTIVE KV CACHE LOADING API TEST")
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
) -> str:
    """Make a request with optional selective loading params"""

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

    start = time.time()
    try:
        response = client.chat.completions.create(**kwargs)
        elapsed = time.time() - start
        content = response.choices[0].message.content
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Response: {content[:60]}...")
        return content
    except Exception as e:
        print(f"  ERROR: {e}")
        return ""


def main():
    print(f"\nConnecting to: {SERVER_URL}")
    client = OpenAI(base_url=SERVER_URL, api_key="dummy")

    # Test connection
    try:
        models = client.models.list()
        print(f"Server ready. Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        print("\nMake sure vLLM server is running:")
        print("  vllm serve cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit --enable-prefix-caching")
        return

    print("\n" + "="*70)
    print("TEST 1: FULL CONTEXT (cache all blocks)")
    print("="*70)

    full_messages = [system_msg] + old_msgs + new_msgs + [
        {"role": "user", "content": "What car do I have?"}
    ]

    # First request - stores all KV in cache
    out1 = make_request(client, full_messages, label="Request 1 (cache miss - compute all)")

    print("\n" + "="*70)
    print("TEST 2: FULL CONTEXT AGAIN (cache hit)")
    print("="*70)

    full_messages_q2 = [system_msg] + old_msgs + new_msgs + [
        {"role": "user", "content": "What is my name?"}
    ]
    out2 = make_request(client, full_messages_q2, label="Request 2 (cache hit)")

    print("\n" + "="*70)
    print("TEST 3: SELECTIVE LOADING (skip old blocks)")
    print("="*70)

    # Build selective params - only system + new, skip old
    selective_messages = [system_msg] + new_msgs
    selective_params = build_selective_params(selective_messages)

    selective_messages_q = selective_messages + [
        {"role": "user", "content": "What car do I have?"}
    ]
    out3 = make_request(
        client,
        selective_messages_q,
        selective_params=selective_params,
        label="Request 3 (selective - skip old)"
    )

    print("\n" + "="*70)
    print("TEST 4: SELECTIVE - Ask about OLD content")
    print("="*70)

    selective_messages_q2 = selective_messages + [
        {"role": "user", "content": "What is my name?"}
    ]
    out4 = make_request(
        client,
        selective_messages_q2,
        selective_params=selective_params,
        label="Request 4 (selective - ask about old)"
    )

    print("\n" + "="*70)
    print("TEST 5: FRESH COMPUTE (same 6 messages, NO selective params)")
    print("="*70)

    # Same messages as selective, but WITHOUT selective_hashes
    # This forces fresh KV computation
    selective_messages_fresh = selective_messages + [
        {"role": "user", "content": "What car do I have?"}
    ]
    out5 = make_request(
        client,
        selective_messages_fresh,
        selective_params=None,  # No selective loading - fresh compute
        label="Request 5 (fresh compute - no cache)"
    )

    print("\n" + "="*70)
    print("TEST 6: SELECTIVE AGAIN (should be faster if cache works)")
    print("="*70)

    out6 = make_request(
        client,
        selective_messages_q,
        selective_params=selective_params,
        label="Request 6 (selective - second time)"
    )

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    knows_tesla_full = "tesla" in out1.lower()
    knows_bob_full = "bob" in out2.lower()
    knows_tesla_sel = "tesla" in out3.lower()
    knows_bob_sel = "bob" in out4.lower()

    print(f"""
                          Full Context    Selective
                          ────────────    ─────────
Knows Tesla (new):        {'YES' if knows_tesla_full else 'NO':^12}    {'YES' if knows_tesla_sel else 'NO':^9}
Knows Bob (old):          {'YES' if knows_bob_full else 'NO':^12}    {'YES' if knows_bob_sel else 'NO':^9} (want NO)
""")

    if knows_tesla_full and knows_tesla_sel:
        print("[OK] NEW content accessible in both modes")

    if knows_bob_full and not knows_bob_sel:
        print("[OK] OLD content correctly excluded in selective mode")
    elif knows_bob_sel:
        print("[FAIL] Model knows Bob in selective mode (should not)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
