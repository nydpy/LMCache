#!/usr/bin/env python3
"""
Selective KV Cache Loading Tests

Tests the core modifications to enable non-prefix cache hits:
1. Content-only hashing (no prefix chain)
2. UUID-based retrieval via hashes parameter
3. Skip missing blocks instead of invalidating all following blocks

Run: python test_selective_simple.py
Or:  ./run_selective_tests.sh
"""

import sys
sys.path.insert(0, '/Users/youndukn/projects/memory-machine/vendor/LMCache')

import torch

# Mock the vLLM imports that token_database needs
class MockHashModule:
    @staticmethod
    def get_hash_fn_by_name(name):
        return hash

sys.modules['vllm'] = type(sys)('vllm')
sys.modules['vllm.utils'] = MockHashModule
sys.modules['vllm.utils.hashing'] = MockHashModule
sys.modules['vllm.v1'] = type(sys)('vllm.v1')
sys.modules['vllm.v1.core'] = type(sys)('vllm.v1.core')
sys.modules['vllm.v1.core.kv_cache_utils'] = type(sys)('vllm.v1.core.kv_cache_utils')

# Now import our modified code
from lmcache.v1.token_database import ChunkedTokenDatabase


def test_content_only_hashing():
    """
    Test that same content produces same hash, regardless of prefix.

    OLD behavior (prefix chain): hash(chunk2) = f(chunk2, hash(chunk1))
    NEW behavior (content only): hash(chunk2) = f(chunk2)
    """
    print("\n" + "="*60)
    print("TEST: Content-Only Hashing")
    print("="*60)

    # Create database without config/metadata (use defaults)
    db = ChunkedTokenDatabase()
    db.chunk_size = 256
    db.save_unfull_chunk = True

    # Same target chunk
    target_chunk = torch.tensor([100] * 256)

    # Different prefixes
    prefix_a = torch.tensor([1] * 256)
    prefix_b = torch.tensor([2] * 256)

    tokens_a = torch.cat([prefix_a, target_chunk])
    tokens_b = torch.cat([prefix_b, target_chunk])

    # Process both
    results_a = list(db.process_tokens(tokens=tokens_a, make_key=False))
    results_b = list(db.process_tokens(tokens=tokens_b, make_key=False))

    print(f"Prefix A tokens: {prefix_a[:5].tolist()}...")
    print(f"Prefix B tokens: {prefix_b[:5].tolist()}...")
    print(f"Target chunk: {target_chunk[:5].tolist()}...")
    print()

    # Get second chunk hash from each
    hash_a_chunk1 = results_a[0][2]
    hash_a_chunk2 = results_a[1][2]
    hash_b_chunk1 = results_b[0][2]
    hash_b_chunk2 = results_b[1][2]

    print(f"Hash of chunk 1 (prefix A): {hash_a_chunk1}")
    print(f"Hash of chunk 1 (prefix B): {hash_b_chunk1}")
    print(f"Hash of chunk 2 (prefix A): {hash_a_chunk2}")
    print(f"Hash of chunk 2 (prefix B): {hash_b_chunk2}")
    print()

    # With content-only hashing, chunk 2 hashes should be EQUAL
    if hash_a_chunk2 == hash_b_chunk2:
        print("PASS: Same content -> Same hash (content-only hashing works!)")
        return True
    else:
        print("FAIL: Different hashes for same content (still using prefix chain)")
        return False


def test_hash_based_retrieval():
    """
    Test that we can process using pre-computed hashes/UUIDs.
    """
    print("\n" + "="*60)
    print("TEST: Hash/UUID-Based Retrieval")
    print("="*60)

    db = ChunkedTokenDatabase()
    db.chunk_size = 256
    db.save_unfull_chunk = True

    # Create UUIDs (as ints)
    uuid1 = hash("conversation-turn-1")
    uuid2 = hash("conversation-turn-2")
    uuid3 = hash("conversation-turn-3")

    print(f"UUID 1: {uuid1}")
    print(f"UUID 2: {uuid2}")
    print(f"UUID 3: {uuid3}")
    print()

    # Process by hashes instead of tokens
    results = list(db.process_tokens(
        hashes=[uuid1, uuid2, uuid3],
        offsets=[256, 256, 256],
        make_key=False
    ))

    print(f"Number of results: {len(results)}")
    for i, (start, end, h) in enumerate(results):
        print(f"  Block {i}: start={start}, end={end}, hash={h}")

    # Verify
    success = True

    if len(results) != 3:
        print(f"FAIL: Expected 3 results, got {len(results)}")
        success = False

    if results[0][2] != uuid1 or results[1][2] != uuid2 or results[2][2] != uuid3:
        print("FAIL: Returned hashes don't match input UUIDs")
        success = False

    if results[0][0] != 0 or results[0][1] != 256:
        print("FAIL: Block 0 position wrong")
        success = False

    if results[1][0] != 256 or results[1][1] != 512:
        print("FAIL: Block 1 position wrong")
        success = False

    if results[2][0] != 512 or results[2][1] != 768:
        print("FAIL: Block 2 position wrong")
        success = False

    if success:
        print("\nPASS: Hash/UUID-based retrieval works!")

    return success


def test_selective_blocks():
    """
    Test loading only specific blocks (skip some in the middle).
    """
    print("\n" + "="*60)
    print("TEST: Selective Block Loading")
    print("="*60)

    db = ChunkedTokenDatabase()
    db.chunk_size = 256
    db.save_unfull_chunk = True

    # Request only blocks 1 and 3, skip block 2
    uuid1 = hash("block-1")
    uuid3 = hash("block-3")

    print(f"Requesting only UUID 1 and UUID 3 (skipping 2)")
    print(f"  UUID 1: {uuid1}")
    print(f"  UUID 3: {uuid3}")
    print()

    results = list(db.process_tokens(
        hashes=[uuid1, uuid3],
        offsets=[256, 256],
        make_key=False
    ))

    print(f"Number of results: {len(results)}")
    for i, (start, end, h) in enumerate(results):
        print(f"  Block {i}: start={start}, end={end}, hash={h}")

    success = True

    if len(results) != 2:
        print(f"FAIL: Expected 2 results, got {len(results)}")
        success = False

    if results[0][2] != uuid1:
        print("FAIL: First result should be uuid1")
        success = False

    if results[1][2] != uuid3:
        print("FAIL: Second result should be uuid3")
        success = False

    if success:
        print("\nPASS: Selective block loading works!")

    return success


def test_variable_offsets():
    """
    Test blocks with different sizes.
    """
    print("\n" + "="*60)
    print("TEST: Variable Block Sizes")
    print("="*60)

    db = ChunkedTokenDatabase()
    db.chunk_size = 256
    db.save_unfull_chunk = True

    uuid1 = hash("small-block")
    uuid2 = hash("large-block")

    print(f"Block 1: 128 tokens")
    print(f"Block 2: 512 tokens")
    print()

    results = list(db.process_tokens(
        hashes=[uuid1, uuid2],
        offsets=[128, 512],
        make_key=False
    ))

    print(f"Number of results: {len(results)}")
    for i, (start, end, h) in enumerate(results):
        print(f"  Block {i}: start={start}, end={end}, size={end-start}")

    success = True

    if len(results) != 2:
        print(f"FAIL: Expected 2 results, got {len(results)}")
        success = False

    if results[0][1] - results[0][0] != 128:
        print("FAIL: Block 0 should be 128 tokens")
        success = False

    if results[1][1] - results[1][0] != 512:
        print("FAIL: Block 1 should be 512 tokens")
        success = False

    if results[1][0] != 128:
        print("FAIL: Block 1 should start at 128")
        success = False

    if success:
        print("\nPASS: Variable block sizes work!")

    return success


def test_contiguous_mapping_simulation():
    """
    Simulate how vLLM's contiguous slot mapping would work.

    If we have blocks at original positions 0, 512, 1024
    but only load blocks 0 and 1024 (skip 512),
    with Option A they get placed at positions 0, 256 (contiguous).
    """
    print("\n" + "="*60)
    print("TEST: Contiguous Mapping Simulation (Option A)")
    print("="*60)

    db = ChunkedTokenDatabase()
    db.chunk_size = 256
    db.save_unfull_chunk = True

    # Simulate: we have 4 blocks, but only want to load blocks 0 and 2
    uuid0 = hash("block-0")
    uuid2 = hash("block-2")

    print("Original positions: Block 0 @ 0-256, Block 2 @ 512-768")
    print("With contiguous mapping: Block 0 @ 0-256, Block 2 @ 256-512")
    print()

    # Get blocks with their hashes
    results = list(db.process_tokens(
        hashes=[uuid0, uuid2],
        offsets=[256, 256],
        make_key=False
    ))

    # Simulate slot_mapping creation
    total_tokens = sum([256, 256])
    slot_mapping = list(range(total_tokens))  # [0, 1, 2, ..., 511]

    print(f"Total tokens to load: {total_tokens}")
    print(f"Contiguous slot_mapping: [0..{total_tokens-1}]")
    print()

    # In vLLM, this would be:
    # slot_mapping[:256] -> block 0's KV data
    # slot_mapping[256:512] -> block 2's KV data (placed contiguously!)

    success = True

    if len(results) == 2 and results[0][0] == 0 and results[1][0] == 256:
        print("PASS: Blocks correctly positioned for contiguous mapping")
    else:
        print("FAIL: Block positions incorrect")
        success = False

    return success


if __name__ == "__main__":
    print("="*60)
    print("SELECTIVE KV CACHE LOADING TESTS")
    print("="*60)

    results = []

    results.append(("Content-Only Hashing", test_content_only_hashing()))
    results.append(("Hash-Based Retrieval", test_hash_based_retrieval()))
    results.append(("Selective Blocks", test_selective_blocks()))
    results.append(("Variable Block Sizes", test_variable_offsets()))
    results.append(("Contiguous Mapping", test_contiguous_mapping_simulation()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
