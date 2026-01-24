# SPDX-License-Identifier: Apache-2.0
"""
Test selective KV cache loading - the ability to load specific blocks
by hash/UUID without requiring all prefix blocks to be present.

This tests the modifications made to enable non-prefix cache hits:
1. Content-only hashing (no prefix chain)
2. UUID-based retrieval via hashes parameter
3. Skip missing blocks instead of invalidating all following blocks
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.token_database import ChunkedTokenDatabase

# Local
from .utils import dumb_metadata, generate_tokens


class TestContentOnlyHashing:
    """Test that hashing is now content-only (no prefix chain)"""

    def test_same_content_same_hash(self):
        """Same token content should produce same hash regardless of position"""
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=256, backend="cpu")
        metadata = dumb_metadata()
        db = ChunkedTokenDatabase(cfg, metadata)

        # Generate same tokens
        tokens = generate_tokens(256, "cpu", fixed=True)

        # Process tokens alone
        results1 = list(db.process_tokens(tokens=tokens))
        hash1 = results1[0][2].chunk_hash

        # Process same tokens again (should get same hash with content-only)
        results2 = list(db.process_tokens(tokens=tokens))
        hash2 = results2[0][2].chunk_hash

        assert hash1 == hash2, "Same content should produce same hash"

    def test_different_prefix_same_content_same_hash(self):
        """
        With content-only hashing, the same chunk should have the same hash
        even if preceded by different prefixes.

        OLD behavior: hash(chunk2) depends on hash(chunk1)
        NEW behavior: hash(chunk2) only depends on chunk2 content
        """
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=256, backend="cpu")
        metadata = dumb_metadata()
        db = ChunkedTokenDatabase(cfg, metadata)

        # Chunk that we want to test
        target_chunk = generate_tokens(256, "cpu", fixed=True)

        # Different prefixes
        prefix_a = torch.randint(0, 10000, size=[256]).to("cpu")
        prefix_b = torch.randint(0, 10000, size=[256]).to("cpu")

        # Combine with different prefixes
        tokens_a = torch.cat([prefix_a, target_chunk])
        tokens_b = torch.cat([prefix_b, target_chunk])

        results_a = list(db.process_tokens(tokens=tokens_a))
        results_b = list(db.process_tokens(tokens=tokens_b))

        # Get hash of second chunk (index 1)
        hash_a_chunk2 = results_a[1][2].chunk_hash
        hash_b_chunk2 = results_b[1][2].chunk_hash

        # With content-only hashing, these should be equal
        assert hash_a_chunk2 == hash_b_chunk2, (
            "Same chunk content should have same hash regardless of prefix"
        )


class TestHashBasedRetrieval:
    """Test that we can process tokens by hash instead of content"""

    def test_process_by_hashes(self):
        """Should be able to process using pre-computed hashes"""
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=256, backend="cpu")
        metadata = dumb_metadata()
        db = ChunkedTokenDatabase(cfg, metadata)

        # Create UUIDs (as ints)
        uuid1 = hash("block-uuid-1")
        uuid2 = hash("block-uuid-2")
        uuid3 = hash("block-uuid-3")

        # Process by hashes instead of tokens
        results = list(db.process_tokens(
            hashes=[uuid1, uuid2, uuid3],
            offsets=[256, 256, 256]
        ))

        assert len(results) == 3

        # Check positions
        assert results[0][0] == 0    # start
        assert results[0][1] == 256  # end
        assert results[1][0] == 256
        assert results[1][1] == 512
        assert results[2][0] == 512
        assert results[2][1] == 768

        # Check hashes match what we passed
        assert results[0][2].chunk_hash == uuid1
        assert results[1][2].chunk_hash == uuid2
        assert results[2][2].chunk_hash == uuid3

    def test_selective_hashes(self):
        """
        Should be able to request specific UUIDs only,
        enabling selective block loading.
        """
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=256, backend="cpu")
        metadata = dumb_metadata()
        db = ChunkedTokenDatabase(cfg, metadata)

        # Only request block 1 and 3, skip block 2
        uuid1 = hash("block-1")
        uuid3 = hash("block-3")

        results = list(db.process_tokens(
            hashes=[uuid1, uuid3],
            offsets=[256, 256]
        ))

        assert len(results) == 2
        assert results[0][2].chunk_hash == uuid1
        assert results[1][2].chunk_hash == uuid3


class TestVariableBlockSizes:
    """Test with different block sizes"""

    def test_different_offsets(self):
        """Blocks can have different sizes via offsets"""
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=256, backend="cpu")
        metadata = dumb_metadata()
        db = ChunkedTokenDatabase(cfg, metadata)

        uuid1 = hash("small-block")
        uuid2 = hash("large-block")

        # Different sized blocks
        results = list(db.process_tokens(
            hashes=[uuid1, uuid2],
            offsets=[128, 512]  # Different sizes
        ))

        assert len(results) == 2
        assert results[0][0] == 0
        assert results[0][1] == 128  # Small block
        assert results[1][0] == 128
        assert results[1][1] == 640  # Large block (128 + 512)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
