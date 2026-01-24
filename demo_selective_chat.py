#!/usr/bin/env python3
"""
Interactive Demo: Selective KV Cache Loading

This demo simulates a conversation where you can selectively load
cached message blocks. Run with a real vLLM server for actual inference,
or use simulation mode to see the concept.

Usage:
    python demo_selective_chat.py              # Simulation mode
    python demo_selective_chat.py --server URL # Real vLLM server
"""

import sys
import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Try to import openai client for real server mode
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class CachedMessage:
    """A message that has been cached as a KV block"""
    role: str
    content: str
    uuid: int
    token_count: int  # Simulated token count

    def __str__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"[{self.role}] {preview}"


class SelectiveCacheDemo:
    def __init__(self, server_url: Optional[str] = None):
        self.cached_messages: List[CachedMessage] = []
        self.server_url = server_url
        self.client = None

        if server_url and HAS_OPENAI:
            self.client = OpenAI(base_url=server_url, api_key="dummy")
            print(f"Connected to vLLM server: {server_url}")
        else:
            print("Running in SIMULATION mode (no real inference)")
        print()

    def _generate_uuid(self, content: str, role: str) -> int:
        """Generate a deterministic UUID for a message"""
        data = f"{role}:{content}"
        return int(hashlib.sha256(data.encode()).hexdigest()[:16], 16)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)"""
        return max(256, (len(text) // 4 // 256 + 1) * 256)

    def add_message(self, role: str, content: str):
        """Add a message to the cache"""
        uuid = self._generate_uuid(content, role)
        token_count = self._estimate_tokens(content)

        msg = CachedMessage(
            role=role,
            content=content,
            uuid=uuid,
            token_count=token_count
        )
        self.cached_messages.append(msg)
        print(f"Cached: [{role}] {content[:40]}... (UUID: {uuid}, ~{token_count} tokens)")

    def show_cache(self):
        """Display all cached messages"""
        print("\n" + "="*60)
        print("CACHED MESSAGES (select by number)")
        print("="*60)

        for i, msg in enumerate(self.cached_messages):
            print(f"  [{i}] {msg}")
            print(f"      UUID: {msg.uuid}, Tokens: {msg.token_count}")
        print()

    def select_messages(self) -> List[int]:
        """Let user select which messages to load"""
        self.show_cache()

        print("Enter message numbers to load (comma-separated)")
        print("Example: 0,2,4 (loads messages 0, 2, 4 - skips 1, 3)")
        print("Or 'all' for all messages, 'q' to quit")

        while True:
            selection = input("\nSelect> ").strip().lower()

            if selection == 'q':
                return None

            if selection == 'all':
                return list(range(len(self.cached_messages)))

            try:
                indices = [int(x.strip()) for x in selection.split(',')]
                if all(0 <= i < len(self.cached_messages) for i in indices):
                    return indices
                else:
                    print(f"Invalid index. Use 0-{len(self.cached_messages)-1}")
            except ValueError:
                print("Invalid input. Use comma-separated numbers.")

    def build_request(self, selected_indices: List[int], new_message: str) -> Dict:
        """Build the request with selective cache loading params"""

        selected_msgs = [self.cached_messages[i] for i in selected_indices]

        # Build the selective loading params
        selective_hashes = [msg.uuid for msg in selected_msgs]
        selective_offsets = [msg.token_count for msg in selected_msgs]

        # Build messages array
        messages = [{"role": msg.role, "content": msg.content} for msg in selected_msgs]
        messages.append({"role": "user", "content": new_message})

        request = {
            "model": "default",
            "messages": messages,
            "extra_body": {
                "kv_transfer_params": {
                    "lmcache.selective_hashes": selective_hashes,
                    "lmcache.selective_offsets": selective_offsets,
                }
            }
        }

        return request

    def generate(self, selected_indices: List[int], new_message: str) -> str:
        """Generate a response using selective cache loading"""

        request = self.build_request(selected_indices, new_message)

        print("\n" + "-"*60)
        print("REQUEST (with selective KV cache loading)")
        print("-"*60)
        print(f"Selected messages: {selected_indices}")
        print(f"Selective hashes: {request['extra_body']['kv_transfer_params']['lmcache.selective_hashes']}")
        print(f"Selective offsets: {request['extra_body']['kv_transfer_params']['lmcache.selective_offsets']}")
        print(f"Total cached tokens: {sum(request['extra_body']['kv_transfer_params']['lmcache.selective_offsets'])}")
        print()

        if self.client:
            # Real inference
            try:
                response = self.client.chat.completions.create(**request)
                return response.choices[0].message.content
            except Exception as e:
                return f"[Server error: {e}]"
        else:
            # Simulation
            selected_msgs = [self.cached_messages[i] for i in selected_indices]
            context = "\n".join([f"[{m.role}] {m.content[:30]}..." for m in selected_msgs])

            return f"""[SIMULATED RESPONSE]

Context loaded from cache:
{context}

New message: {new_message}

In a real scenario, the LLM would:
1. Load KV cache for selected messages ONLY (skipping others)
2. Use contiguous slot mapping (Option A)
3. Generate response with full context from selected blocks
"""

    def run_interactive(self):
        """Run the interactive demo"""
        print("="*60)
        print("SELECTIVE KV CACHE LOADING DEMO")
        print("="*60)
        print()
        print("This demo shows how to selectively load cached conversation")
        print("blocks instead of loading everything (prefix-only mode).")
        print()

        # Add some sample messages
        print("Adding sample conversation to cache...")
        print()

        self.add_message("system", "You are a helpful AI assistant.")
        self.add_message("user", "Hi, my name is Alice and I love coffee.")
        self.add_message("assistant", "Hello Alice! Nice to meet you. I'll remember that you love coffee.")
        self.add_message("user", "What's the weather like today?")
        self.add_message("assistant", "I don't have access to real-time weather data, but I can help you find weather information for your location.")
        self.add_message("user", "Actually, I prefer tea now.")
        self.add_message("assistant", "Got it! I'll update my notes - you now prefer tea over coffee.")

        print()
        print("-"*60)
        print("DEMO: Select which messages to load for next generation")
        print("-"*60)
        print()
        print("Traditional prefix caching: Must load ALL messages 0-6")
        print("Selective loading: Load ONLY the messages you need!")
        print()
        print("Example: Load only 0, 5, 6 to remember the tea preference")
        print("         without loading weather discussion (3, 4)")
        print()

        while True:
            indices = self.select_messages()
            if indices is None:
                print("Goodbye!")
                break

            new_msg = input("\nNew message> ").strip()
            if not new_msg:
                new_msg = "What do I like to drink?"

            response = self.generate(indices, new_msg)

            print("\n" + "="*60)
            print("RESPONSE")
            print("="*60)
            print(response)
            print()

            cont = input("\nContinue? (y/n) ").strip().lower()
            if cont != 'y':
                break


def main():
    server_url = None

    if len(sys.argv) > 2 and sys.argv[1] == '--server':
        server_url = sys.argv[2]

    demo = SelectiveCacheDemo(server_url)
    demo.run_interactive()


if __name__ == "__main__":
    main()
