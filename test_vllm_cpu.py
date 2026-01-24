#!/usr/bin/env python3
"""
Test vLLM on CPU with our modified LMCache.

Run with: VLLM_TARGET_DEVICE=cpu python test_vllm_cpu.py
"""

import os
os.environ["VLLM_TARGET_DEVICE"] = "cpu"

from vllm import LLM, SamplingParams

def main():
    print("Loading Qwen2-0.5B on CPU...")

    llm = LLM(
        model="Qwen/Qwen2-0.5B",
        dtype="float32",
        max_model_len=512,
        enforce_eager=True,
    )

    print("Model loaded! Running inference...")

    # Test prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=50,
    )

    outputs = llm.generate(prompts, sampling_params)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
