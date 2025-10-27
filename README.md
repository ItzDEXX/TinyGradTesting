tinygrad-vit-mlp-fusion (proof of work)

Goal: show that we can make the MLP part of a tiny ViT block faster (same answers, less time) by writing it in a fused style and JIT-compiling it in tinygrad.

What this repo demonstrates

A baseline MLP: Linear → +bias → GELU → Linear → +bias

A fused MLP: (x @ W1 + b1).gelu() @ W2 + b2 (fewer kernels; JIT cached)

Optional single-head attention to form a mini ViT block: Attention → MLP

Benchmarks that compare latency (p50) for baseline vs fused, with identical outputs
