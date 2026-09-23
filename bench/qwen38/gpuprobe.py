"""GB10 fast/slow state probe (method from tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark#1).

Decode-shaped bf16 GEMV, 6x5120 @ 5120x16384. Prints effective GB/s.
Fast state measured ~224-233 GB/s, slow state ~66-80 GB/s.
"""
import time, torch
a = torch.randn(6, 5120, dtype=torch.bfloat16, device="cuda")
w = torch.randn(5120, 16384, dtype=torch.bfloat16, device="cuda")
for _ in range(20):
    a @ w
torch.cuda.synchronize()
n = 200
t = time.time()
for _ in range(n):
    a @ w
torch.cuda.synchronize()
dt = (time.time() - t) / n
print(f"gemv_GBps={w.numel() * 2 / dt / 1e9:.0f}")
