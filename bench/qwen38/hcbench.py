"""HC projection microbench: BF16 linear vs FP8 rowwise _scaled_mm at decode shapes."""
import time, torch
def bench(fn, n=500):
    for _ in range(50): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e6
for (N, K) in ((336, 10240), (10240, 320)):
    w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    s = w.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
    w8 = (w.float() / s).to(torch.float8_e4m3fn); st = s.t().contiguous()
    for M in (4, 16):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        tb = bench(lambda: torch.nn.functional.linear(x, w))
        def f8():
            xf = x.float(); xs = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
            return torch._scaled_mm((xf / xs).to(torch.float8_e4m3fn), w8.t(), scale_a=xs, scale_b=st, out_dtype=torch.bfloat16)
        tf = bench(f8)
        err = (f8().float() - torch.nn.functional.linear(x, w).float()).abs().mean() / torch.nn.functional.linear(x, w).float().abs().mean()
        print(f"N={N:5d} K={K:5d} M={M:2d}: bf16 {tb:6.1f} us  fp8 {tf:6.1f} us  rel_err {err:.4f}")
