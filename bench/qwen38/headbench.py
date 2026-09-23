"""Draft-head microbench: BF16 linear vs FP8 rowwise _scaled_mm, reduced vocab 47172 x 2560."""
import time, torch
torch.manual_seed(0)
V, H = 47184, 2560  # 47172 padded to a multiple of 16
w = (torch.randn(V, H, device="cuda") * 0.02).to(torch.bfloat16)
wscale = w.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
w8 = (w.float() / wscale).to(torch.float8_e4m3fn)
def bench(fn, n=300):
    for _ in range(30): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
for M in (1, 4):
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, w).argmax(-1)
    t_bf16 = bench(lambda: torch.nn.functional.linear(x, w).argmax(-1))
    def fp8():
        xs = x.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
        x8 = (x.float() / xs).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(x8, w8.t(), scale_a=xs, scale_b=wscale.t(), out_dtype=torch.bfloat16)
        return out.argmax(-1)
    try:
        agree = (fp8() == ref).float().mean().item()
        t_fp8 = bench(fp8)
    except Exception as e:
        agree, t_fp8 = float("nan"), float("nan"); print("fp8 failed:", type(e).__name__, str(e)[:160])
    print(f"M={M}: bf16 {t_bf16:.3f} ms  fp8 {t_fp8:.3f} ms  argmax agreement {agree:.3f}")
