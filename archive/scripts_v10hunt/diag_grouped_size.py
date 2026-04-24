"""Is te.GroupedLinear output bf16-sensitive to num_gemms (144 vs 4×36)?"""
import torch, torch.nn.functional as F
import transformer_engine.pytorch as te
torch.manual_seed(42)

E_full = 144
E_ep = 36  # per-EP-rank
EP = 4
in_f = 512
out_f = 320  # 2×160

# Weights for all 144 experts
w_full = torch.randn(E_full, out_f, in_f, dtype=torch.float32, device="cuda") / 32

# Random tokens per expert
m_splits_full = [torch.randint(50, 200, (1,)).item() for _ in range(E_full)]
N_full = sum(m_splits_full)
print(f"Total N={N_full}")

x = torch.randn(N_full, in_f, dtype=torch.bfloat16, device="cuda") * 0.2

# Method A: single GroupedLinear with 144 experts
gl_full = te.GroupedLinear(num_gemms=E_full, in_features=in_f, out_features=out_f, bias=False,
                            params_dtype=torch.float32, device="cuda")
for e in range(E_full):
    getattr(gl_full, f"weight{e}").data = w_full[e].clone()

with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out_A = gl_full(x, m_splits=m_splits_full)

# Method B: 4× GroupedLinear with 36 experts each, concatenate
gls = []
for rank in range(EP):
    gl = te.GroupedLinear(num_gemms=E_ep, in_features=in_f, out_features=out_f, bias=False,
                          params_dtype=torch.float32, device="cuda")
    for e in range(E_ep):
        global_e = rank * E_ep + e
        getattr(gl, f"weight{e}").data = w_full[global_e].clone()
    gls.append(gl)

outs = []
start = 0
for rank in range(EP):
    m_splits_rank = m_splits_full[rank*E_ep:(rank+1)*E_ep]
    N_rank = sum(m_splits_rank)
    x_rank = x[start:start+N_rank]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out_rank = gls[rank](x_rank, m_splits=m_splits_rank)
    outs.append(out_rank)
    start += N_rank
out_B = torch.cat(outs, dim=0)

d = (out_A.float() - out_B.float()).abs()
print(f"GroupedLinear(144) vs 4×GroupedLinear(36) concat:")
print(f"  L_inf = {d.max():.3e}")
print(f"  L1 = {d.mean():.3e}")
print(f"  nonzero = {(d>0).sum().item()}/{d.numel()}")
