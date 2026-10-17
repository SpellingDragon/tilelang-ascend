# w11 scan1 micro — 8-mode single-shot diagnosis (one process per mode).
#   mode=1: S=0, K/V const, beta=1, h0=0, fp16   (all-finite numeric baseline)
#   mode=2: same, fp32
#   mode=3: random K/V/beta, S=0, fp32           (isolates g magnitude)
#   mode=4: random all, fp32                     (= L0 basic fp32 repro)
#   mode=5: random all, fp16                     (= L0 basic fp16 repro)
#   mode=6: random all + h0!=0, fp32             (h0 path)
#   mode=7: random all, T=128 (2 chunks), fp32   (multi-chunk H chain)
#   mode=8: random all, fp32, run scan1 TWICE on identical inputs and compare
#           nan masks bit-for-bit (determinism / race detector)
# Output: ws_g/ws_a stats + ws_h per-block (H_in vs H_out) stats.
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
from gdn_bwd_v2 import gdn_bwd_v2_scan1, gdn_bwd_v2_cumsum, gdn_bwd_v2_solve_a  # noqa: E402

C, BV, BK = 64, 64, 64
msk_j = torch.tril(torch.ones(C, C, dtype=torch.float32), -1).npu()
idt = torch.eye(C, dtype=torch.float32).npu()


def _stat(name, x):
    f = x.float()
    print(f"CHK-{name} nan={int(torch.isnan(f).sum())} inf={int(torch.isinf(f).sum())} "
          f"absmax={float(f.abs().max()):.3e} mean={float(f.mean()):.3e}", flush=True)


def _stat_blocks(name, h):
    f = h.float()  # [B,H,nchunk+1,DV,DK]
    for ci in range(f.shape[2]):
        blk = f[:, :, ci]
        print(f"CHK-{name}[{ci}] nan={int(torch.isnan(blk).sum())} "
              f"absmax={float(blk.abs().max()):.3e} mean={float(blk.mean()):.3e}", flush=True)


def run(mode):
    T_len = 128 if mode == 7 else 64
    dt = torch.float16 if mode in (1, 5) else torch.float32
    dtype = "float16" if mode in (1, 5) else "float32"
    B, H, DK, DV = 1, 1, 64, 64
    torch.manual_seed(7)
    if mode in (1, 2):
        k = torch.full((B, H, T_len, DK), 0.1, dtype=dt)
        v = torch.full((B, H, T_len, DV), 0.2, dtype=dt)
        q = torch.full((B, H, T_len, DK), 0.3, dtype=dt)
        g = torch.zeros(B, H, T_len)
        beta = torch.full((B, H, T_len), 0.5)
        h0 = torch.zeros(B, H, DV, DK, dtype=dt)
    elif mode == 3:
        k = torch.randn(B, H, T_len, DK).to(dt)
        v = torch.randn(B, H, T_len, DV).to(dt)
        q = torch.randn(B, H, T_len, DK).to(dt)
        g = torch.zeros(B, H, T_len)
        beta = torch.sigmoid(torch.randn(B, H, T_len))
        h0 = torch.zeros(B, H, DV, DK, dtype=dt)
    else:
        k = torch.randn(B, H, T_len, DK).to(dt)
        v = torch.randn(B, H, T_len, DV).to(dt)
        q = torch.randn(B, H, T_len, DK).to(dt)
        g = -(torch.randn(B, H, T_len).abs() * 0.5 + 0.05)
        beta = torch.sigmoid(torch.randn(B, H, T_len))
        h0 = (torch.randn(B, H, DV, DK) * 0.1 if mode == 6
              else torch.zeros(B, H, DV, DK)).to(dt)
    k, v, q, g, beta, h0 = k.npu(), v.npu(), q.npu(), g.npu(), beta.npu(), h0.npu()
    print(f"=== MODE {mode} dtype={dtype} T={T_len} h0={'nonzero' if mode == 6 else 'zero'} ===",
          flush=True)
    ws_g = gdn_bwd_v2_cumsum(B, H, T_len, C)(g)
    ws_a = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, BK, dtype=dtype)(k, beta, ws_g, msk_j, idt)
    torch.npu.synchronize()
    _stat("ws_g", ws_g)
    _stat("ws_a", ws_a)
    ker = gdn_bwd_v2_scan1(B, H, T_len, DK, DV, C, BV, BK, dtype=dtype)
    ws_h = ker(k, v, beta, ws_g, ws_a, h0)
    torch.npu.synchronize()
    _stat_blocks("ws_h", ws_h)
    # golden H-chain: fp64 exact forward recursion (test's _np_fwd_exact)
    from test_gdn_bwd_v2 import _np_fwd_exact
    O, HIST, Sf = _np_fwd_exact(
        H, T_len, DK, DV,
        q[0].cpu().double().numpy(), k[0].cpu().double().numpy(),
        v[0].cpu().double().numpy(), beta[0].cpu().double().numpy(),
        g[0].cpu().double().numpy(), h0[0].cpu().double().numpy())
    got = ws_h[0, :, 1].float().cpu().numpy()  # [H,DV,DK] chunk0-end state
    gerr = np.abs(got - Sf)
    print(f"CHK-golden max_abs_err={float(np.nanmax(gerr)):.3e} "
          f"gold_absmax={float(np.abs(Sf).max()):.3e} "
          f"rel={float(np.nanmax(gerr) / (np.abs(Sf).max() + 1e-30)):.3e}", flush=True)
    if mode == 8:
        ws_h2 = ker(k, v, beta, ws_g, ws_a, h0)
        torch.npu.synchronize()
        f1, f2 = ws_h.float(), ws_h2.float()
        m1, m2 = torch.isnan(f1), torch.isnan(f2)
        same_mask = bool(torch.equal(m1, m2))
        diff = float((f1[~m1] - f2[~m1]).abs().max()) if int((~m1).sum()) else 0.0
        print(f"CHK-determinism nan_mask_equal={same_mask} "
              f"nan1={int(m1.sum())} nan2={int(m2.sum())} max_abs_diff={diff:.3e}", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]))
    print("S1MICRO-DONE")
