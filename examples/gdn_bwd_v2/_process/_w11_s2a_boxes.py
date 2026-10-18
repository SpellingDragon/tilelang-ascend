# w11_s2a_boxes.py — scan2a per-box golden compare (single chunk, fp16/fp32)
#   Builds the fp64 chunk cache via test's _np_chunk_fwd, replays B1-B12 for the
#   LAST chunk (dp_rel = 0), and compares each scan2a output box.
#   Box map (gdn_bwd_v2.py scan2a out_idx=[11..19]):
#     out0 DQ, out1 DH0, out2 WS_DV, out3 WS_DW, out4 WS_DKT,
#     out5 WS_DE, out6 WS_DGAM2, out7 WS_DGAM, out8 WS_DK
#   Input modes (discriminating experiments):
#     normal  : test distribution
#     g0      : g == 0  -> Gam == 1 (kills any Gam error)
#     gconst  : g == -0.1 -> Gam = exp(-0.1*(r-c)); a sign flip shows as e^6.3
#   Usage: python _w11_s2a_boxes.py <dtype> [normal|g0|gconst] [--golden-only]
#   --golden-only runs the fp64 side only (no tilelang / NPU needed).
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from test_gdn_bwd_v2 import _case_inputs, _np_chunk_fwd  # noqa: E402


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    bad = int((~np.isfinite(a)).sum())
    if mask.sum() == 0:
        return float("nan"), bad
    d = np.abs(a[mask] - b[mask])
    s = np.abs(b[mask]).max() + 1e-30
    return float(d.max() / s), bad


def run(dtype_str, imode="normal", golden_only=False, h0mode="zero"):
    B, H, T_len, DK, DV, C = 1, 1, 64, 64, 64, 64
    (q, k, v, g, beta, h0, do), _gold = _case_inputs(B, H, T_len, DK, DV, dtype_str, h0mode, False)
    if imode == "g0":
        g = torch.zeros_like(g)
    elif imode == "gconst":
        g = torch.full_like(g, -0.1)
    # fp64 chunk cache (single chunk); tensors are [B,H,T,*] -> take [0,0]
    Q, K, V = q[0, 0].numpy(), k[0, 0].numpy(), v[0, 0].numpy()
    BETA, G = beta[0, 0].numpy(), g[0, 0].numpy()
    H0 = h0[0, 0].numpy()
    dO = do[0, 0].numpy()
    _, cache = _np_chunk_fwd(T_len, DK, DV, Q, K, V, BETA, G, H0, C)
    c = cache[0]
    gam_2d, kk, msk_m, msk_j = c["gam_2d"], c["kk"], c["msk_m"], c["msk_j"]
    a_mat, e_vec, glast, th = c["a_mat"], c["e_vec"], c["glast"], c["th"]
    kt, vnew, qh, hin, w_mat, p_mat = c["kt"], c["vnew"], c["qh"], c["hin"], c["w_mat"], c["p_mat"]
    raw_qkt = Q @ K.T
    dp_rel = np.zeros((DV, DK))
    # ---- scan2a's ACTUAL computation (single chunk, BV=DV, bk_num=1) ----
    u0 = a_mat @ (BETA[:, None] * V)
    dvnew = p_mat.T @ dO                               # B1 (Delta'=0)
    vnew_s2a = u0 - w_mat @ hin.T                       # = cache vnew
    qh = Q @ hin.T
    de_s2a = (dO * qh).sum(1)                           # B5 only (B8 lives in s2b)
    e_do = e_vec[:, None] * dO
    dp = dO @ vnew_s2a.T                                # B2
    dqkt = dp * gam_2d * msk_m                          # dQKt
    dq_s2a = dqkt @ K + e_do @ hin                      # B4 + B5b
    dh_kernel = e_do.T @ Q - vnew_s2a.T @ w_mat         # as coded (vnew^T (-W))
    dh_math = e_do.T @ Q - dvnew.T @ w_mat              # B5c + B6 (chain: (-W)^T dVnew)
    dw_s2a = -(dvnew @ hin)                             # B6 (neg folded at write-back)
    dgam2_s2a = (dp * msk_m) * raw_qkt                  # B3 only (B11 merges in s2b)
    dk_s2a = dqkt.T @ Q                                 # B4' (B10's th*dkt merges in s2b)
    dkt_s2a = np.zeros((C, DK))                         # Delta'=0
    dgam_s2a = np.zeros((C,))                           # B9 scalar only (Delta'=0)

    gold = {
        "out0_DQ": dq_s2a, "out1_DH0": dh_kernel, "out2_WS_DV": dvnew,
        "out3_WS_DW": dw_s2a, "out4_WS_DKT": dkt_s2a, "out5_WS_DE": de_s2a,
        "out6_WS_DGAM2": dgam2_s2a, "out7_WS_DGAM": dgam_s2a, "out8_WS_DK": dk_s2a,
    }
    alt = {"out1_DH0": dh_math}
    print(f"=== S2A-BOX {dtype_str} imode={imode} h0={h0mode} ===")
    if golden_only:
        for name, gv in gold.items():
            print(f"GOLD {name:14s} max={float(np.abs(gv).max()):.3e} "
                  f"nonfinite={int((~np.isfinite(gv)).sum())}")
        print("S2ABOX-GOLDEN-ONLY-DONE")
        return

    # ---- kernel side (lazy: tilelang + NPU only needed here) ----
    from gdn_bwd_v2 import gdn_bwd_v2_scan2a, gdn_bwd_v2_scan1, gdn_bwd_v2_solve_a, gdn_bwd_v2_cumsum  # noqa: E402
    from test_gdn_bwd_v2 import _npu  # noqa: E402
    dt = torch.float16 if dtype_str == "float16" else torch.float32
    qn, kn, vn, gn, bn, hn, don = (_npu(q), _npu(k), _npu(v), _npu(g), _npu(beta), _npu(h0), _npu(do))
    ws_g, ws_gam = gdn_bwd_v2_cumsum(B, H, T_len, C)(gn)
    ws_dq1 = torch.empty(B * H, C, C, dtype=torch.float32).npu()
    ws_dq0 = torch.empty(B * H, C, C, dtype=torch.float32).npu()
    msk_m_t = torch.tril(torch.ones(64, 64, dtype=torch.float32)).npu()
    msk_j_t = torch.tril(torch.ones(64, 64, dtype=torch.float32), -1).npu()
    idt = torch.eye(64, dtype=torch.float32).npu()
    ones_dv = torch.ones(DV, 16, dtype=dt).npu()
    ws_a = gdn_bwd_v2_solve_a(B, H, T_len, DK, C, 64, dtype=dtype_str)(kn, bn, ws_g, msk_j_t, idt)
    ws_h = gdn_bwd_v2_scan1(B, H, T_len, DK, DV, C, 64, 64, dtype=dtype_str)(kn, vn, bn, ws_g, ws_a, hn)
    outs = gdn_bwd_v2_scan2a(B, H, T_len, DK, DV, C, 64, 64, dtype=dtype_str)(
        qn, kn, vn, don, bn, ws_g, ws_a, msk_m_t, ones_dv, ws_h, ws_gam, ws_dq1, ws_dq0)
    torch.npu.synchronize()

    for i, name in enumerate(gold):
        got = outs[i].float().cpu().numpy().reshape(gold[name].shape)
        r, bad = rel(got, gold[name])
        line = f"{name:14s} rel={r:.3e} nonfinite={bad} gold_max={float(np.abs(gold[name]).max()):.3e}"
        if name in alt:
            r2, _ = rel(got, alt[name])
            line += f"  [alt(dvnew-form) rel={r2:.3e}]"
        print(line)
    print("S2ABOX-DONE")


if __name__ == "__main__":
    dt = sys.argv[1] if len(sys.argv) > 1 else "float16"
    im = sys.argv[2] if len(sys.argv) > 2 else "normal"
    hm = sys.argv[3] if len(sys.argv) > 3 else "zero"
    go = "--golden-only" in sys.argv
    run(dt, im, go, hm)
