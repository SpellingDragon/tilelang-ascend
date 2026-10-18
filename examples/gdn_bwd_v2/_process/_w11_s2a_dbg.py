# _w11_s2a_dbg.py — one-run split: "dqkt construction wrong" vs "GEMM/L1 path wrong".
#   Runs scan2a with GDN_S2A_DBG=1, so its WS_DKT output (out4) carries dqkt
#   (da_ub) instead of the (single-chunk-zero) dKt box.
#   Reads back:
#     out4 = dqkt(card)      vs golden dqkt      -> construction check
#     out0 = DQ(card)        vs golden DQ        -> end-to-end
#   Verdict table:
#     dqkt ~1e-3 (ok) and DQ ~1.0  => the GEMM/L1 read path is wrong
#     dqkt ~1.0 or 4e-2 (bad)      => dqkt construction (dP/Gam/M) is wrong
#   Usage: GDN_S2A_DBG=1 python _w11_s2a_dbg.py [float16] [--golden-only]
import os

os.environ["GDN_S2A_DBG"] = "1"          # must precede the gdn_bwd_v2 import

import sys  # noqa: E402

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402

_GOLDEN_ONLY = "--golden-only" in sys.argv

from test_gdn_bwd_v2 import _case_inputs, _np_chunk_fwd  # noqa: E402


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    bad = int((~np.isfinite(a)).sum())
    if m.sum() == 0:
        return float("nan"), bad
    return float(np.abs(a[m] - b[m]).max() / (np.abs(b[m]).max() + 1e-30)), bad


def golden(dtype_str="float16"):
    (q, k, v, g, beta, h0, do), _ = _case_inputs(1, 1, 64, 64, 64, dtype_str, "zero", False)
    Q, K, V = q[0, 0].numpy(), k[0, 0].numpy(), v[0, 0].numpy()
    BETA, G = beta[0, 0].numpy(), g[0, 0].numpy()
    dO = do[0, 0].numpy()
    _, cache = _np_chunk_fwd(64, 64, 64, Q, K, V, BETA, G, h0[0, 0].numpy(), 64)
    c = cache[0]
    raw = Q @ K.T
    dp = dO @ c["vnew"].T
    dqkt = dp * c["gam_2d"] * c["msk_m"]
    e_do = c["e_vec"][:, None] * dO
    dq = dqkt @ K + e_do @ c["hin"]        # hin = 0 here, kept for the formula
    return (q, k, v, g, beta, h0, do), dqkt, dq


def run(dtype_str="float16"):
    (q, k, v, g, beta, h0, do), dqkt_g, dq_g = golden(dtype_str)
    print(f"=== S2A-DBG {dtype_str} (GDN_S2A_DBG=1, dqkt via WS_DKT) ===")
    print(f"GOLD dqkt max={float(np.abs(dqkt_g).max()):.3e} dq max={float(np.abs(dq_g).max()):.3e}")
    if _GOLDEN_ONLY:
        print("S2ADBG-GOLDEN-ONLY-DONE")
        return

    from gdn_bwd_v2 import (  # noqa: E402
        gdn_bwd_v2_cumsum, gdn_bwd_v2_scan1, gdn_bwd_v2_scan2a, gdn_bwd_v2_solve_a,
    )
    from test_gdn_bwd_v2 import _npu  # noqa: E402

    dt = torch.float16 if dtype_str == "float16" else torch.float32
    qn, kn, vn, gn, bn, hn, don = (_npu(q), _npu(k), _npu(v), _npu(g), _npu(beta), _npu(h0), _npu(do))
    ws_g = gdn_bwd_v2_cumsum(1, 1, 64, 64)(gn)
    msk_m = torch.tril(torch.ones(64, 64, dtype=torch.float32)).npu()
    msk_j = torch.tril(torch.ones(64, 64, dtype=torch.float32), -1).npu()
    idt = torch.eye(64, dtype=torch.float32).npu()
    ones_dv = torch.ones(64, 16, dtype=dt).npu()
    ws_a = gdn_bwd_v2_solve_a(1, 1, 64, 64, 64, 64, dtype=dtype_str)(kn, bn, ws_g, msk_j, idt)
    ws_h = gdn_bwd_v2_scan1(1, 1, 64, 64, 64, 64, 64, 64, dtype=dtype_str)(kn, vn, bn, ws_g, ws_a, hn)
    outs = gdn_bwd_v2_scan2a(1, 1, 64, 64, 64, 64, 64, 64, dtype=dtype_str)(
        qn, kn, vn, don, bn, ws_g, ws_a, msk_m, ones_dv, ws_h)
    torch.npu.synchronize()

    dqkt_card = outs[4].float().cpu().numpy().reshape(64, 64)   # WS_DKT  <- dqkt
    gam_card = outs[6].float().cpu().numpy().reshape(64, 64)    # WS_DGAM2 <- Gam
    dq_card = outs[0].float().cpu().numpy().reshape(64, 64)
    from test_gdn_bwd_v2 import _case_inputs, _np_chunk_fwd  # noqa: E402,F401
    # golden Gam
    (q2, k2, v2, g2, beta2, h02, do2), _ = _case_inputs(1, 1, 64, 64, 64, dtype_str, "zero", False)
    _, cache = _np_chunk_fwd(64, 64, 64, q2[0, 0].numpy(), k2[0, 0].numpy(), v2[0, 0].numpy(),
                             beta2[0, 0].numpy(), g2[0, 0].numpy(), h02[0, 0].numpy(), 64)
    gam_g = cache[0]["gam_2d"]
    r0, b0 = rel(dqkt_card, dqkt_g)
    rg, _ = rel(gam_card, gam_g)
    r2, b2 = rel(dq_card, dq_g)

    def cols(a):
        nz = np.abs(a).max(axis=0)
        return int((nz > 0).sum()), np.round(nz[:20], 3).tolist()

    nz_d, head_d = cols(dqkt_card)
    nz_g, head_g = cols(gam_card)
    nz_G, head_G = cols(dq_g)
    print(f"COLS-dqkt  nz={nz_d}/64 cols[:20]={head_d}  rel_vs_gold={r0:.3e}")
    print(f"COLS-Gam   nz={nz_g}/64 cols[:20]={head_g}  rel_vs_gold={rg:.3e}")
    print(f"COLS-DQ    nz={nz_G}/64")  # golden reference only
    print(f"CHK-DQ     rel={r2:.3e} nonfinite={b2}")
    if nz_g < 64:
        print("VERDICT Gam-partial => 2D Parallel exp covers only part of the columns (D9)")
    elif r2 > 5e-2:
        print("VERDICT Gam-full but DQ still wrong => L0C->GM realTailN (D13) or GEMM")
    else:
        print("VERDICT all-good")
    print("S2ADBG-DONE")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "float16")
