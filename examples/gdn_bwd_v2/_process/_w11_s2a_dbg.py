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

    da_l1_card = outs[4].float().cpu().numpy().reshape(64, 64)   # WS_DKT <- GM dqkt
    da_ub_card = outs[6].float().cpu().numpy().reshape(64, 64)   # WS_DGAM2 <- da_ub
    dq_card = outs[0].float().cpu().numpy().reshape(64, 64)
    dqkt_card = da_ub_card
    r1, b1 = rel(da_l1_card, dq_g)        # fresh dqkt@K  vs golden DQ
    r0, b0 = rel(da_ub_card, dqkt_g)      # UB dqkt       vs golden dqkt
    rc, _ = rel(da_l1_card, dq_g)
    r2, b2 = rel(dq_card, dq_g)
    print(f"CARD da_ub absmax={float(np.abs(da_ub_card).max()):.3e} "
          f"da_l1 absmax={float(np.abs(da_l1_card).max()):.3e}")
    print(f"CHK-freshGEMM(dqkt@K) vs golden DQ rel={r1:.3e}")
    print(f"CHK-UB dqkt          vs golden dqkt rel={r0:.3e}")
    print(f"CHK-DQ(out0)         vs golden DQ   rel={r2:.3e}")
    if r1 < 5e-3 and r2 > 5e-2:
        print("VERDICT da_l1-OK => GEMM on da_l1 is fine; the dq0_l0 accumulate/writeback is wrong")
    elif r0 > 5e-2:
        print("VERDICT dqkt-BAD => UB dqkt construction wrong (upstream of the chain)")
    else:
        print("VERDICT mixed => inspect r1/r0")
    print(f"CARD dqkt absmax={float(np.abs(dqkt_card).max()):.3e} "
          f"mean={float(dqkt_card.mean()):.3e} head={np.round(dqkt_card.ravel()[:4], 4).tolist()}")
    print(f"GOLD dqkt absmax={float(np.abs(dqkt_g).max()):.3e} "
          f"mean={float(dqkt_g.mean()):.3e} head={np.round(dqkt_g.ravel()[:4], 4).tolist()}")
    print(f"CARD dq   absmax={float(np.abs(dq_card).max()):.3e} "
          f"head={np.round(dq_card.ravel()[:4], 4).tolist()}")
    print(f"GOLD dq   absmax={float(np.abs(dq_g).max()):.3e} "
          f"head={np.round(dq_g.ravel()[:4], 4).tolist()}")
    print(f"CHK-dqkt card_vs_gold rel={r1:.3e} nonfinite={b1}")
    print(f"CHK-DQ   card_vs_gold rel={r2:.3e} nonfinite={b2}")
    for nm, a, g in (("dqkt", dqkt_card, dqkt_g), ("dq", dq_card, dq_g)):
        nz = np.abs(a).max(axis=0)          # per-column absmax
        ng = np.abs(g).max(axis=0)
        rn = np.abs(a).max(axis=1)          # per-row absmax
        print(f"COLS-{nm} card_nz_cols={int((nz > 0).sum())}/64 gold_nz_cols={int((ng > 0).sum())}/64 "
              f"card_cols[:8]={np.round(nz[:8], 3).tolist()} gold_cols[:8]={np.round(ng[:8], 3).tolist()}")
        print(f"ROWS-{nm} card_nz_rows={int((rn > 0).sum())}/64 "
              f"card_rows[:4]={np.round(rn[:4], 3).tolist()} card_rows[-4:]={np.round(rn[-4:], 3).tolist()}")
        print(f"FULL-{nm} card_rows={np.round(rn, 2).tolist()}")
        print(f"FULLC-{nm} card_cols={np.round(nz, 2).tolist()}")
        print(f"FULL-{nm}-gold rows={np.round(np.abs(g).max(axis=1), 2).tolist()}")
    if r1 < 5e-3:
        print("VERDICT dqkt-OK => GEMM/L1 read path wrong (dqkt is right, DQ is not)")
    else:
        print("VERDICT dqkt-BAD => dqkt construction (dP/Gam/M) wrong")
    print("S2ADBG-DONE")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "float16")
