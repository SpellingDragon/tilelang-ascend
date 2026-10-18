# _w11_scan1_g.py — scan1 H-chain vs fp64 golden on the REAL distribution (g != 0).
#   Previously scan1 was only ever judged with mode1 (g == 0), which hides any
#   error in solve_a's A (the D15 bug made A wrong for g != 0). This leg closes
#   that hole: it feeds the l0_basic fp16 inputs, runs cumsum -> solve_a ->
#   scan1, and compares ws_h[...,1] against the fp64 recursion's Sf.
#   Legs: cumsum -> solve_a -> scan1. Also reports ws_a's own sanity (row sums).
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402

_GOLDEN_ONLY = "--golden-only" in sys.argv

from test_gdn_bwd_v2 import _case_inputs, _np_fwd_exact  # noqa: E402


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float("nan")
    return float(np.abs(a[m] - b[m]).max() / (np.abs(b[m]).max() + 1e-30))


def run(dtype_str="float16"):
    (q, k, v, g, beta, h0, do), _ = _case_inputs(1, 1, 64, 64, 64, dtype_str, "zero", False)
    # fp64 golden H chain (HIST is per-token; chunk0-end state == Sf for T=64)
    # _np_fwd_exact takes [H, T, ...] tensors (batch already stripped).
    O, HIST, Sf = _np_fwd_exact(
        1, 64, 64, 64,
        q[0].numpy().astype(np.float64), k[0].numpy().astype(np.float64),
        v[0].numpy().astype(np.float64), beta[0].numpy().astype(np.float64),
        g[0].numpy().astype(np.float64), h0[0].numpy().astype(np.float64))
    print(f"GOLD H-chain Sf max={float(np.abs(Sf).max()):.3e} finite={np.isfinite(Sf).all()}")
    if _GOLDEN_ONLY:
        print("SCAN1G-GOLDEN-ONLY-DONE")
        return

    from gdn_bwd_v2 import gdn_bwd_v2_cumsum, gdn_bwd_v2_scan1, gdn_bwd_v2_solve_a  # noqa: E402
    from test_gdn_bwd_v2 import _npu  # noqa: E402

    dt = torch.float16 if dtype_str == "float16" else torch.float32
    qn, kn, vn, gn, bn, hn = (_npu(q), _npu(k), _npu(v), _npu(g), _npu(beta), _npu(h0))
    ws_g = gdn_bwd_v2_cumsum(1, 1, 64, 64)(gn)
    msk_j = torch.tril(torch.ones(64, 64, dtype=torch.float32), -1).npu()
    idt = torch.eye(64, dtype=torch.float32).npu()
    ws_a = gdn_bwd_v2_solve_a(1, 1, 64, 64, 64, 64, dtype=dtype_str)(kn, bn, ws_g, msk_j, idt)
    torch.npu.synchronize()
    a_card = ws_a.float().cpu().numpy().reshape(64, 64)
    a_gold = np.linalg.inv(np.eye(64) + np.tril(np.ones((64, 64)), -1)
                           * (beta[0, 0].numpy()[:, None]
                              * (np.exp(np.cumsum(g[0, 0].numpy())[:, None]
                                        - np.cumsum(g[0, 0].numpy())[None, :])
                                 * (k[0, 0].numpy().astype(np.float64) @ k[0, 0].numpy().astype(np.float64).T))))
    print(f"CHK-A  rel={rel(a_card, a_gold):.3e}  (rowsum card[:4]={np.round(a_card.sum(1)[:4], 4).tolist()} "
          f"gold[:4]={np.round(a_gold.sum(1)[:4], 4).tolist()})")

    ws_h = gdn_bwd_v2_scan1(1, 1, 64, 64, 64, 64, 64, 64, dtype=dtype_str)(kn, vn, bn, ws_g, ws_a, hn)
    torch.npu.synchronize()
    h_card = ws_h.float().cpu().numpy()[0, 0, 1]
    print(f"CHK-H  rel={rel(h_card, Sf):.3e}  card_absmax={float(np.abs(h_card).max()):.3e}")
    if rel(h_card, Sf) < 5e-3:
        print("VERDICT scan1-H-OK (solve_a fix holds on the real g != 0 distribution)")
    else:
        print("VERDICT scan1-H-BAD => remaining error inside scan1/solve_a beyond D15")
    print("SCAN1G-DONE")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "float16")
